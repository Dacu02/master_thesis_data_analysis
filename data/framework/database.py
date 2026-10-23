import sys
import os

CUT_STRING = "src/master_thesis_data_analysis/data"

sys.path.append(os.path.join(os.getcwd().split(CUT_STRING)[0], CUT_STRING))
from framework import *

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks
from scipy.optimize import brentq
from scipy.special import erf, erfinv
from scipy.integrate import cumulative_trapezoid
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------ Input files
CSV_ROOT = os.path.join(os.getcwd()).split('src')[0] + "/recording_csv"
CSV_FILES = sorted(Path(CSV_ROOT).glob("**/*.csv"))
YAML_PATHS_FILE = os.path.join(os.getcwd()).split('src')[0] + "/src/unisa_acg_ros2/haptics/haptic_interaction_teach_and_record_demo/config/paths.yaml"
OUTPUT_ROOT = os.path.join(os.getcwd()).split('src')[0] + "/reconstructed_csv"
DEBUG = False
POSITION_COLUMNS = ["position_x", "position_y", "position_z"]
VELOCITY_COLUMNS = ["twist_linear_x", "twist_linear_y", "twist_linear_z"]
GEOMETRY_OUTPUT_COLUMNS = ['lambda_s', 'u_s', 'lambda_e', 'u_e', 'delta_p']

# ------------------------------------------------- Low-pass Chebyshev II filter
FILTER_ORDER = 11
FILTER_CUTOFF_HZ = 16
FILTER_STOPBAND_DB = 80
# ----------------------------------------------------------- Resampling
TARGET_FREQUENCY = 500  # Hz

# ------------------------------------------ Waypoints: normalized -> device frame
SWAP_YZ_WAYPOINTS = False
WS_MIN = np.array([-0.082, -0.065, -0.085])
WS_MAX = np.array([0.082, 0.065, 0.085])

# ---------------------------------------------------------------- Lognormal fit
FIT_OPTIONS = dict(
    sigma_bounds=(0.05, 1.75),  # allowed range for the lognormal sigma
    max_lead=1.0,              # t0 may precede the window start by max_lead * window duration
)

# ------------------------------------------------------------------------ Lengths
LOCK_D_TO_GEOMETRY = True  # D of each stroke = measured path length between its two speed minima

# ------------------------------------------------------- Geometric reconstruction
ANGLE_STEP_RAD = np.deg2rad(1.0)  # angular sensitivity of the exit/entry angles
ANGLE_LIMITS = (-np.pi, np.pi)
DIRECTION_DISTANCE = 7e-3          # m: radius of the neighbourhood used for the junction tangents

# ------------------------------------------ Model speed trimmed by travelled distance
TRIM_START_M = 0.001     # 1 mm
TRIM_END_M = 0.001       # 1 mm

# ---------------------------------------------------------------- Stroke detection
TIME_DISTANCE = .1 #s
PROMINENCE = .005 #m/s
VEL_THRESHOLD = 0.2 #m/s
MAX_ENDPOINT_REL_ERROR = None # maximum relative error of the stroke end points with respect to the nominal segment


def load_waypoints(yaml_file, task):
    with open(yaml_file, "r") as f:
        paths = yaml.safe_load(f)["paths"]
    for path in paths:
        if path["name"] == task:
            return path["waypoints"]
    raise KeyError(f"Task '{task}' not found in {yaml_file}")


def waypoints_to_array(waypoints, swap_yz=False, ws_min=None, ws_max=None):
    """Convert the list of {'x','y','z'} dicts (normalized) into an Nx3 array in device coordinates."""
    arr = np.array([[w["x"], w["y"], w["z"]] for w in waypoints], dtype=float)
    if swap_yz:
        arr[:, [1, 2]] = arr[:, [2, 1]]
    if ws_min is not None and ws_max is not None:
        ws_min = np.asarray(ws_min, dtype=float)
        ws_max = np.asarray(ws_max, dtype=float)
        arr = ws_min + arr * (ws_max - ws_min)
    return arr

def find_minima_maxima(speed:np.ndarray, samples_distance:float|None=None, prominence:float|None=None) -> tuple[list[int], list[int]]:
    """Find the indices of the speed minima and maxima in the given speed array."""
    # Find local maxima
    maxima_indices, _ = find_peaks(speed, distance=samples_distance, prominence=prominence)
    # Find local minima by inverting the speed
    minima_indices, _ = find_peaks(-speed, distance=samples_distance, prominence=prominence)
    try:
        if minima_indices[0] > maxima_indices[0]:
            if speed[0] < speed[maxima_indices[0]]:
                minima_indices = np.r_[0, minima_indices]
            else:
                if speed[0] > speed[minima_indices[0]]:
                    maxima_indices = np.r_[0, maxima_indices]
            
        if minima_indices[-1] < maxima_indices[-1]:
            if speed[-1] < speed[maxima_indices[-1]]:
                minima_indices = np.r_[minima_indices, speed.size - 1]
            else:
                if speed[-1] > speed[minima_indices[-1]]:
                    maxima_indices = np.r_[maxima_indices, speed.size - 1]
    except IndexError:
        if len(minima_indices) == 0 and len(maxima_indices) == 1:
            minima_indices = np.array([0, speed.size - 1])
    return minima_indices, maxima_indices # type: ignore

def lognormal_cdf(t, t0, mu, sigma):
    """Fraction of the stroke length travelled at time t (zero before t0).

    t can be a scalar while t0, mu and sigma are arrays (one value per stroke).
    """
    tau = np.asarray(t, dtype=float) - np.asarray(t0, dtype=float)
    out = np.zeros_like(tau)
    positive = tau > 0
    mu = np.broadcast_to(mu, tau.shape)
    sigma = np.broadcast_to(sigma, tau.shape)
    out[positive] = 0.5 * (
        1.0 + erf((np.log(tau[positive]) - mu[positive]) / (sigma[positive] * np.sqrt(2.0)))
    )
    return out

def trim_strokes(strokes_to_trim, trim_start_m, trim_end_m=None) -> tuple[np.ndarray, tuple[float, float]]:
    """Trim the first and the last metres travelled by the strokes and shift the onsets.

    The distance travelled at time t is s(t) = sum_k D_k * Phi_k(t), where Phi_k is the
    lognormal cumulative of stroke k. The trim times are defined by
        s(t_start) = trim_start_m
        s(t_end)   = sum_k D_k - trim_end_m
    and every onset is shifted so that t = 0 corresponds to t_start.

    Parameters
    ----------
    strokes : sequence of objects with attributes `to` (onset t0), `mu`, `sigma` and `D`.
    trim_start_m : distance (m) removed at the beginning.
    trim_end_m : distance (m) removed at the end. If None, the same value is used.

    Returns
    -------
    new_t0 : np.ndarray
        Shifted onsets, one per stroke. The first onset is usually negative: the first
        stroke starts before `trim_start_m` has been travelled.
    (t_start, t_end) : tuple of float
        Trim times on the original time axis. The end trim cannot be encoded in the onsets:
        the trimmed signal lasts `t_end - t_start` on the shifted axis (t_end = inf if
        trim_end_m == 0).
    """
    if trim_end_m is None:
        trim_end_m = trim_start_m
    if trim_start_m < 0 or trim_end_m < 0:
        raise ValueError("The trim distances must be non-negative.")

    t0 = np.array([float(s.to) for s in strokes_to_trim])
    mu = np.array([float(s.mu) for s in strokes_to_trim])
    sigma = np.array([float(s.sigma) for s in strokes_to_trim])
    D = np.array([abs(float(s.D)) for s in strokes_to_trim])

    total = D.sum()
    if trim_start_m + trim_end_m >= total:
        raise ValueError(
            f"The trim distances ({trim_start_m + trim_end_m:.6g} m) must be smaller than "
            f"the total distance ({total:.6g} m)."
        )

    def distance(t):
        return float(np.sum(D * lognormal_cdf(t, t0, mu, sigma)))

    # Bracket: s = 0 at the first onset, and every stroke is complete at t_hi
    t_lo = t0.min()
    t_hi = (t0 + np.exp(mu + 8.0 * sigma)).max()

    if trim_start_m == 0:
        t_start = t_lo
    else:
        t_start = brentq(lambda t: distance(t) - trim_start_m, t_lo, t_hi, xtol=1e-12)

    if trim_end_m == 0:
        t_end = np.inf
    else:
        t_end = brentq(lambda t: distance(t) - (total - trim_end_m), t_lo, t_hi, xtol=1e-12)

    return t0 - t_start, (float(t_start), float(t_end)) # type: ignore

def path_fraction(t_seg, speed_seg):
  cumulative = cumulative_trapezoid(speed_seg, t_seg, initial=0.0)
  if cumulative[-1] <= 0:
    raise ValueError("The model speed is zero over this segment.")
  return cumulative / cumulative[-1], cumulative[-1]

def process_file(file):

    ls = LeastSquares(lock_D=LOCK_D_TO_GEOMETRY, refine=False)
    approx = BezierOptimization()
    filter = Chebyshev(order=FILTER_ORDER, cutoff=FILTER_CUTOFF_HZ, stopband_attenuation=FILTER_STOPBAND_DB)
    resampler = Resample(target_fs=TARGET_FREQUENCY)

    try:
        subject, task, recording_id = file.parts[-3], file.parts[-2], int(file.stem.split("_")[-1])
        waypoints = load_waypoints(YAML_PATHS_FILE, task)
        #waypoints_xyz = waypoints_to_array(waypoints, SWAP_YZ_WAYPOINTS, WS_MIN, WS_MAX)

        csv_data = pd.read_csv(file)
        csv_data['timestamp'] = csv_data['timestamp'] - csv_data['timestamp'].iloc[0]  # normalize timestamps to start at 0
        csv_data['timestamp'] = csv_data['timestamp'] / 1e9  # convert from nanoseconds to seconds

        recording = Recording(
            trajectory=Trajectory(
                vx  = csv_data['twist_linear_x'].to_numpy(),
                vy  = csv_data['twist_linear_y'].to_numpy(),
                vz  = csv_data['twist_linear_z'].to_numpy(),
            ),
            curve=Curve(
                x   = csv_data['position_x'].to_numpy(),
                y   = csv_data['position_y'].to_numpy(),
                z   = csv_data['position_z'].to_numpy(),
            ),
            t   = csv_data['timestamp'].to_numpy()
        )

        recording.trajectory = resampler.apply(filter.apply(recording.trajectory)) # type: ignore
        recording.curve = resampler.apply(filter.apply(recording.curve)) # type: ignore

        minima_idx, maxima_idx = find_minima_maxima(recording.trajectory.speed, samples_distance=int(TIME_DISTANCE * TARGET_FREQUENCY), prominence=PROMINENCE)
        ls.set_boundaries(minima_idx)
        strokes = ls.approximate(recording)
        t = recording.t

        # Contiguous strokes whose peak speed reaches the threshold: one per segment of the task
        filtered_strokes, corrective_last_stroke = strokes.filter_strokes_by_peak(VEL_THRESHOLD)
        n_segments = len(waypoints) - 1
        if len(filtered_strokes) != n_segments:
            if DEBUG:
                plot_filtered_strokes(recording, strokes, filtered_strokes, f"{subject} {task} {file.name}")
            return (file, "warning", f"Number of filtered strokes ({len(filtered_strokes)}) does not match number of segments ({n_segments}).")

        # Stroke k spans the window [minima_idx[k], minima_idx[k + 1]]
        first = strokes.index(filtered_strokes[0])
        minima_idx_strokes = [int(i) for i in minima_idx[first:first + len(filtered_strokes) + 1]]

        # Geometry: one Bezier per window, arc length = D of the stroke (shared with the lognormal amplitude)
        Ds = [float(stroke.D) for stroke in filtered_strokes]
        approx.set_problem(minima_idx=minima_idx_strokes, Ds=Ds, lambda_min=1e-3, lambda_max=None,
                        direction_distance=DIRECTION_DISTANCE)
        curve_parameters, bezier = approx.approximate(recording.curve)

        # Each stroke must reach (approximately) the waypoint of its segment, in the device frame
        if MAX_ENDPOINT_REL_ERROR is not None:
            waypoints_xyz = waypoints_to_array(waypoints, SWAP_YZ_WAYPOINTS, WS_MIN, WS_MAX)
            nominal_steps = np.diff(waypoints_xyz, axis=0)
            rel_errors = [np.linalg.norm(p["delta_p"] - step) / np.linalg.norm(step)
                        for p, step in zip(curve_parameters, nominal_steps)]
            if max(rel_errors) > MAX_ENDPOINT_REL_ERROR:
                return (file, "warning", f"Stroke end points far from their segments (relative errors {np.round(rel_errors, 2).tolist()}).")

        # Speed quality of the kept strokes; shape error of the Bezier chain at equal curvilinear abscissa
        kept_speed = StrokeSequence(filtered_strokes).v_array(t)
        hl_speed = HL(recording.trajectory.speed, kept_speed, len(filtered_strokes), n_segments)

        i0, i1 = minima_idx_strokes[0], minima_idx_strokes[-1]
        curve_cropped = Curve(x=recording.curve.x[i0:i1 + 1], y=recording.curve.y[i0:i1 + 1], z=recording.curve.z[i0:i1 + 1])
        mse_curve, max_err = curve_cropped.mse(bezier)

        output_file = os.path.join(OUTPUT_ROOT, subject, task, f"recording_{recording_id}.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            f.write(f"stroke_id,t0,mu,sigma,D,{','.join(GEOMETRY_OUTPUT_COLUMNS)}\n")
            for k, (stroke, params) in enumerate(zip(filtered_strokes, curve_parameters)):
                values = [str(k), str(stroke.to), str(stroke.mu), str(stroke.sigma), str(stroke.D)]
                for key in GEOMETRY_OUTPUT_COLUMNS:
                    value = params[key]
                    values.append(";".join(map(str, value)) if isinstance(value, (np.ndarray, list, tuple)) else str(value))
                f.write(",".join(values) + "\n")
        return (file, "ok", {"hl_speed": hl_speed, "mse": mse_curve, "max_err": max_err})
    except Exception as e:
        if DEBUG:
            raise e
        return (file, "error", str(e))

if __name__ == '__main__':
    results = []
    if not DEBUG:
        with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as pool: # type: ignore
            futures = {pool.submit(process_file, f): f for f in CSV_FILES}
            for fut in as_completed(futures):
                file, status, msg = fut.result()
                print(f"{str(file).split('recording_csv')[-1]}: {status} {msg}")
                if status == "ok":
                    results.append(msg)
    else:
        for f in CSV_FILES:
            file, status, msg = process_file(f)
            print(f"{str(file).split('recording_csv')[-1]}: {status} {msg}")
            if status == "ok":
                results.append(msg)

    if results:
        mse_values = [result["mse"] for result in results]
        hl_values = [result["hl_speed"] for result in results]
        print(f"Average curve MSE: {np.mean(mse_values)}")
        print(f"Average HL speed: {np.mean(hl_values)}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(mse_values, bins="auto", edgecolor="black")
        axes[0].set_title("Reconstructed vs Recorded Curve MSE")
        axes[0].set_xlabel("MSE")
        axes[0].set_ylabel("Frequency")

        axes[1].hist(hl_values, bins="auto", edgecolor="black")
        axes[1].set_title("Reconstructed vs Recorded Speed SNR")
        axes[1].set_xlabel("SNR")
        axes[1].set_ylabel("Frequency")

        fig.tight_layout()
        plt.show()