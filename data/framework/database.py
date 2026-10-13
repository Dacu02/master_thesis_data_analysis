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
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------ Input files
CSV_ROOT = os.path.join(os.getcwd()).split('src')[0] + "recording_csv"
CSV_FILES = sorted(Path(CSV_ROOT).glob("**/*.csv"))
YAML_PATHS_FILE = os.path.join(os.getcwd()).split('src')[0] + "src/unisa_acg_ros2/haptics/haptic_interaction_teach_and_record_demo/config/paths.yaml"
OUTPUT_ROOT = os.path.join(os.getcwd()).split('src')[0] + "reconstructed_csv"
DEBUG = False

POSITION_COLUMNS = ["position_x", "position_y", "position_z"]
VELOCITY_COLUMNS = ["twist_linear_x", "twist_linear_y", "twist_linear_z"]

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

# --------------------------------------------------------------- Joint refinement
REFINE_OPTIONS = dict(
    t0_shift_frac=0.5,  # t0 may move by +/- this fraction of the stroke window duration
    mu_shift=1.0,       # mu may move by +/- this amount
    reg_weight=0.0,     # > 0 pulls the parameters toward the sequential solution
)

# ------------------------------------------------------------------------ Lengths
LOCK_D_TO_GEOMETRY = True  # D of each stroke = measured path length between its two speed minima

# ------------------------------------------------------- Geometric reconstruction
ANGLE_STEP_RAD = np.deg2rad(1.0)  # angular sensitivity of the exit/entry angles
ANGLE_LIMITS = (-np.pi, np.pi)

# ------------------------------------------ Model speed trimmed by travelled distance
TRIM_START_M = 0.001     # 1 mm
TRIM_END_M = 0.001       # 1 mm

# ---------------------------------------------------------------- Stroke detection
TIME_DISTANCE = .1 #s
PROMINENCE = .005 #m/s
VEL_THRESHOLD = 0.2 #m/s

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

def lognormal_peak(stroke):
    """Time and value of the speed maximum of a single lognormal stroke."""
    t0 = float(stroke.to)
    mu = float(stroke.mu)
    sigma = float(stroke.sigma)
    D = abs(float(stroke.D))

    t_peak = t0 + np.exp(mu - sigma**2)
    v_peak = D / (sigma * np.sqrt(2 * np.pi)) * np.exp(-mu + sigma**2 / 2)
    return t_peak, v_peak


def lognormal_time_at_fraction(stroke, fraction):
    """Return the time at which a stroke reaches a fraction of its distance."""
    return float(stroke.to) + np.exp(
        float(stroke.mu)
        + float(stroke.sigma) * np.sqrt(2.0) * erfinv(2.0 * fraction - 1.0)
    )


def filter_strokes_by_peak(strokes, velocity_threshold):
    """Skip the weak strokes at the beginning, keep the strong ones, stop at the first weak one after them.

    Returns the kept strokes and the first weak stroke found after them (corrective last stroke), or None.
    """
    filtered, corrective = [], None
    for stroke in strokes:
        _, v_peak = lognormal_peak(stroke)
        if v_peak >= velocity_threshold:
            filtered.append(stroke)
        elif filtered:
            corrective = stroke
            break
    return filtered, corrective



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

def process_file(file):

    ls = LeastSquares(lock_D=LOCK_D_TO_GEOMETRY, refine=False)
    approx = CubicBezierC1Approximation()
    filter = Chebyshev(order=FILTER_ORDER, cutoff=FILTER_CUTOFF_HZ, stopband_attenuation=FILTER_STOPBAND_DB)
    resampler = Resample(target_fs=TARGET_FREQUENCY)

    try:
        subject, task, recording_id = file.parts[-3], file.parts[-2], int(file.stem.split("_")[-1])
        waypoints = load_waypoints(YAML_PATHS_FILE, task)
        waypoints_xyz = waypoints_to_array(waypoints, SWAP_YZ_WAYPOINTS, WS_MIN, WS_MAX)

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

        minima_idx, maxima_idx = find_minima_maxima(recording.trajectory.speed, samples_distance=int(TIME_DISTANCE * TARGET_FREQUENCY), prominence = PROMINENCE)
        ls.set_boundaries(minima_idx)
        strokes = ls.approximate(recording)
        
        t = recording.t
        reconstructed_speed = np.zeros_like(t, dtype=float)
        stroke_densities = []

        for stroke in strokes:
            t0 = float(stroke.to)
            mu = float(stroke.mu)
            sigma = float(stroke.sigma)
            D = abs(float(stroke.D))

            tau = t - t0
            valid = tau > 0
            density = np.zeros_like(t, dtype=float)

            density[valid] = (
                D
                / (sigma * np.sqrt(2 * np.pi) * tau[valid])
                * np.exp(-((np.log(tau[valid]) - mu) ** 2) / (2 * sigma**2))
            )

            reconstructed_speed += density
            stroke_densities.append(density)

        filtered_strokes = []
        corrective_last_stroke = None

        filtered_strokes, corrective_last_stroke = filter_strokes_by_peak(strokes, VEL_THRESHOLD)

        Ds = [stroke.D for stroke in strokes]

        approx.set_problem(trajectory=recording.trajectory, boundaries=minima_idx, D=Ds) # type: ignore
        
        curve_parameters, bezier = approx.approximate(recording.curve)

        # HL indexes
        if len(filtered_strokes) != len(waypoints) - 1:
            if not DEBUG:
                return (file, "warning", f"Number of filtered strokes ({len(filtered_strokes)}) does not match number of waypoints ({len(waypoints) - 1}).")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(recording.t, recording.trajectory.speed, color="black",
                linewidth=1.5, label="Recorded speed")
            ax.plot(recording.t, reconstructed_speed, color="tab:blue",
                linewidth=1.5, label="Reconstructed velocity")

            filtered_indices = {
                strokes.index(stroke) for stroke in filtered_strokes
            }
            for stroke_index, density in enumerate(stroke_densities):
                filtered_out = stroke_index not in filtered_indices
                color = "tab:red" if filtered_out else "tab:green"
                label = (f"Stroke {stroke_index + 1} (filtered out)"
                    if filtered_out else f"Stroke {stroke_index + 1}")
                ax.plot(recording.t, density, color=color, linewidth=1.0,
                    label=label)
                ax.fill_between(recording.t, 0, density, color=color, alpha=0.18)

            ax.axhline(VEL_THRESHOLD, color="tab:orange", linestyle="--",
                linewidth=1.5, label="Velocity filter")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Velocity / stroke density (m/s)")
            ax.set_title(f"Recorded and reconstructed velocity: {subject} {task} {file.name}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            plt.show()
            print(f"Warning: Number of filtered strokes ({len(filtered_strokes)}) does not match number of waypoints ({len(waypoints) - 1}). for file {file}")
            return (file, "warning", f"Number of filtered strokes ({len(filtered_strokes)}) does not match number of waypoints ({len(waypoints) - 1}).")
            

        hl_speed = HL(recording.trajectory.speed, reconstructed_speed, len(filtered_strokes), len(waypoints) - 1)
        
        i0, i1 = minima_idx[0], minima_idx[-1]
        curve_cropped = Curve(
            x=recording.curve.x[i0:i1+1],
            y=recording.curve.y[i0:i1+1],
            z=recording.curve.z[i0:i1+1],
        )

        mse_curve, max_err = curve_cropped.mse(bezier)

        output_file = os.path.join(OUTPUT_ROOT, subject, task, "recording_" + str(recording_id) + ".csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        keys_ordered = list(curve_parameters['by_stroke'][0].keys())
        keys_ordered.remove('stroke')
        keys_ordered.remove('duration')
        with open(output_file, "w") as f:
            f.write(f"stroke_id,t0,mu,sigma,D,{','.join(keys_ordered)}\n")

            # check if a previous stroke does still have an effect on the first filtered stroke
            #if strokes.index(filtered_strokes[0]) > 0:
            #    prev_stroke = strokes[filtered_strokes.index(filtered_strokes[0]) - 1]
            #    first_filtered_stroke = filtered_strokes[0]
            #    previous_stroke_95_time = lognormal_time_at_fraction(prev_stroke, 0.95)
            #    current_stroke_start_time = lognormal_time_at_fraction(first_filtered_stroke, 0.05)
            #    print(f"Previous stroke 95% time: {previous_stroke_95_time}, Current stroke 5% time: {current_stroke_start_time} in file {file}")
            for stroke in filtered_strokes:
                f.write(f"{filtered_strokes.index(stroke)},{stroke.to},{stroke.mu},{stroke.sigma},{stroke.D}")
                for key in keys_ordered:
                    value = curve_parameters['by_stroke'][strokes.index(stroke)][key]
                    if isinstance(value, np.ndarray):
                        value_str = ';'.join(map(str, value))
                    elif isinstance(value, (list, tuple)):
                        value_str = ';'.join(map(str, value))
                    else:
                        value_str = str(value)
                    f.write(f",{value_str}")
                f.write("\n")
        return (file, "ok", {"hl_speed": hl_speed, "mse": mse_curve, "max_err": max_err})
    except Exception as e:
        return (file, "error", str(e))


with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as pool:
    futures = {pool.submit(process_file, f): f for f in CSV_FILES}
    for fut in as_completed(futures):
        file, status, msg = fut.result()
        print(f"{file.name}: {status} {msg}")