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
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------ Input files
CSV_ROOT = os.path.join(os.getcwd()).split('src')[0] + "/recording_csv"
CSV_FILES = sorted(Path(CSV_ROOT).glob("**/*.csv"))
YAML_PATHS_FILE = os.path.join(os.getcwd()).split('src')[0] + "/src/unisa_acg_ros2/haptics/haptic_interaction_teach_and_record_demo/config/paths.yaml"
OUTPUT_ROOT = os.path.join(os.getcwd()).split('src')[0] + "/reconstructed_csv"
DEBUG = False
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
LOCK_D_TO_GEOMETRY = True  # D of each stroke = measured path length between its two speed minima

# ---------------------------------------------------------------- Stroke detection
TIME_DISTANCE = .1 #s
PROMINENCE = .005 #m/s
VEL_THRESHOLD = 0.2 #m/s

# ---------------------------------------------------------------- Bezier geometry
LAMBDA_MIN = 1e-3                                          # m: lower bound of the tangent lengths
SWEEP = True                                               # True: compare DIRECTION_DISTANCES without writing the dataset
DIRECTION_DISTANCES = [k * 1e-3 for k in range(1, 11)]     # m: radii tried for the tangent neighbourhood (SWEEP = True)
DIRECTION_DISTANCE = 2.5e-3                                # m: radius used to write the dataset (SWEEP = False)
MAX_ENDPOINT_REL_ERROR = None  # e.g. 0.5: maximum ||delta_p - nominal step|| / ||nominal step||; None disables the check


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


def plot_filtered_strokes(recording, strokes, filtered_strokes, title):
    """Recorded speed, model speed and every stroke, with the strokes left out by the peak filter in red."""
    t = recording.t
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, recording.trajectory.speed, color="black", linewidth=1.5, label="Recorded speed")
    ax.plot(t, strokes.v_array(t), color="tab:blue", linewidth=1.5, label="Reconstructed speed")
    for index, stroke in enumerate(strokes):
        kept = stroke in filtered_strokes
        color = "tab:green" if kept else "tab:red"
        density = StrokeSequence([stroke]).v_array(t)
        ax.plot(t, density, color=color, linewidth=1.0, label=f"Stroke {index + 1}" + ("" if kept else " (filtered out)"))
        ax.fill_between(t, 0, density, color=color, alpha=0.18)
    ax.axhline(VEL_THRESHOLD, color="tab:orange", linestyle="--", linewidth=1.5, label="Velocity filter")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title(f"Recorded and reconstructed speed: {title}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def write_strokes(output_file, filtered_strokes, curve_parameters):
    """One row per kept stroke: lognormal parameters and Bezier geometry."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(f"stroke_id,t0,mu,sigma,D,{','.join(GEOMETRY_OUTPUT_COLUMNS)}\n")
        for k, (stroke, params) in enumerate(zip(filtered_strokes, curve_parameters)):
            values = [str(k), str(stroke.to), str(stroke.mu), str(stroke.sigma), str(stroke.D)]
            for key in GEOMETRY_OUTPUT_COLUMNS:
                value = params[key]
                values.append(";".join(map(str, value)) if isinstance(value, (np.ndarray, list, tuple)) else str(value))
            f.write(",".join(values) + "\n")


def process_file(file, direction_distances, write_output):
    """
    Lognormal and Bezier extraction of one recording. The lognormal fit is done once; the Bezier fit is
    repeated for every value in `direction_distances`. Returns (file, status, message), where on success
    the message is a list with one dict of metrics per direction distance.
    """
    ls = LeastSquares(lock_D=LOCK_D_TO_GEOMETRY, refine=False)
    filter = Chebyshev(order=FILTER_ORDER, cutoff=FILTER_CUTOFF_HZ, stopband_attenuation=FILTER_STOPBAND_DB)
    resampler = Resample(target_fs=TARGET_FREQUENCY)

    try:
        subject, task, recording_id = file.parts[-3], file.parts[-2], int(file.stem.split("_")[-1])
        waypoints = load_waypoints(YAML_PATHS_FILE, task)

        csv_data = pd.read_csv(file)
        csv_data['timestamp'] = (csv_data['timestamp'] - csv_data['timestamp'].iloc[0]) / 1e9  # ns from the first sample -> s

        recording = Recording(
            trajectory=Trajectory(
                vx=csv_data['twist_linear_x'].to_numpy(),
                vy=csv_data['twist_linear_y'].to_numpy(),
                vz=csv_data['twist_linear_z'].to_numpy(),
            ),
            curve=Curve(
                x=csv_data['position_x'].to_numpy(),
                y=csv_data['position_y'].to_numpy(),
                z=csv_data['position_z'].to_numpy(),
            ),
            t=csv_data['timestamp'].to_numpy()
        )
        recording.trajectory = resampler.apply(filter.apply(recording.trajectory)) # type: ignore
        recording.curve = resampler.apply(filter.apply(recording.curve)) # type: ignore
        t = recording.t

        minima_idx, _ = find_minima_maxima(recording.trajectory.speed, samples_distance=int(TIME_DISTANCE * TARGET_FREQUENCY), prominence=PROMINENCE)
        ls.set_boundaries(minima_idx)
        strokes = ls.approximate(recording)

        # Contiguous strokes whose peak speed reaches the threshold: one per segment of the task
        filtered_strokes, _ = strokes.filter_strokes_by_peak(VEL_THRESHOLD)
        n_segments = len(waypoints) - 1
        if len(filtered_strokes) != n_segments:
            if DEBUG:
                plot_filtered_strokes(recording, strokes, filtered_strokes, f"{subject} {task} {file.name}")
            return (file, "warning", f"Number of filtered strokes ({len(filtered_strokes)}) does not match number of segments ({n_segments}).")

        # Stroke k spans the window [minima_idx[k], minima_idx[k + 1]]
        first = strokes.index(filtered_strokes[0])
        minima_idx_strokes = [int(i) for i in minima_idx[first:first + len(filtered_strokes) + 1]]
        points = recording.curve.points

        # Each stroke must reach (approximately) the waypoint of its segment, in the device frame
        if MAX_ENDPOINT_REL_ERROR is not None:
            nominal_steps = np.diff(waypoints_to_array(waypoints, SWAP_YZ_WAYPOINTS, WS_MIN, WS_MAX), axis=0)
            delta_ps = np.diff(points[minima_idx_strokes], axis=0)
            rel_errors = np.linalg.norm(delta_ps - nominal_steps, axis=1) / np.linalg.norm(nominal_steps, axis=1)
            if rel_errors.max() > MAX_ENDPOINT_REL_ERROR:
                return (file, "warning", f"Stroke end points far from their segments (relative errors {np.round(rel_errors, 2).tolist()}).")

        # Speed quality of the kept strokes (independent of the Bezier geometry)
        kept_speed = StrokeSequence(filtered_strokes).v_array(t)
        hl_speed = HL(recording.trajectory.speed, kept_speed, len(filtered_strokes), n_segments)

        # Recorded samples covered by the Bezier chain
        i0, i1 = minima_idx_strokes[0], minima_idx_strokes[-1]
        curve_cropped = Curve(x=recording.curve.x[i0:i1 + 1], y=recording.curve.y[i0:i1 + 1], z=recording.curve.z[i0:i1 + 1])

        # D of each stroke, shared by its lognormal amplitude and by the length of its Bezier segment
        Ds = [float(stroke.D) for stroke in filtered_strokes]

        metrics = []
        for direction_distance in direction_distances:
            approx = BezierOptimization()
            approx.set_problem(minima_idx=minima_idx_strokes, Ds=Ds, lambda_min=LAMBDA_MIN, lambda_max=None,
                               direction_distance=direction_distance)
            try:
                curve_parameters, bezier = approx.approximate(recording.curve)
            except Exception as e:
                if DEBUG:
                    raise e
                metrics.append({"direction_distance": direction_distance, "status": "error", "message": str(e)})
                continue

            # Bezier samples at the same curvilinear abscissa as the recorded ones: shape error only
            mse_curve, max_err = curve_cropped.mse(bezier)
            metrics.append({"direction_distance": direction_distance, "status": "ok",
                            "mse": mse_curve, "max_err": max_err, "hl_speed": hl_speed})

            if write_output:
                write_strokes(os.path.join(OUTPUT_ROOT, subject, task, f"recording_{recording_id}.csv"),
                              filtered_strokes, curve_parameters)
        return (file, "ok", metrics)
    except Exception as e:
        if DEBUG:
            raise e
        return (file, "error", str(e))


def summarize(rows, n_distances):
    """Mean and median metrics per direction distance, on the recordings fitted with every distance
    (or, if there is none, on the successful fits of each distance)."""
    table = pd.DataFrame(rows)
    if table.empty:
        print("No recording processed.")
        return table

    table["direction_distance [mm]"] = table["direction_distance"] * 1e3
    failures = table.groupby("direction_distance [mm]")["status"].apply(lambda s: int((s != "ok").sum())).rename("failed fits")
    ok = table[table["status"] == "ok"].copy()
    complete = ok.groupby("file")["direction_distance"].nunique()
    common = ok[ok["file"].isin(complete[complete == n_distances].index)].copy()

    if common.empty:
        print("No recording was fitted with every direction distance: each row uses its own successful fits.\n")
        common = ok
    else:
        print(f"Comparison on the {common['file'].nunique()} recordings fitted with every direction distance.\n")

    common["mse [mm^2]"] = common["mse"] * 1e6
    common["max error [mm]"] = np.sqrt(common["max_err"]) * 1e3   # largest absolute error of a single coordinate
    summary = common.groupby("direction_distance [mm]").agg(
        recordings=("file", "nunique"),
        mean_mse_mm2=("mse [mm^2]", "mean"),
        median_mse_mm2=("mse [mm^2]", "median"),
        mean_max_error_mm=("max error [mm]", "mean"),
        median_max_error_mm=("max error [mm]", "median"),
        mean_hl_speed=("hl_speed", "mean"),
    )
    summary = summary.join(failures, how="outer")
    print(summary.round(4).to_string())

    if len(summary) > 1:
        best = summary["mean_mse_mm2"].idxmin()
        print(f"\nLowest mean curve MSE with direction distance = {best:g} mm "
              f"(lowest median: {summary['median_mse_mm2'].idxmin():g} mm)")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].plot(summary.index, summary["mean_mse_mm2"], "o-", label="mean")
        axes[0].plot(summary.index, summary["median_mse_mm2"], "s--", label="median")
        axes[0].set_xlabel("direction distance [mm]")
        axes[0].set_ylabel("curve MSE [mm²]")
        axes[0].set_title("Curve MSE")
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        axes[1].plot(summary.index, summary["mean_max_error_mm"], "o-", label="mean")
        axes[1].plot(summary.index, summary["median_max_error_mm"], "s--", label="median")
        axes[1].set_xlabel("direction distance [mm]")
        axes[1].set_ylabel("max coordinate error [mm]")
        axes[1].set_title("Maximum error")
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        fig.tight_layout()
        plt.show()
    return summary


if __name__ == '__main__':
    direction_distances = DIRECTION_DISTANCES if SWEEP else [DIRECTION_DISTANCE]
    write_output = not SWEEP
    rows, outcomes = [], []

    def collect(file, status, msg):
        name = str(file).split('recording_csv')[-1]
        outcomes.append(status)
        if status != "ok":
            print(f"{name}: {status} {msg}")
            return
        for m in msg:
            rows.append({"file": name, **m})
            if m["status"] != "ok":
                print(f"{name}: Bezier fit failed with direction distance {m['direction_distance'] * 1e3:g} mm: {m['message']}")

    if not DEBUG:
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
            futures = {pool.submit(process_file, f, direction_distances, write_output): f for f in CSV_FILES}
            for fut in as_completed(futures):
                collect(*fut.result())
    else:
        for f in CSV_FILES:
            collect(*process_file(f, direction_distances, write_output))

    print(f"\nRecordings: {outcomes.count('ok')} processed, {outcomes.count('warning')} skipped, {outcomes.count('error')} errors\n")
    summarize(rows, len(direction_distances))