"""
Interactive Plotly viewer for the reconstructed CSVs.

One HTML file per (subject, task). Each figure has two panels:
  - left : speed as the sum of the lognormal strokes (+ the single stroke component when one is selected)
  - right: the C1 cubic Bezier spline (+ the selected segment and its control polygon)

Two dropdowns drive the view:
  - Recording : which recording of that subject/task is shown
  - View      : "All strokes" or a single stroke k (left: component k, right: segment k)

Expected CSV layout (written by the reconstruction script):
    stroke_id,t0,mu,sigma,D,lambda_s,u_s,lambda_e,u_e,delta_p
where u_s, u_e and delta_p are stored as "x;y;z".

NOTE: the CSVs do not store the first point p0 of the spline, only the per-segment
delta_p, so the spline is drawn starting at the origin (shape is exact up to a translation).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots
from scipy.special import erfinv

# ------------------------------------------------------------------ Paths
ROOT = os.path.join(os.getcwd()).split("src")[0]
CSV_ROOT = Path(ROOT) / "reconstructed_csv"
PLOT_ROOT = Path(ROOT) / "reconstructed_plots"

# ------------------------------------------------------------------ Options
SAMPLING_HZ = 500             # time resolution of the speed curves
END_FRACTION = 0.995          # a stroke is considered finished when it covers this fraction of D
BEZIER_SAMPLES = 100          # samples per Bezier segment
COMPONENTS_OPACITY_ALL = 0.0  # opacity of the stroke components in "All strokes" (0 = hidden, e.g. 0.4 = faint)
DIM_SEGMENT_OPACITY = 0.12    # opacity of the non-selected Bezier segments in single-stroke view
DIM_TOTAL_OPACITY = 0.35      # opacity of the summed speed in single-stroke view
PALETTE = qualitative.Plotly


# ------------------------------------------------------------------ Loading
def parse_vec(cell) -> np.ndarray:
    return np.array([float(v) for v in str(cell).split(";")])


def load_recording(file: Path) -> dict:
    df = pd.read_csv(file)
    return dict(
        t0=df["t0"].to_numpy(float),
        mu=df["mu"].to_numpy(float),
        sigma=df["sigma"].to_numpy(float),
        D=df["D"].to_numpy(float),
        lambda_s=df["lambda_s"].to_numpy(float),
        lambda_e=df["lambda_e"].to_numpy(float),
        u_s=np.stack(df["u_s"].map(parse_vec).to_list()),
        u_e=np.stack(df["u_e"].map(parse_vec).to_list()),
        delta_p=np.stack(df["delta_p"].map(parse_vec).to_list()),
    )


# ------------------------------------------------------------------ Speed model
def lognormal_density(t, t0, mu, sigma, D):
    tau = t - t0
    out = np.zeros_like(t)
    m = tau > 0
    out[m] = (
        D / (sigma * np.sqrt(2.0 * np.pi) * tau[m])
        * np.exp(-((np.log(tau[m]) - mu) ** 2) / (2.0 * sigma ** 2))
    )
    return out


def speed_components(rec: dict):
    """Return the time axis, the per-stroke speeds (K, N) and their sum (N,)."""
    t_start = rec["t0"].min()
    t_end = (
        rec["t0"] + np.exp(rec["mu"] + np.sqrt(2.0) * rec["sigma"] * erfinv(2.0 * END_FRACTION - 1.0))
    ).max()
    t = np.arange(t_start, t_end, 1.0 / SAMPLING_HZ)
    comps = np.stack([
        lognormal_density(t, rec["t0"][k], rec["mu"][k], rec["sigma"][k], rec["D"][k])
        for k in range(len(rec["t0"]))
    ])
    return t, comps, comps.sum(axis=0)


# ------------------------------------------------------------------ Bezier spline
def bezier_chain(rec: dict):
    """Rebuild the chain of cubic Bezier segments: p1 = p0 + lambda_s*u_s, p2 = p3 - lambda_e*u_e."""
    s = np.linspace(0.0, 1.0, BEZIER_SAMPLES)[:, None]
    p0 = np.zeros(3)
    segments = []
    for k in range(len(rec["t0"])):
        p3 = p0 + rec["delta_p"][k]
        p1 = p0 + rec["lambda_s"][k] * rec["u_s"][k]
        p2 = p3 - rec["lambda_e"][k] * rec["u_e"][k]
        points = (
            (1 - s) ** 3 * p0
            + 3 * (1 - s) ** 2 * s * p1
            + 3 * (1 - s) * s ** 2 * p2
            + s ** 3 * p3
        )
        segments.append(dict(points=points, control=np.vstack([p0, p1, p2, p3])))
        p0 = p3
    nodes = np.vstack([segments[0]["control"][0]] + [seg["control"][3] for seg in segments])
    return segments, nodes


# ------------------------------------------------------------------ Figure
def opacity_of(role: str, k, selected) -> float:
    """Opacity of a trace given its role/stroke and the selected stroke (None = all)."""
    if selected is None:
        return dict(total=1.0, comp=COMPONENTS_OPACITY_ALL, seg=1.0, ctrl=0.0, nodes=1.0)[role]
    on = k == selected
    return dict(
        total=DIM_TOTAL_OPACITY,
        comp=1.0 if on else 0.0,
        seg=1.0 if on else DIM_SEGMENT_OPACITY,
        ctrl=1.0 if on else 0.0,
        nodes=0.6,
    )[role]


def build_figure(title: str, recordings: list[tuple[str, dict]]) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        column_widths=[0.5, 0.5],
        subplot_titles=("Speed = sum of lognormal strokes", "Bézier spline (starts at origin)"),
        horizontal_spacing=0.04,
    )
    meta = []  # (recording index, role, stroke index) for every trace, in insertion order

    def add(trace, row, col, r, role, k=None):
        trace.update(opacity=opacity_of(role, k, None), visible=(r == 0))
        fig.add_trace(trace, row=row, col=col)
        meta.append((r, role, k))

    n_max = 0
    for r, (_, rec) in enumerate(recordings):
        t, comps, total = speed_components(rec)
        segments, nodes = bezier_chain(rec)
        n_max = max(n_max, len(segments))

        add(go.Scatter(x=t, y=total, mode="lines", name="Sum of strokes",
                       line=dict(color="black", width=2.5), legendgroup=f"r{r}"),
            1, 1, r, "total")

        for k in range(len(segments)):
            color = PALETTE[k % len(PALETTE)]
            group = f"r{r}_s{k}"
            add(go.Scatter(x=t, y=comps[k], mode="lines", name=f"Stroke {k + 1}",
                           line=dict(color=color, width=2), fill="tozeroy",
                           legendgroup=group, hoverinfo="skip"),
                1, 1, r, "comp", k)

        for k, seg in enumerate(segments):
            color = PALETTE[k % len(PALETTE)]
            pts = seg["points"]
            add(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="lines",
                             line=dict(color=color, width=6), name=f"Segment {k + 1}",
                             legendgroup=f"r{r}_s{k}", showlegend=False),
                1, 2, r, "seg", k)

        for k, seg in enumerate(segments):
            c = seg["control"]
            add(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="lines+markers+text",
                             text=["P0", "P1", "P2", "P3"], textposition="top center",
                             line=dict(color="gray", width=2, dash="dash"),
                             marker=dict(size=4, color="gray"),
                             name=f"Control polygon {k + 1}", showlegend=False,
                             hoverinfo="skip"),
                1, 2, r, "ctrl", k)

        add(go.Scatter3d(x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2], mode="markers",
                         marker=dict(size=4, color="black"), name="Stroke boundaries",
                         showlegend=False),
            1, 2, r, "nodes")

    # ---- Dropdown 1: recording (drives `visible`)
    recording_buttons = [
        dict(label=name, method="restyle",
             args=[{"visible": [m[0] == r for m in meta]}])
        for r, (name, _) in enumerate(recordings)
    ]
    # ---- Dropdown 2: stroke view (drives `opacity`, so it composes with the recording choice)
    view_buttons = [
        dict(label="All strokes", method="restyle",
             args=[{"opacity": [opacity_of(role, k, None) for _, role, k in meta]}])
    ] + [
        dict(label=f"Stroke {k + 1}", method="restyle",
             args=[{"opacity": [opacity_of(role, kk, k) for _, role, kk in meta]}])
        for k in range(n_max)
    ]

    fig.update_layout(
        title=dict(text=title, x=0.01),
        template="plotly_white",
        height=700,
        margin=dict(t=140, b=40, l=60, r=20),
        legend=dict(orientation="h", y=-0.05),
        updatemenus=[
            dict(buttons=recording_buttons, direction="down", x=0.08, y=1.17,
                 xanchor="left", yanchor="top", active=0),
            dict(buttons=view_buttons, direction="down", x=0.40, y=1.17,
                 xanchor="left", yanchor="top", active=0),
        ],
        annotations=list(fig.layout.annotations) + [
            dict(text="Recording", x=0.0, y=1.145, xref="paper", yref="paper",
                 showarrow=False, xanchor="left"),
            dict(text="View", x=0.35, y=1.145, xref="paper", yref="paper",
                 showarrow=False, xanchor="left"),
        ],
    )
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Speed (m/s)", row=1, col=1)
    fig.update_scenes(aspectmode="data",
                      xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)")
    return fig


# ------------------------------------------------------------------ Main
def recording_id(file: Path) -> int:
    return int(file.stem.split("_")[-1])


if __name__ == "__main__":
    groups: dict[tuple[str, str], list[Path]] = {}
    for f in sorted(CSV_ROOT.glob("*/*/recording_*.csv")):
        groups.setdefault((f.parts[-3], f.parts[-2]), []).append(f)

    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    written = []

    for (subject, task), files in sorted(groups.items()):
        files = sorted(files, key=recording_id)
        try:
            recordings = [(f.stem, load_recording(f)) for f in files]
            fig = build_figure(f"{subject} — {task}", recordings)
            out = PLOT_ROOT / f"{subject}__{task}.html"
            # "directory": plotly.min.js is saved once next to the HTML files (works offline)
            fig.write_html(out, include_plotlyjs="directory")
            written.append(out)
            print(f"{subject}/{task}: {len(files)} recordings -> {out}")
        except Exception as e:
            print(f"{subject}/{task}: error -> {e}")

    # Small index page linking all the figures
    links = "\n".join(f'<li><a href="{p.name}">{p.stem.replace("__", " / ")}</a></li>' for p in written)
    (PLOT_ROOT / "index.html").write_text(f"<html><body><h2>Reconstructed plots</h2><ul>\n{links}\n</ul></body></html>")