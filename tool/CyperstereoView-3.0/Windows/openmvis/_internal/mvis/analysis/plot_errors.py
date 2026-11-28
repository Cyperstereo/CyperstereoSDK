# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-FileCopyrightText: 2026 Yulin Yang <contact@openmvis.com>
# SPDX-License-Identifier: GPL-3.0-only
"""Matplotlib plots for reprojection error analysis.

All functions return matplotlib Figure objects (not saved to disk).
The HTML report generator embeds them as base64 PNG.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from mvis.detection.board import Board
from mvis.core.types import FrameObservations


def plot_frame_tag_heatmap(
    frame_tag_errors: dict[int, dict[int, float]],
    num_frames: int,
    num_tags: int,
    camera_id: int = 0,
) -> Figure:
    """Heatmap of reprojection error: frame index (rows) vs tag ID (columns).

    Args:
        frame_tag_errors: frame_idx → {tag_id → rms_error}.
        num_frames: Total number of frames.
        num_tags: Total number of tags.
        camera_id: Camera ID for title.

    Returns:
        Matplotlib Figure.
    """
    grid = np.full((num_frames, num_tags), np.nan)
    for fi, tag_errs in frame_tag_errors.items():
        for tid, err in tag_errs.items():
            if tid < num_tags and fi < num_frames:
                grid[fi, tid] = err

    fig, ax = plt.subplots(figsize=(max(8, num_tags * 0.25), max(4, num_frames * 0.08)))
    im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest", aspect="auto",
                   vmin=0, vmax=np.nanpercentile(grid, 95) if np.any(~np.isnan(grid)) else 1)
    fig.colorbar(im, ax=ax, label="RMS Error (px)")

    ax.set_xlabel("Tag ID")
    ax.set_ylabel("Frame Index")
    ax.set_title(f"cam{camera_id} — Per-Frame Per-Tag Error")
    fig.tight_layout()
    return fig


def plot_detection_rate_per_tag(
    tag_detection_counts: dict[int, int],
    num_tags: int,
    total_frames: int,
    camera_id: int = 0,
    is_chessboard: bool = False,
) -> Figure:
    """Bar chart of detection count per tag/corner ID.

    Args:
        tag_detection_counts: tag_id → number of frames this tag was detected.
        num_tags: Total number of tags (or corners for chessboard).
        total_frames: Total number of frames.
        camera_id: Camera ID for title.
        is_chessboard: If True, label as "Corner" instead of "Tag".

    Returns:
        Matplotlib Figure.
    """
    unit = "Corner" if is_chessboard else "Tag"
    tag_ids = list(range(num_tags))
    counts = [tag_detection_counts.get(tid, 0) for tid in tag_ids]

    fig, ax = plt.subplots(figsize=(max(8, num_tags * 0.3), 3.5))
    colors = ["#F44336" if c < total_frames * 0.5 else "#4CAF50" for c in counts]
    ax.bar(tag_ids, counts, color=colors, edgecolor="white", width=0.8)
    ax.axhline(total_frames, color="#333", linestyle="--", linewidth=1,
               label=f"Total frames = {total_frames}")
    ax.legend(fontsize=9)

    ax.set_xlabel(f"{unit} ID")
    ax.set_ylabel("Detection Count")
    ax.set_title(f"cam{camera_id} — Detections Per {unit}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_per_frame_rms(
    frame_rms: list[float],
    camera_id: int = 0,
) -> Figure:
    """Line chart of per-frame RMS reprojection error.

    Args:
        frame_rms: RMS error per frame (NaN for skipped frames).
        camera_id: Camera ID for title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    frames = np.arange(len(frame_rms))
    rms = np.array(frame_rms)
    valid = ~np.isnan(rms)

    ax.plot(frames[valid], rms[valid], ".-", color="#2196F3", markersize=4, linewidth=1)
    if np.any(valid):
        mean_rms = float(np.nanmean(rms))
        ax.axhline(mean_rms, color="#F44336", linestyle="--", linewidth=1,
                    label=f"Mean RMS = {mean_rms:.3f} px")
        ax.legend(fontsize=9)

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("RMS Error (px)")
    ax.set_title(f"cam{camera_id} — Per-Frame Reprojection RMS")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_frame_robust(
    frame_median: list[float],
    frame_p95: list[float],
    camera_id: int = 0,
) -> Figure:
    """Line chart of per-frame median and 95th-percentile reprojection error.

    Robust alternative to per-frame RMS — a single bad observation can't
    dominate either metric. Median shows typical error; p95 shows tail.

    Args:
        frame_median: Median error per frame (NaN for skipped frames).
        frame_p95: 95th percentile error per frame (NaN for skipped frames).
        camera_id: Camera ID for title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    frames = np.arange(max(len(frame_median), len(frame_p95)))

    med = np.array(frame_median, dtype=np.float64)
    p95 = np.array(frame_p95, dtype=np.float64)
    valid_m = ~np.isnan(med)
    valid_p = ~np.isnan(p95)

    if np.any(valid_m):
        ax.plot(frames[:len(med)][valid_m], med[valid_m], ".-",
                color="#2196F3", markersize=4, linewidth=1, label="Median")
    if np.any(valid_p):
        ax.plot(frames[:len(p95)][valid_p], p95[valid_p], ".-",
                color="#FF9800", markersize=4, linewidth=1, alpha=0.75,
                label="95th pct")

    if np.any(valid_m):
        mean_med = float(np.nanmean(med))
        ax.axhline(mean_med, color="#1976D2", linestyle="--", linewidth=1,
                    alpha=0.6,
                    label=f"Mean median = {mean_med:.3f} px")

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Error (px)")
    ax.set_title(f"cam{camera_id} — Per-Frame Error (Median & 95th Pct)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_error_histogram(
    errors: np.ndarray,
    camera_id: int = 0,
    bins: int = 50,
) -> Figure:
    """Histogram of per-observation pixel error magnitudes.

    Args:
        errors: 1D array of pixel error magnitudes.
        camera_id: Camera ID for title.
        bins: Number of histogram bins.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 3.5))
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return fig
    ax.hist(errors, bins=bins, color="#4CAF50", edgecolor="white", alpha=0.85)

    rms = float(np.sqrt(np.mean(errors ** 2)))
    median = float(np.median(errors))
    ax.axvline(rms, color="#F44336", linestyle="--", linewidth=1.5,
               label=f"RMS = {rms:.3f} px")
    ax.axvline(median, color="#FF9800", linestyle="--", linewidth=1.5,
               label=f"Median = {median:.3f} px")

    ax.set_xlabel("Reprojection Error (px)")
    ax.set_ylabel("Count")
    ax.set_title(f"cam{camera_id} — Error Distribution ({len(errors)} observations)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _robust_gaussian_params(data: np.ndarray) -> tuple[float, float]:
    """Fit a Gaussian to the bulk of the data using median + MAD.

    The classical (mean, std) estimator is pulled toward outliers; for
    fisheye residuals the std of 71K observations can be inflated 5-10×
    by a handful of spurious detections, so the resulting bell curve
    looks flat and wide compared to the histogram bars.

    The **median** is an unbiased estimator of μ for a symmetric Gaussian
    and is insensitive to outliers. The **MAD** (median absolute
    deviation from the median), scaled by the Gaussian consistency
    constant 1.4826, is the corresponding robust estimator of σ.
    For pure-Gaussian data, (median, 1.4826·MAD) ≈ (mean, std); for
    data with a Gaussian core plus outliers, the robust pair recovers
    the core parameters.

    Returns:
        (mu_robust, sigma_robust).
    """
    arr = np.asarray(data, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    sigma = 1.4826 * mad
    # If MAD is essentially zero (e.g. >50% ties at one value) fall back
    # to a trimmed std over the inner 95% so the overlay is still plotted.
    if sigma < 1e-12:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        sigma = float(np.std(trimmed)) if trimmed.size else 1e-6
    return med, max(sigma, 1e-6)


def plot_error_histogram_uv(
    error_vectors: np.ndarray,
    camera_id: int = 0,
    bins: int = 80,
    zoom_percentile: float = 99.0,
) -> Figure:
    """Two-panel histogram of signed u and v errors with a robust Gaussian overlay.

    The x-axis is zoomed into the central `zoom_percentile` of the data
    so a few huge outliers do not flatten the bell-curve core into a
    single bar. The Gaussian overlay is a **robust** fit computed from
    (median, 1.4826·MAD) so it captures the bulk of the distribution
    rather than being pulled toward the tails. We also show the raw
    (mean, std) fit as a thin dashed curve so the reader can tell at a
    glance how much the outliers inflate the non-robust estimate.

    Markers:
      - blue dashed vertical: mean
      - orange dashed vertical: 2.5% and 97.5% percentiles
      - solid blue curve: N(median, 1.4826·MAD) — robust fit
      - dashed gray curve: N(mean, std) — raw fit

    Args:
        error_vectors: Nx2 array of signed [u_error, v_error] per observation.
        camera_id: Camera ID for title.
        bins: Number of histogram bins per panel (over the zoomed range).
        zoom_percentile: Two-sided percentile used to clip the x-axis
            (e.g. 99 → clip at P0.5 / P99.5).
    """
    ev = np.asarray(error_vectors, dtype=np.float64)
    if ev.ndim != 2 or ev.shape[1] != 2 or ev.shape[0] == 0:
        fig, _ = plt.subplots(figsize=(10, 3.5))
        return fig
    # Remove NaN rows (from frames referencing diverged/identity poses)
    valid = np.isfinite(ev).all(axis=1)
    ev = ev[valid]
    if ev.shape[0] == 0:
        fig, _ = plt.subplots(figsize=(10, 3.5))
        return fig

    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)

    # Shared zoom range: symmetric around 0 using the widest P(lo..hi) across
    # u and v so both panels use the same scale. Never smaller than ±0.2 px
    # to avoid a useless micro-zoom on near-zero residuals.
    lo_pct = (100.0 - zoom_percentile) / 2.0
    hi_pct = 100.0 - lo_pct
    p_lo = float(min(np.percentile(ev[:, 0], lo_pct),
                      np.percentile(ev[:, 1], lo_pct)))
    p_hi = float(max(np.percentile(ev[:, 0], hi_pct),
                      np.percentile(ev[:, 1], hi_pct)))
    lim = max(abs(p_lo), abs(p_hi), 0.2)

    n_out = int(np.sum((np.abs(ev[:, 0]) > lim) | (np.abs(ev[:, 1]) > lim)))

    for ax, data, label in ((ax_u, ev[:, 0], "u"), (ax_v, ev[:, 1], "v")):
        # Raw (mean, std) — sensitive to outliers
        mu = float(np.mean(data))
        std = float(np.std(data))
        # Robust (median, MAD-based σ) — fits the bulk
        mu_r, std_r = _robust_gaussian_params(data)
        p025 = float(np.percentile(data, 2.5))
        p975 = float(np.percentile(data, 97.5))

        counts, edges, _ = ax.hist(
            data, bins=bins, range=(-lim, lim),
            color="#4CAF50", edgecolor="white", alpha=0.85,
        )
        bin_w = edges[1] - edges[0]

        xs = np.linspace(-lim, lim, 400)

        # Robust Gaussian (primary)
        pdf_r = (1.0 / (std_r * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((xs - mu_r) / std_r) ** 2
        )
        ax.plot(xs, pdf_r * len(data) * bin_w,
                color="#1976D2", linewidth=1.8,
                label=f"robust N(μ̃={mu_r:.3f}, σ̃={std_r:.3f})")

        # Raw Gaussian (dim, for reference) — only shown when it meaningfully
        # differs from the robust one, otherwise it just clutters the panel.
        if std > 0 and abs(std - std_r) > 0.05 * std_r:
            pdf = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(
                -0.5 * ((xs - mu) / std) ** 2
            )
            ax.plot(xs, pdf * len(data) * bin_w,
                    color="#888888", linewidth=1.0, linestyle="--", alpha=0.8,
                    label=f"raw N(μ={mu:.3f}, σ={std:.3f})")

        ax.axvline(mu_r, color="#1976D2", linestyle=":", linewidth=1.0,
                   label=f"median = {mu_r:.3f} px")
        ax.axvline(p025, color="#FF9800", linestyle="--", linewidth=1.0,
                   label=f"2.5% = {p025:.3f}")
        ax.axvline(p975, color="#FF9800", linestyle="--", linewidth=1.0,
                   label=f"97.5% = {p975:.3f}")

        ax.set_xlim(-lim, lim)
        ax.set_xlabel(f"{label} error (px)")
        ax.set_title(
            f"{label} residuals — robust μ̃={mu_r:.3f}, σ̃={std_r:.3f}, "
            f"95% width={p975 - p025:.3f} px"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    ax_u.set_ylabel("Count")
    fig.suptitle(
        f"cam{camera_id} — u/v Residual Distribution ({len(ev)} obs) "
        f"— zoomed to ±{lim:.2f} px "
        f"({n_out} obs outside range)"
    )
    fig.tight_layout()
    return fig


def plot_corner_heatmap(
    corner_errors: dict[int, list[float]],
    board: Board,
    camera_id: int = 0,
) -> Figure:
    """Heatmap of mean error at each board corner position.

    Reveals spatial error patterns (distortion model mismatch, lens issues).

    Args:
        corner_errors: point_id → list of error magnitudes at that corner.
        board: Board object (for 3D point positions to build grid layout).
        camera_id: Camera ID for title.

    Returns:
        Matplotlib Figure.
    """
    pts = board.all_points_3d()
    n_pts = board.num_points()

    # Determine grid dimensions from point positions
    xs = np.unique(np.round(pts[:, 0], 6))
    ys = np.unique(np.round(pts[:, 1], 6))
    nx, ny = len(xs), len(ys)

    grid = np.full((ny, nx), np.nan)
    for pid in range(n_pts):
        if pid not in corner_errors or not corner_errors[pid]:
            continue
        pt = pts[pid]
        ix = np.argmin(np.abs(xs - pt[0]))
        iy = np.argmin(np.abs(ys - pt[1]))
        grid[iy, ix] = float(np.mean(corner_errors[pid]))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, cmap="YlOrRd", interpolation="nearest", aspect="auto")
    fig.colorbar(im, ax=ax, label="Mean Error (px)")

    ax.set_xlabel("Board Column")
    ax.set_ylabel("Board Row")
    ax.set_title(f"cam{camera_id} — Corner Error Heatmap")
    fig.tight_layout()
    return fig


def _draw_vector_panel(ax, ev_all, ev_by_frame, lim, show_legend=True):
    """Shared drawing code for one error-vector panel.

    Draws: scatter (colored by frame), per-frame mean arrows, global
    mean/median markers, 95% confidence ellipse.
    """
    from matplotlib.patches import Ellipse

    # --- Scatter ---
    if ev_by_frame:
        pts = []
        frame_ids = []
        for fid, vecs in ev_by_frame.items():
            pts.append(vecs)
            frame_ids.extend([fid] * len(vecs))
        if pts:
            pts = np.vstack(pts)
            frame_ids = np.array(frame_ids)
            if len(pts) > 5000:
                sub = np.random.default_rng(0).choice(len(pts), 5000, replace=False)
                pts = pts[sub]
                frame_ids = frame_ids[sub]
            ax.scatter(pts[:, 0], pts[:, 1], s=3, c=frame_ids,
                       cmap="viridis", alpha=0.35)
    else:
        n = len(ev_all)
        ev_plot = ev_all if n <= 5000 else ev_all[
            np.random.default_rng(0).choice(n, 5000, replace=False)]
        ax.scatter(ev_plot[:, 0], ev_plot[:, 1], s=2,
                   alpha=0.3, color="#2196F3")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    # --- Per-frame mean arrows ---
    if ev_by_frame:
        frames_sorted = sorted(ev_by_frame.items(),
                               key=lambda kv: -len(kv[1]))[:200]
        for _, vecs in frames_sorted:
            if len(vecs) == 0:
                continue
            m = np.mean(vecs, axis=0)
            ax.arrow(0.0, 0.0, m[0], m[1], head_width=0.0, linewidth=0.6,
                     color="#9E9E9E", alpha=0.6, length_includes_head=True)

    # --- Global median + mean ---
    med = np.median(ev_all, axis=0)
    mean_vec = np.mean(ev_all, axis=0)
    ax.plot([0, med[0]], [0, med[1]], color="#F44336", linewidth=2.0, alpha=0.9,
            label=f"Median=({med[0]:.3f},{med[1]:.3f})")
    ax.plot(med[0], med[1], "o", color="#F44336", markersize=9,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.plot(mean_vec[0], mean_vec[1], "D", color="#1976D2", markersize=9,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5,
            label=f"Mean=({mean_vec[0]:.3f},{mean_vec[1]:.3f})")

    # --- 95% ellipse via Mahalanobis trim ---
    mu = np.mean(ev_all, axis=0)
    Sigma = np.cov(ev_all.T) if len(ev_all) >= 2 else np.eye(2)
    semi_major_95 = None
    try:
        inv = np.linalg.inv(Sigma + 1e-12 * np.eye(2))
        d2 = np.einsum("ni,ij,nj->n", ev_all - mu, inv, ev_all - mu)
        keep = d2 <= np.percentile(d2, 95)
        if np.sum(keep) >= 10:
            ev_clean = ev_all[keep]
            mu_c = np.mean(ev_clean, axis=0)
            S_c = np.cov(ev_clean.T)
            evals, evecs = np.linalg.eigh(S_c)
            scale = np.sqrt(5.991)  # chi-square 95% 2-DoF
            width = 2.0 * scale * np.sqrt(max(evals[1], 0.0))
            height = 2.0 * scale * np.sqrt(max(evals[0], 0.0))
            semi_major_95 = max(width, height) / 2.0
            angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
            ell = Ellipse(xy=(mu_c[0], mu_c[1]), width=width, height=height,
                          angle=angle, facecolor="none", edgecolor="black",
                          linewidth=1.5, linestyle="--",
                          label=f"95% (a={width/2:.2f},b={height/2:.2f})")
            ax.add_patch(ell)
    except np.linalg.LinAlgError:
        pass

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("u error (px)")
    ax.set_ylabel("v error (px)")
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=8, loc="upper right")
    return semi_major_95


def plot_error_vectors_dual(
    error_vectors: np.ndarray,
    error_vectors_by_frame: dict | None = None,
    camera_id: int = 0,
) -> Figure:
    """Two-panel error-vector plot.

    Left panel ("Full range"): axis = ±max(|u|,|v|)*1.05, shows tail
    shape. Right panel ("Zoomed"): axis = 2 × 95%-ellipse semi-major
    axis, guarantees the ellipse covers ≥ half the panel. Both panels
    carry mean, median, per-frame arrows, and the 95% ellipse so each
    tells a complete story.
    """
    ev_all = np.asarray(error_vectors, dtype=np.float64)
    if ev_all.ndim != 2 or ev_all.shape[1] != 2 or ev_all.shape[0] == 0:
        fig, _ = plt.subplots(figsize=(6, 6))
        return fig

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 7))

    # Full-range axis
    full_lim = max(float(np.max(np.abs(ev_all))) * 1.05, 1.0)
    semi_major = _draw_vector_panel(ax_full, ev_all, error_vectors_by_frame,
                                     full_lim, show_legend=False)
    ax_full.set_title(f"Full range (±{full_lim:.1f} px)", fontsize=10)

    # Zoomed: axis_lim = 2 × semi_major so ellipse fills ≥ half the panel.
    # Fall back to P99 if ellipse fit failed.
    if semi_major is not None and semi_major > 0:
        zoom_lim = max(2.0 * semi_major, 0.5)
    else:
        zoom_lim = max(float(np.percentile(np.max(np.abs(ev_all), axis=1), 99.0))
                       * 1.1, 1.0)
    _draw_vector_panel(ax_zoom, ev_all, error_vectors_by_frame,
                       zoom_lim, show_legend=True)
    n_out = int(np.sum(np.max(np.abs(ev_all), axis=1) > zoom_lim))
    subtitle = (f"Zoomed to ±{zoom_lim:.2f} px"
                + (f" ({n_out} outside)" if n_out > 0 else ""))
    ax_zoom.set_title(subtitle, fontsize=10)

    fig.suptitle(f"cam{camera_id} — Error Vectors",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    return fig


def plot_reprojection_scatter(
    error_vectors_by_frame: dict,
    camera_id: int = 0,
    max_points: int = 60000,
) -> Figure:
    """Kalibr-style reprojection scatter: every corner's (u, v) error as an ×, coloured by image index.

    Points are drawn in frame order, so later frames sit on top (as in Kalibr's imucam report). The
    dashed circle is 3σ of the residuals, σ from the median absolute deviation of u and v pooled, so
    a few bad corners don't inflate it. The axes span the circle and 99.9 % of the points; the count
    left outside goes in the title. Above `max_points` an evenly spread subset is drawn.
    """
    frames = sorted((int(f), np.asarray(v, dtype=np.float64)) for f, v in (error_vectors_by_frame or {}).items()
                    if len(v))
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    if not frames:
        ax.set_title(f"cam{camera_id} — reprojection errors (no data)")
        return fig
    pts = np.vstack([v for _, v in frames])
    idx = np.concatenate([np.full(len(v), f) for f, v in frames])
    sigma = 1.4826 * float(np.median(np.abs(pts - np.median(pts, axis=0))))
    r3 = 3.0 * sigma
    lim = max(float(np.percentile(np.abs(pts), 99.9)), r3) * 1.15
    n_out = int(np.sum(np.max(np.abs(pts), axis=1) > lim))
    if len(pts) > max_points:
        keep = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts, idx = pts[keep], idx[keep]
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=idx, cmap="jet", marker="x", s=14, linewidths=1.0,
                    alpha=0.6, vmin=0, vmax=max(int(idx.max()), 1), rasterized=True)
    ax.add_patch(plt.Circle((0, 0), r3, fill=False, color="black", linestyle="--", linewidth=1.6,
                            label=f"3σ = {r3:.2f} px (σ = {sigma:.2f})"))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.4)
    ax.set_xlabel("error x (px)")
    ax.set_ylabel("error y (px)")
    fig.colorbar(sc, ax=ax, label="image index")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"cam{camera_id} — reprojection errors"
                 + (f" ({n_out:,} outside the axes)" if n_out else ""), fontsize=11)
    fig.tight_layout()
    return fig


def plot_error_per_tag_robust(
    tag_errors: dict[int, list[float]],
    num_tags: int,
    camera_id: int = 0,
    is_chessboard: bool = False,
) -> Figure:
    """Per-tag/corner median and 95th-percentile error in ONE panel.

    Each tag/corner ID gets a wide light bar for its P95 and a narrow dark bar for its median
    drawn over it. A bar turns red when its value exceeds 1.5x that metric's mean over the
    tags/corners with data (flags tags/corners with anomalous error); each metric's mean is a
    dashed line in its own colour.
    """
    unit = "Corner" if is_chessboard else "Tag"
    tag_ids = np.arange(num_tags)
    med_vals, p95_vals = [], []
    for tid in tag_ids:
        errs = np.asarray(tag_errors.get(int(tid), []), dtype=np.float64)
        if errs.size:
            med_vals.append(float(np.median(errs)))
            p95_vals.append(float(np.percentile(errs, 95)))
        else:
            med_vals.append(0.0)
            p95_vals.append(0.0)

    fig, ax = plt.subplots(figsize=(max(8, num_tags * 0.3), 3.6))

    def _bars(vals, label, width, lo, hi, line):
        nonzero = [v for v in vals if v > 0]
        ref = float(np.mean(nonzero)) if nonzero else 0.0
        colors = [hi if v > ref * 1.5 else lo for v in vals]
        ax.bar(tag_ids, vals, color=colors, edgecolor="white", width=width, label=label)
        if ref > 0:
            ax.axhline(ref, color=line, linestyle="--", linewidth=1, label=f"{label} mean = {ref:.3f} px")

    _bars(p95_vals, "P95", 0.8, "#BBDEFB", "#FFCDD2", "#64B5F6")
    _bars(med_vals, "Median", 0.45, "#1565C0", "#D32F2F", "#FF9800")
    ax.set_xlabel(f"{unit} ID")
    ax.set_ylabel("Error (px)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, borderaxespad=0.2)
    ax.set_title(f"cam{camera_id} — Per-{unit} Error (Median over P95)", fontsize=11, loc="left")
    fig.tight_layout()
    return fig


def plot_tag_tracking_history(
    per_frame_tag_ids: list[set[int] | list[int]],
    num_tags: int,
    camera_id: int = 0,
    is_chessboard: bool = False,
) -> Figure:
    """Frame×Tag/Corner rasterised heatmap — which tag/corner was detected in which frame.

    Args:
        per_frame_tag_ids: list (one entry per frame) of tag/corner ids detected
            in that frame. Order = frame order.
        num_tags: total number of tags/corners on the board.
        camera_id: for the title.
        is_chessboard: If True, label as "Corner" instead of "Tag".

    Output: black cell if tag/corner was detected in that frame, white otherwise.
    Right-axis annotations show the overall detection rate per tag/corner.
    """
    unit = "Corner" if is_chessboard else "Tag"
    n_frames = len(per_frame_tag_ids)
    matrix = np.zeros((num_tags, max(n_frames, 1)), dtype=np.uint8)
    for fi, tids in enumerate(per_frame_tag_ids):
        for t in tids:
            if 0 <= t < num_tags:
                matrix[t, fi] = 1

    # Figure size scales with frame count.
    width = min(max(8, n_frames * 0.02), 20)
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.imshow(1 - matrix, cmap="Greys_r", aspect="auto",
               interpolation="nearest", vmin=0, vmax=1)
    ax.set_xlabel("Frame index")
    ax.set_ylabel(f"{unit} ID")
    ax.set_title(f"cam{camera_id} — {unit} tracking history "
                 f"(black = detected, white = missed)")

    # Label left y-axis by tag id.
    step = max(1, num_tags // 12)
    ax.set_yticks(np.arange(0, num_tags, step))

    # Right y-axis: detection rate per tag (%).
    if n_frames > 0:
        rates = 100.0 * matrix.sum(axis=1) / n_frames
        ax2 = ax.twinx()
        ax2.set_yticks(np.arange(0, num_tags, step))
        ax2.set_yticklabels([f"{rates[i]:.0f}%" for i in
                              range(0, num_tags, step)])
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel("Detection rate (%)")

    # Faint horizontal gridlines between tags.
    ax.set_yticks(np.arange(-0.5, num_tags, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.2)

    fig.tight_layout()
    return fig


def plot_timing_breakdown(timings: dict, camera_id: int | None = None) -> Figure:
    """Horizontal stacked bar + table from a pipeline `timings:` block.

    ``timings`` is the dict produced by ``Timer.to_dict()``; the total
    and per-stage seconds are visualised as a single horizontal stacked
    bar, coloured per stage. Each cell is labelled with the stage name
    and percentage when it exceeds 3% of total.
    """
    sections = (timings or {}).get("sections", [])
    total = timings.get("total_seconds", 0.0) if timings else 0.0
    fig, ax = plt.subplots(figsize=(14, 1.8))

    if not sections or total <= 0:
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center",
                 transform=ax.transAxes)
        ax.axis("off")
        return fig

    colors = [
        "#9FA8DA",  # stage1 soft indigo
        "#81D4FA",  # stage2 sky
        "#FFE082",  # stage3 amber
        "#FFAB91",  # stage4 coral
        "#CE93D8",  # stage5 lilac
        "#B0BEC5",  # fallback grey
    ]

    left = 0.0
    for i, sec in enumerate(sections):
        secs = sec.get("seconds", 0.0)
        name = sec.get("name", f"section{i}")
        color = colors[i % len(colors)]
        ax.barh([0], [secs], left=left, color=color,
                 edgecolor="white", linewidth=1)
        frac = secs / total if total > 0 else 0
        if frac >= 0.03:
            ax.text(left + secs / 2.0, 0, f"{name}\n{secs:.1f}s ({frac*100:.0f}%)",
                     ha="center", va="center", fontsize=9)
        left += secs

    ax.set_xlim(0, total * 1.02)
    ax.set_xlabel("Seconds")
    ax.set_yticks([])
    ax.set_title(f"Pipeline wall-clock breakdown ({total:.1f}s total)")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def plot_feature_coverage_multi(
    all_frames: dict[int, list[FrameObservations]],
    image_sizes: dict[int, tuple[int, int]],
) -> Figure:
    """Plot feature coverage for all cameras side-by-side.

    Each camera gets a subplot showing all detected feature positions
    as a density heatmap. For stereo (2 cameras), plots are side-by-side.

    Coverage is computed as the fraction of the image area (using a grid
    of 50×50 pixel cells) that contains at least one detection.

    Args:
        all_frames: cam_id → list of FrameObservations.
        image_sizes: cam_id → (width, height).

    Returns:
        matplotlib Figure with one subplot per camera.
    """
    cam_ids = sorted(all_frames.keys())
    n_cams = len(cam_ids)
    if n_cams == 0:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    # Layout: up to 3 per row
    n_cols = min(n_cams, 3)
    n_rows = (n_cams + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              squeeze=False)

    for idx, cam_id in enumerate(cam_ids):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        frames = all_frames.get(cam_id, [])
        w, h = image_sizes.get(cam_id, (640, 480))

        us, vs = [], []
        for frame in frames:
            for obs in frame.observations:
                us.append(obs.pixel[0])
                vs.append(obs.pixel[1])

        if not us:
            ax.text(0.5, 0.5, "No detections", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="red")
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_title(f"cam{cam_id}")
            continue

        us = np.array(us)
        vs = np.array(vs)

        # Fine-grid density heatmap (50px cells)
        cell_size = 50
        bins_x = max(w // cell_size, 10)
        bins_y = max(h // cell_size, 10)
        heatmap, xedges, yedges = np.histogram2d(
            us, vs, bins=[bins_x, bins_y],
            range=[[0, w], [0, h]])

        ax.imshow(heatmap.T, origin="upper", extent=[0, w, h, 0],
                  cmap="YlOrRd", alpha=0.7, aspect="equal",
                  interpolation="bilinear")

        # Scatter overlay
        ax.scatter(us, vs, s=0.2, c="#1565C0", alpha=0.1, edgecolors="none")

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")

        # Coverage metrics
        # 1. Grid-cell coverage: fraction of 50px cells with detections
        occupied = int(np.sum(heatmap > 0))
        total_cells = bins_x * bins_y
        cell_coverage = occupied / total_cells * 100

        # 2. Convex hull coverage: hull area / image area
        from mvis.analysis.motion_analysis import hull_coverage
        hull_coverage_pct, hull_pts = hull_coverage(np.column_stack([us, vs]), w, h)
        if hull_pts is not None:              # draw the hull boundary
            ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                    color="#1565C0", linewidth=1.5, linestyle="--", alpha=0.7)

        n_pts = len(us)
        n_frames = len(frames)
        ax.set_title(f"cam{cam_id} — {n_pts:,} pts, "
                     f"{n_frames} frames\n"
                     f"cell coverage: {cell_coverage:.0f}%, "
                     f"hull coverage: {hull_coverage_pct:.0f}%")

    # Hide unused subplots
    for idx in range(n_cams, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.suptitle("Feature Coverage", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_detection_accumulation(
    all_frames: dict[int, list],
    image_sizes: dict[int, tuple[int, int]],
    board=None,
) -> Figure:
    """Plot all detected corners accumulated onto a single image per camera.

    Each frame's detected corners are drawn as small dots, with connecting
    lines for chessboard grids.  The background is white with the image
    boundary shown as a rectangle.  This complements the feature coverage
    heatmap by showing the raw spatial distribution of detections.

    Args:
        all_frames: cam_id → list of FrameObservations.
        image_sizes: cam_id → (width, height).
        board: Board object (used to detect chessboard for grid lines).

    Returns:
        matplotlib Figure.
    """
    cam_ids = sorted(all_frames.keys())
    n_cams = len(cam_ids)
    if n_cams == 0:
        fig, _ = plt.subplots(figsize=(6, 4))
        return fig

    n_cols = min(n_cams, 3)
    n_rows = (n_cams + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(6 * n_cols, 5 * n_rows),
                              squeeze=False)

    is_chess = board is not None and getattr(board, '_num_tags', -1) == 0
    bcols = getattr(board, '_cols', 1) if is_chess else 1

    for idx, cam_id in enumerate(cam_ids):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        frames = all_frames.get(cam_id, [])
        w, h = image_sizes.get(cam_id, (640, 480))

        # White background with image boundary
        ax.set_facecolor("white")
        ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False,
                                     edgecolor="#999", linewidth=1.5,
                                     linestyle="-"))

        n_pts = 0
        for fi, frame in enumerate(frames):
            obs = frame.observations
            if not obs:
                continue

            # Use color cycling per frame (subtle variation)
            alpha = max(0.05, min(0.3, 5.0 / max(len(frames), 1)))
            color = plt.cm.tab10(fi % 10)
            color_alpha = (*color[:3], alpha)

            # Draw grid lines for chessboard
            if is_chess:
                rows_map = {}
                for o in obs:
                    rows_map.setdefault(o.point_id // bcols, []).append(o)
                for rid in sorted(rows_map):
                    pts = sorted(rows_map[rid], key=lambda o: o.point_id)
                    if len(pts) >= 2:
                        xs = [o.pixel[0] for o in pts]
                        ys = [o.pixel[1] for o in pts]
                        ax.plot(xs, ys, color=color_alpha, linewidth=0.5)

            # Draw corner dots
            for o in obs:
                n_pts += 1

            xs = [o.pixel[0] for o in obs]
            ys = [o.pixel[1] for o in obs]
            ax.scatter(xs, ys, s=1, c=[color_alpha], edgecolors="none")

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xlabel("u (px)")
        ax.set_ylabel("v (px)")
        ax.set_title(f"cam{cam_id} — {n_pts:,} detections, "
                     f"{len(frames)} frames")
        ax.set_aspect("equal")

    for idx in range(n_cams, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    fig.suptitle("Detection Accumulation", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_rotation_axis_spread(
    all_metrics: dict,
) -> Figure:
    """Plot rotation axis spread with 3 projections per camera.

    Shows XY, XZ, YZ projections of rotation axis unit vectors
    in the board frame. Good data has points spread across all
    three circles; bad data clusters along one direction.

    Args:
        all_metrics: cam_id → MotionMetrics (with _axis_vectors).

    Returns:
        matplotlib Figure.
    """
    cam_ids = sorted(all_metrics.keys())
    n_cams = len(cam_ids)
    if n_cams == 0:
        fig, _ = plt.subplots(figsize=(4, 4))
        return fig

    projections = [
        (0, 1, "X (board)", "Y (board)"),
        (0, 2, "X (board)", "Z (board)"),
        (1, 2, "Y (board)", "Z (board)"),
    ]

    fig, axes = plt.subplots(n_cams, 3, figsize=(12, 3.5 * n_cams),
                              squeeze=False)

    for row, cam_id in enumerate(cam_ids):
        m = all_metrics[cam_id]

        for col, (i, j, xlabel, ylabel) in enumerate(projections):
            ax = axes[row][col]

            if not hasattr(m, "_axis_vectors") or m._axis_vectors is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                if row == 0:
                    ax.set_title(f"{xlabel} vs {ylabel}")
                continue

            aa = np.array(m._axis_vectors)
            norms = np.linalg.norm(aa, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            dirs = aa / norms

            ax.scatter(dirs[:, i], dirs[:, j], s=6, alpha=0.4, c="#1565C0")
            circle = plt.Circle((0, 0), 1, fill=False, color="#bbb",
                                 linestyle="--", linewidth=0.5)
            ax.add_patch(circle)
            ax.set_xlim(-1.4, 1.4)
            ax.set_ylim(-1.4, 1.4)
            ax.set_aspect("equal")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.15)
            ax.axhline(0, color="#ccc", linewidth=0.3)
            ax.axvline(0, color="#ccc", linewidth=0.3)

            if col == 0:
                ax.set_ylabel(f"cam{cam_id}\n{ylabel}", fontsize=9)

            if row == 0:
                ax.set_title(f"{xlabel} vs {ylabel}", fontsize=10)

    ev_str = ", ".join(f"λ{i+1}={m.axis_eigenvalues[i]:.4f}"
                        for i in range(3)
                        for m in [all_metrics[cam_ids[0]]]
                        if hasattr(m, "axis_eigenvalues"))
    fig.suptitle(f"Rotation Axis Spread (board frame)\n"
                 f"axis ratio: {all_metrics[cam_ids[0]].axis_ratio:.3f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_angular_velocity_timeline(
    all_data: dict,
) -> Figure:
    """Plot angular velocity and RS excitation vs time.

    Two subplots:
      Top: ||ω|| total angular velocity (all cameras overlaid)
      Bottom: ω⊥ RS excitation in camera frame (all cameras overlaid)

    Args:
        all_data: cam_id → {"timestamps": [...], "omega_norms": [...],
                             "omega_perps": [...]}

    Returns:
        matplotlib Figure.
    """
    cam_ids = sorted(all_data.keys())
    n_cams = len(cam_ids)
    if n_cams == 0:
        fig, _ = plt.subplots(figsize=(6, 3))
        return fig

    fig, (ax_total, ax_wx, ax_wy) = plt.subplots(3, 1, figsize=(12, 7),
                                                  sharex=True)
    colors = plt.cm.tab10.colors

    for idx, cam_id in enumerate(cam_ids):
        d = all_data[cam_id]
        ts = d.get("timestamps", [])
        omega = d.get("omega_norms", [])
        omega_xs = d.get("omega_xs", [])
        omega_ys = d.get("omega_ys", [])

        if not ts:
            continue

        t0 = ts[0]
        t_rel = [t - t0 for t in ts]
        c = colors[idx % len(colors)]
        label = f"cam{cam_id}"

        ax_total.plot(t_rel, omega, linewidth=0.8, color=c,
                      label=label, alpha=0.8)
        ax_total.fill_between(t_rel, omega, alpha=0.1, color=c)

        if omega_xs:
            ax_wx.plot(t_rel, omega_xs, linewidth=0.8, color=c,
                       label=label, alpha=0.8)
        if omega_ys:
            ax_wy.plot(t_rel, omega_ys, linewidth=0.8, color=c,
                       label=label, alpha=0.8)

    ax_total.set_ylabel("||ω|| (rad/s)")
    ax_total.set_title("Total Angular Velocity (time offset signal)", fontsize=11)
    ax_total.legend(fontsize=8, loc="upper right")
    ax_total.grid(True, alpha=0.2)
    ax_total.set_ylim(bottom=0)

    ax_wx.set_ylabel("ωx (rad/s)")
    ax_wx.set_title("ωx — Roll (camera frame, RS excitation: horizontal shift per row)",
                     fontsize=11)
    ax_wx.legend(fontsize=8, loc="upper right")
    ax_wx.grid(True, alpha=0.2)

    ax_wy.set_ylabel("ωy (rad/s)")
    ax_wy.set_xlabel("Time (s)")
    ax_wy.set_title("ωy — Pitch (camera frame, RS excitation: vertical shift per row)",
                     fontsize=11)
    ax_wy.legend(fontsize=8, loc="upper right")
    ax_wy.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig


def plot_board_shape(heights_m, board: Board, n_obs=None, title: str = "Board shape") -> Figure:
    """Estimated out-of-plane height of every board corner, drawn at its place on the board (mm).

    Diverging colours, symmetric about the flat plane; corners no camera saw are drawn hollow and
    left out of the colour range and the statistics in the title.

    Args:
        heights_m: per-corner height (m), indexed like `board.all_points_3d()`.
        board: Board object (corner positions in the board plane).
        n_obs: optional per-corner observation counts; 0 marks an unseen corner.
        title: Title prefix.

    Returns:
        Matplotlib Figure.
    """
    pts = np.asarray(board.all_points_3d(), float)
    h = np.asarray(heights_m, float).ravel() * 1e3
    if len(h) != len(pts):
        raise ValueError(f"{len(h)} board heights for a board of {len(pts)} corners")
    seen = np.ones(len(h), bool) if n_obs is None else np.asarray(n_obs).ravel() > 0
    hs = h[seen]
    lim = max(float(np.abs(hs).max()), 1e-3) if hs.size else 1.0
    w, d = np.ptp(pts[:, 0]) or 1.0, np.ptp(pts[:, 1]) or 1.0
    fig, ax = plt.subplots(figsize=(8, max(3.0, min(8.0, 6.5 * d / w + 1.2))))
    size = 60 if len(pts) <= 200 else 24
    sc = ax.scatter(pts[seen, 0], pts[seen, 1], c=hs, cmap="RdBu_r", vmin=-lim, vmax=lim, s=size,
                    edgecolors="none")
    if (~seen).any():
        ax.scatter(pts[~seen, 0], pts[~seen, 1], s=size, facecolors="none", edgecolors="#999999",
                   linewidths=0.8, label="not seen")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.colorbar(sc, ax=ax, label="Height off the flat board (mm)", shrink=0.9)
    ax.set_aspect("equal")
    ax.set_xlabel("Board x (m)")
    ax.set_ylabel("Board y (m)")
    stats = (f"peak to peak {np.ptp(hs):.2f} mm, rms {np.std(hs):.2f} mm, {hs.size} corners seen"
             if hs.size else "no corner seen")
    ax.set_title(f"{title}\n{stats}", fontsize=10)
    fig.tight_layout()
    return fig
