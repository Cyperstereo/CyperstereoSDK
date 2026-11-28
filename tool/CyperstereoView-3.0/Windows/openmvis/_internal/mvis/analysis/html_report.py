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
"""Self-contained HTML calibration report generator.

Embeds all images as base64, matplotlib plots as base64 PNG,
and plotly 3D plots as inline JavaScript. Outputs a single .html file.
"""

import base64
import io
import logging
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly

from mvis.detection.board import Board
from mvis.core.types import CalibrationResult, FrameObservations
from mvis.core.camera_models import DIST_NAMES, KB_THETA_MODELS

logger = logging.getLogger(__name__)

from mvis.analysis.report_shell import (  # shared shell
    ReportStyle, REPORT_STYLE,
    _fig_to_base64, _image_to_base64, _plotly_to_html,
    _grade_cell, render_page,
    config_fold, report_plot_style, result_files_html,
)



# ---------------------------------------------------------------------------
# Report configuration
# ---------------------------------------------------------------------------

def _kb_domain_rows(cameras) -> list:
    """kb4 cameras only: the valid domain of the fit (first theta_d fold and
    its pixel radius vs the image-corner radius); [] when there are none."""
    kb = [c for c in cameras if c.distortion_model in KB_THETA_MODELS]
    if not kb:
        return []
    from mvis.camera.kb_domain import kb_valid_domain
    names = "/".join(sorted({c.distortion_model.upper() for c in kb}))
    out = [f"<h3>{names} Valid Domain</h3>",
           "<p>theta_d(theta) is invertible only up to its first fold; pixels "
           "farther than the fold radius from (cx, cy) have no ray.</p>",
           "<table><tr><th>Camera</th><th>First fold (deg)</th>"
           "<th>Valid radius (px)</th><th>Image corner (px)</th><th>Covers image</th></tr>"]
    for cam in kb:
        info = kb_valid_domain(np.r_[cam.intrinsics, cam.distortion], cam.image_size)
        th = info["fold_theta_rad"]
        cr = info["corner_radius_px"]
        cov = info["covers_image"]
        out.append(
            f"<tr><td>cam{cam.camera_id}</td>"
            f"<td class=\"num\">{'none (monotonic)' if th is None else f'{np.degrees(th):.1f}'}</td>"
            f"<td class=\"num\">{info['max_radius_px']:.1f}</td>"
            f"<td class=\"num\">{'' if cr is None else f'{cr:.1f}'}</td>"
            f"<td>{'' if cov is None else ('yes' if cov else '<strong>NO</strong> (' + ('fold inside image' if th is not None else 'domain ends inside image') + ')')}</td></tr>")
    out.append("</table>")
    return out




def _corner_accounting_rows(result) -> list:
    """What became of each corner at the final poses, per camera (every model);
    [] for a result without the count (older files, a db regeneration)."""
    acc = getattr(result, "corner_accounting", None)
    if not acc:
        return []
    lines = []
    for cid in sorted(acc):
        a = acc[cid]
        why = [(n, a.get(k, 0)) for n, k in (("past the fold", "past_fold"), ("other invalid", "other_invalid"),
                                             ("in no-pose frames", "no_pose_frames"), ("outliers", "outliers"))]
        left = [f"{v:,} {n}" for n, v in why if v]
        lines.append(f"<li><b>cam{cid}</b>: {a['used']:,} corners used"
                     + (f"; left out: {', '.join(left)}" if left else "; none left out")
                     + (" <strong>(fold ALARM)</strong>" if a.get("fold_level") == "alarm" else "") + "</li>")
    out = ["<h3>Corner Accounting</h3>",
           "<p>Corners in frames with a final pose. A corner past the fold or otherwise "
           "out of the model's domain is not in the error figures.</p>",
           "<ul class='acc'>" + "".join(lines) + "</ul>",
           '<details class="appendix"><summary>Full corner accounting (click to expand)</summary>'
           '<div class="appendix-content">',
           "<table><tr><th>Camera</th><th>Used</th><th>Past fold</th><th>Other invalid</th>"
           "<th>No-pose frames</th><th>Outliers</th><th>Max &theta; (deg)</th>"
           "<th>Fold (deg)</th><th>Fold level</th></tr>"]
    for cid in sorted(acc):
        a = acc[cid]
        th, fold = a.get("max_theta_deg"), a.get("fold_deg")
        level = a.get("fold_level", "none")
        n_first = a.get("past_fold_pass1", 0)
        first = f" ({n_first} at the first pass)" if n_first else ""
        out.append(
            f"<tr><td>cam{cid}</td><td class=\"num\">{a['used']}</td>"
            f"<td class=\"num\">{a['past_fold']}{first}</td><td class=\"num\">{a['other_invalid']}</td>"
            f"<td class=\"num\">{a['no_pose_frames']}</td><td class=\"num\">{a['outliers']}</td>"
            f"<td class=\"num\">{'' if th is None else f'{th:.1f}'}</td>"
            f"<td class=\"num\">{'' if fold is None else f'{fold:.1f}'}</td>"
            f"<td>{'<strong>ALARM</strong>' if level == 'alarm' else level}</td></tr>")
    out.append("</table></div></details>")
    return out


def _ucm_kalibr_rows(cameras) -> list[str]:
    """ucm cameras in Kalibr omni-none form: xi = alpha/(1-alpha), gamma = f/(1-alpha)."""
    out = []
    for cam in cameras:
        if cam.distortion_model != "ucm" or not 0.0 <= cam.distortion[0] < 1.0:
            continue
        a = float(cam.distortion[0])
        out.append(f"<p class='note'>cam{cam.camera_id} ucm as Kalibr omni-none: "
                   f"&xi; = {a / (1 - a):.6f}, &gamma; = ({cam.fx / (1 - a):.4f}, "
                   f"{cam.fy / (1 - a):.4f})</p>")
    return out


def _distortion_table_rows(cameras) -> list[str]:
    """Distortion-coefficient table rows (no closing tag); headers adapt per model."""
    rows = []
    # A single model gets its named headers (multi-model tables show model per row)
    models_present = set(c.distortion_model for c in cameras)
    # generic d0..d{N-1}: N = 4 unless a camera carries more (omni p2)
    n_generic = max([4] + [len(c.distortion) for c in cameras])
    generic = tuple(f"d{i}" for i in range(n_generic))
    mixed = len(models_present) != 1
    if not mixed:
        headers = DIST_NAMES.get(models_present.pop(), generic)
    else:
        headers = generic
    rows.append("<table><tr><th>Camera</th><th>Model</th>"
                + "".join(f"<th>{h}</th>" for h in headers if h)
                + "</tr>")
    n_cols = sum(1 for h in headers if h)
    for cam in cameras:
        coefs = list(cam.distortion)
        fmt = "{:.6f}"
        if mixed:
            # generic d0.. header: label each value with its registry name and
            # leave cells past this model's coefficient count blank (no fake 0s)
            names = DIST_NAMES.get(cam.distortion_model, ())
            cells = [(f"{names[i]}=" if i < len(names) else "") + fmt.format(coefs[i])
                     if i < len(coefs) else "" for i in range(n_cols)]
        else:
            coefs += [0.0] * n_cols
            cells = [fmt.format(coefs[i]) for i in range(n_cols)]
        coef_cells = "".join(f"<td class='num'>{c}</td>" for c in cells)
        rows.append(f"<tr><td>cam{cam.camera_id}</td>"
                    f"<td>{cam.distortion_model}</td>"
                    f"{coef_cells}</tr>")
    return rows



def _cam_only_title(title: str, result, calibration_info) -> str:
    """'Camera calibration · <recording(s)> · cam0, cam1' from the cameras' sources; the given title when
    no source is known (it is then a 'dataset / collection' name from the config path)."""
    import os
    ci = (calibration_info or {}).get("cameras_info") or {}
    srcs = []
    for c in sorted(ci):
        sp = (ci[c] or {}).get("source_path")
        if sp:
            b = os.path.basename(str(sp).rstrip("/"))
            if b and b not in srcs:
                srcs.append(b)
    cams = ", ".join(f"cam{c.camera_id}" for c in result.cameras)
    if not srcs:
        return f"Camera calibration · {title}" if title and title != "Calibration Results" else "Camera calibration"
    return f"Camera calibration · {', '.join(srcs)} · {cams}"


def _cam_only_param_folds(result) -> str:
    """Summary: every calibrated parameter of a camera-only result, one closed fold per camera
    (param_folds): intrinsics, named distortion, image size, field of view, readout; for a multi-camera
    solve the pose in the reference camera and the time offset; reprojection statistics; and the camera's
    block of calibration.yaml (OutputWriter.to_calibration_dict) behind a Copy button."""
    from mvis.analysis.param_folds import folds_block, group, num, sensor_fold, val, vec, yaml_block
    from mvis.core.camera_models import dist_names
    from mvis.io.output_writer import OutputWriter
    try:
        doc = OutputWriter.to_calibration_dict(result)
    except Exception:
        doc = {}
    errs = {e.camera_id: e for e in (result.reprojection_errors or [])}
    exts = {x.from_camera: x for x in (result.extrinsics or [])}
    ref = result.reference_camera
    folds = []
    for cam in result.cameras:
        cid = cam.camera_id
        names = dist_names(cam.distortion_model) or [f"d{i}" for i in range(len(cam.distortion))]
        e = errs.get(cid)
        fov = None
        try:
            from mvis.analysis.views.geometry import fov_deg
            from mvis.calibration.factors.camera_projection import k_cal_vec
            if cam.image_size and min(cam.image_size) > 0:
                fov = fov_deg(cam.distortion_model, k_cal_vec(cam.distortion_model, cam.intrinsics[:4], cam.distortion),
                              cam.image_size)
        except Exception:
            fov = None
        rs = float(cam.rolling_shutter_readout or 0.0)
        x = exts.get(cid)
        dt = (result.time_offsets or {}).get(cid)
        key = (f"{cam.distortion_model} · fx {num(cam.fx, 'px2')}"
               + (f" · baseline {num(1e3 * float(np.linalg.norm(x.translation)), 'mm')} mm to cam{ref}" if x is not None else "")
               + (f" · reproj median / P95 {num(e.median, 'px')} / {num(e.p95, 'px')} px" if e is not None else ""))
        groups = [
            group("Intrinsics", [("model", cam.distortion_model),
                                 ("fx", val(cam.fx, "px2")), ("fy", val(cam.fy, "px2")),
                                 ("cx", val(cam.cx, "px2")), ("cy", val(cam.cy, "px2")),
                                 *[(n, val(v, "coef")) for n, v in zip(names, cam.distortion)],
                                 ("image size", f"{cam.image_size[0]} × {cam.image_size[1]}"
                                  if cam.image_size and min(cam.image_size) > 0 else None),
                                 ("field of view H × V", f"{fov['h']:.0f}° × {fov['v']:.0f}°"
                                  if fov and fov.get("h") is not None and fov.get("v") is not None else None)]),
            group("Time", [("readout", "global shutter" if rs == 0.0 else
                            val(1e3 * rs, "ms", 1e3 * cam.rolling_shutter_readout_sigma
                                if cam.rolling_shutter_readout_sigma else None)),
                           (f"time offset to cam{ref}", val(1e3 * dt, "ms") if dt is not None and cid != ref else None)]),
            group(f"Pose in cam{ref} (T_cam{ref}←cam{cid})",
                  [("translation x, y, z (mm)", vec(1e3 * np.asarray(x.translation, float), "mm")),
                   ("|translation|", val(1e3 * float(np.linalg.norm(x.translation)), "mm")),
                   ("log(R) x, y, z (rad)", vec(np.asarray(x.so3, float), "rad")),
                   ("|rotation|", val(float(np.linalg.norm(x.so3)), "rad"))] if x is not None else []),
            group("Reprojection", [("mean", val(e.mean, "px")), ("median", val(e.median, "px")),
                                   ("std", val(e.std, "px") if getattr(e, "std", 0) else None),
                                   ("P90", val(e.p90, "px")), ("P95", val(e.p95, "px")),
                                   ("max", val(e.max, "px") if e.max else None),   # 0 = not recorded (rebuilt from the db)
                                   ("corners", f"{e.num_observations:,}")] if e is not None else []),
        ]
        blk = next((c for c in doc.get("cameras", []) if c.get("id", c.get("camera_id")) == cid), None)
        if blk is not None:
            blk = dict(blk)
            tr = next((t for t in doc.get("extrinsics", {}).get("transforms", []) if t.get("from") == cid), None)
            if tr is not None:
                blk["extrinsics"] = tr
            rp = next((r for r in doc.get("reprojection_errors", {}).get("per_camera", []) if r.get("camera") == cid), None)
            if rp is not None:
                blk["reprojection"] = rp
        folds.append(sensor_fold(f"cam{cid}", key, groups, yaml_block(blk)))
    return folds_block(folds)


@report_plot_style
def generate_html_report(
    result: CalibrationResult,
    all_frames: dict[int, list[FrameObservations]],
    board: Board,
    detection_images: dict[int, list[np.ndarray]],
    error_data: dict,
    plot_3d_fig,
    title: str = "Calibration Results",
    output_path: str = "report.html",
    all_detection_images: dict[int, list[np.ndarray]] | None = None,
    detection_stats: dict | None = None,
    calibration_info: dict | None = None,
    motion_metrics: dict | None = None,
    motion_omega_data: dict | None = None,
    views: dict | None = None,
    appendix_max_frames: int = 0,
):
    """Generate a self-contained HTML calibration report.

    Args:
        result: CalibrationResult from optimizer.
        all_frames: camera_id -> list of FrameObservations.
        board: Board object.
        detection_images: camera_id -> list of sample annotated images (3-5).
        error_data: Dict per camera_id with keys:
            frame_rms, errors, error_vectors, corner_errors,
            tag_errors (tag_id -> [errors]),
            frame_tag_errors (frame_idx -> {tag_id -> rms}),
            tag_detection_counts (tag_id -> count)
        plot_3d_fig: Plotly Figure for 3D visualization (the fallback when `views` has no scene).
        views: canvas view specs (analysis.views): "scene" (the calibration process) and "rig" (multi-camera rig);
            each replaces its Plotly figure when present.
        title: Report title.
        output_path: Path to write HTML file.
        all_detection_images: camera_id -> list of ALL annotated images for appendix.
        appendix_max_frames: Appendix A shows up to this many lowest-detection frames per camera;
            0 leaves Appendix A out (ReportConfig.appendix_max_frames).
        detection_stats: camera_id -> {total_frames, detected_frames, mean_tags, ...}
    """
    from mvis.analysis.plot_errors import (
        plot_per_frame_rms, plot_per_frame_robust,
        plot_error_histogram, plot_error_histogram_uv,
        plot_error_vectors_dual, plot_error_per_tag_robust, plot_reprojection_scatter,
        plot_corner_heatmap, plot_frame_tag_heatmap,
        plot_detection_rate_per_tag,
        plot_tag_tracking_history, plot_timing_breakdown,
        plot_feature_coverage_multi, plot_detection_accumulation,
        plot_rotation_axis_spread, plot_angular_velocity_timeline,
    )
    from mvis.analysis.plot_3d import (
        plot_stereo_extrinsics_3d,
        plot_stereo_extrinsics_3d_interactive,
    )

    sections = []
    toc_entries = []  # (id, label) pairs for TOC
    num_tags = board.num_points() // 4 if hasattr(board, '_num_tags') else board.num_points()
    if hasattr(board, '_num_tags'):
        num_tags = board._num_tags

    # Board type detection
    is_chessboard = not hasattr(board, "_num_tags") or board._num_tags == 0
    unit_label = "Corner" if is_chessboard else "Tag"

    # corners-per-tag divisor for the Tags/Corners column
    corners_per_tag = 4 if hasattr(board, "_num_tags") else 1

    # Section counter for auto-numbering (skips conditional sections)
    style = REPORT_STYLE
    _BG_DEFAULT = style.section_bg
    _BG_ALT = style.section_alt_bg
    sec_num = [0]
    sec_idx = [0]  # total section count for alternating backgrounds
    cur_bg = [_BG_DEFAULT]  # current section background for plot facecolor
    def next_sec():
        sec_num[0] += 1
        return sec_num[0]
    app_letters = iter("ABCDEFGHIJ")    # appendices lettered in the order they appear
    def sec_div(sec_id: str) -> str:
        """Return opening <div> tag with alternating background class."""
        sec_idx[0] += 1
        is_alt = sec_idx[0] % 2 == 0
        cur_bg[0] = _BG_ALT if is_alt else _BG_DEFAULT
        alt = " section-alt" if is_alt else ""
        return f'<div class="section{alt}" id="{sec_id}">'
    def fig_b64(fig) -> str:
        """Convert figure to base64 using current section background."""
        return _fig_to_base64(fig, cur_bg[0], style.plot_dpi)

    # ========== 1. SUMMARY ==========
    sn = next_sec()
    total_obs = sum(e.num_observations for e in result.reprojection_errors)
    total_frames = sum(len(f) for f in all_frames.values())
    n_cameras = len(result.cameras)

    # Overall robust aggregates across all cameras (concat scalar residuals)
    # via the shared Kalibr-convention helper (same code path as cam-IMU)
    from mvis.calibration.reproj_stats import corner_stats
    all_scalar = [np.asarray(ed["errors"]) for ed in error_data.values()
                  if ed.get("errors") is not None and len(ed["errors"]) > 0]
    overall = corner_stats(np.concatenate(all_scalar)
                           if all_scalar else np.empty(0))
    overall_median, overall_p95 = overall["median"], overall["p95"]

    toc_entries.append(("sec-summary", f"{sn}. Summary"))
    # the same Summary as the cam-IMU report: graded tiles, a facts line, the per-camera table, the folds
    from mvis.analysis.verdict import GRADE_DISPLAY, MED_GRADE, P95_GRADE, grade as _vgrade
    from mvis.analysis.report_shell import cam_label, graded, html_table, stat_box

    def _g(v, th):
        return GRADE_DISPLAY.get(_vgrade(v, *th))
    tiles = ("<div class='stat-row'>"
             + stat_box(f"{overall_median:.3f} px", "Reproj median", _g(overall_median, MED_GRADE))
             + stat_box(f"{overall_p95:.3f} px", "Reproj P95", _g(overall_p95, P95_GRADE))
             + stat_box(n_cameras, "Cameras") + stat_box(f"{total_frames:,}", "Frames")
             + stat_box(f"{total_obs:,}", "Observations") + "</div>")
    facts = [("Board", board.__class__.__name__.replace("Board", " board").strip()),
             ("Points", f"{board.num_points()}")] + ([("Tags", f"{num_tags}")] if not is_chessboard else [])
    if result.reference_camera is not None and n_cameras > 1:
        facts.append(("Reference", f"cam{result.reference_camera}"))
    facts_html = "<div class='facts'>" + "".join(f"<span><b>{k}</b> {v}</span>" for k, v in facts) + "</div>"
    per_cam = html_table(
        ["Camera", "Mean (px)", "Median (px)", "Std (px)", "P90 (px)", "P95 (px)", "Max (px)", "Corners"],
        [[cam_label(e.camera_id), f"{e.mean:.3f}", graded(f"{e.median:.3f}", _g(e.median, MED_GRADE)),
          f"{e.std:.3f}" if e.std else "—", f"{e.p90:.3f}", graded(f"{e.p95:.3f}", _g(e.p95, P95_GRADE)),
          f"{e.max:.2f}" if e.max else "—", f"{e.num_observations:,}"]
         for e in result.reprojection_errors])
    sections.append(f"""
    {sec_div("sec-summary")}
    <h2>{sn}. Summary</h2>
    {tiles}{facts_html}
    <h3>Reprojection per camera</h3>{per_cam}""")
    sections.append(_cam_only_param_folds(result))

    # Diverged pose stats
    dp = getattr(result, "diverged_pose_stats", None)
    if dp and dp.get("removed", 0) > 0:
        sections.append(f"""
    <h3>Diverged Pose Removal</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total board poses</td><td>{dp['total']}</td></tr>
        <tr><td>Removed (diverged)</td><td style="color:var(--danger);font-weight:bold">{dp['removed']}</td></tr>
        <tr><td>Kept</td><td>{dp['kept']}</td></tr>
    </table>""")

    sections.append("</div>")

    # ========== CAMERA INTRINSICS ==========
    sn = next_sec()
    toc_entries.append(("sec-intrinsics", f"{sn}. Camera Intrinsics"))
    sections.append(f'{sec_div("sec-intrinsics")}<h2>{sn}. Camera Intrinsics</h2>')

    # Linear intrinsics
    sections.append("<h3>Linear Intrinsics</h3>")
    sections.append("""<table>
        <tr><th>Camera</th><th>Model</th><th>fx (px)</th><th>fy (px)</th><th>cx (px)</th><th>cy (px)</th>
            <th>Image Size</th></tr>""")
    for cam in result.cameras:
        sections.append(f"""<tr>
            <td>cam{cam.camera_id}</td><td>pinhole + {cam.distortion_model}</td>
            <td class="num">{cam.fx:.2f}</td><td class="num">{cam.fy:.2f}</td>
            <td class="num">{cam.cx:.2f}</td><td class="num">{cam.cy:.2f}</td>
            <td>{cam.image_size[0]}&times;{cam.image_size[1]}</td></tr>""")
    sections.append("</table>")

    # Distortion coefficients — column headers adapt per model
    sections.append("<h3>Distortion Coefficients</h3>")
    sections.extend(_distortion_table_rows(result.cameras))
    sections.append("</table>")
    sections.extend(_ucm_kalibr_rows(result.cameras))
    sections.extend(_kb_domain_rows(result.cameras))
    sections.extend(_corner_accounting_rows(result))
    sections.append("</div>")

    # ========== BOARD SHAPE (only when estimate.board_points was on) ==========
    if getattr(result, "board_heights", None) is not None:
        from mvis.analysis.plot_errors import plot_board_shape
        h_mm = np.asarray(result.board_heights, float) * 1e3
        # corners the solve never used (e.g. the two per tag dropped by two_anti) stay at the prior
        n_obs = getattr(result, "board_height_n_obs", None)
        seen = n_obs > 0 if n_obs is not None else np.ones(len(h_mm), bool)
        h_seen = h_mm[seen] if seen.any() else h_mm
        sn = next_sec()
        toc_entries.append(("sec-board-shape", f"{sn}. Board Shape"))
        sections.append(
            f'{sec_div("sec-board-shape")}<h2>{sn}. Board Shape</h2>'
            "<p>Per-corner height off the flat board, estimated with the cameras "
            "(<code>estimate.board_points</code>). The mean plane and tilt belong to the board "
            "pose, so a flat board shows only noise here.</p>"
            "<table><tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Peak to peak</td><td class='num'>{np.ptp(h_seen):.2f} mm</td></tr>"
            f"<tr><td>RMS</td><td class='num'>{np.std(h_seen):.2f} mm</td></tr>"
            f"<tr><td>Corners seen</td><td class='num'>{int(seen.sum())} of {len(h_mm)}</td></tr></table>"
            f'<div class="plot-container"><img src="data:image/png;base64,'
            f'{fig_b64(plot_board_shape(result.board_heights, board, n_obs))}"></div></div>')

    # ========== CAMERA EXTRINSICS ==========
    if result.extrinsics:
        sn = next_sec()
        toc_entries.append(("sec-extrinsics", f"{sn}. Camera Extrinsics"))
        sections.append(f'{sec_div("sec-extrinsics")}<h2>{sn}. Camera Extrinsics</h2>')
        sections.append(f"<p>Reference camera: <strong>{result.reference_camera}</strong></p>")
        sections.append("""<table>
            <tr><th>From</th><th>To</th><th>Translation (m)</th>
                <th>Rotation SO(3) (rad)</th><th>|Rotation| (rad)</th>
                <th>Baseline (m)</th></tr>""")
        for ext in result.extrinsics:
            t_str = f"[{ext.translation[0]:.6f}, {ext.translation[1]:.6f}, {ext.translation[2]:.6f}]"
            r_str = f"[{ext.so3[0]:.6f}, {ext.so3[1]:.6f}, {ext.so3[2]:.6f}]"
            rot_rad = float(np.linalg.norm(ext.so3))
            baseline = float(np.linalg.norm(ext.translation))
            sections.append(f"""<tr><td>cam{ext.from_camera}</td><td>cam{ext.to_camera}</td>
                <td>{t_str}</td><td>{r_str}</td><td class="num">{rot_rad:.6f}</td>
                <td class="num">{baseline:.6f}</td></tr>""")
        sections.append("</table>")

        # The solved rig (canvas view), else the interactive 3D extrinsics plot
        rig_spec = (views or {}).get("rig")
        if rig_spec:
            from mvis.analysis.views import assets_html, rig_html
            sections.append(assets_html())
            sections.append(rig_html(rig_spec, lead="The solved rig to scale, in the reference camera's frame. Tap a camera "
                                     "for its values. Field-of-view rims are traced through each lens model from the image edges."))
        else:
            try:
                fig_ext = plot_stereo_extrinsics_3d_interactive(
                    result.extrinsics, result.reference_camera)
                sections.append(
                    f'<div class="plot-container">{_plotly_to_html(fig_ext)}</div>'
                )
            except Exception as e:
                try:
                    fig = plot_stereo_extrinsics_3d(
                        result.extrinsics, result.reference_camera)
                    sections.append(
                        f'<div class="plot-container">'
                        f'<img src="data:image/png;base64,{fig_b64(fig)}"></div>'
                    )
                except Exception:
                    sections.append(f"<p class='note'>3D extrinsics plot failed: {e}</p>")

        sections.append("</div>")

    # ========== TEMPORAL CALIBRATION (Time Offset + Rolling Shutter) ==========
    has_time_offsets = bool(result.time_offsets)
    has_rs = any(abs(cam.rolling_shutter_readout) > 1e-9 for cam in result.cameras)
    if has_time_offsets or has_rs:
        sn = next_sec()
        toc_entries.append(("sec-temporal", f"{sn}. Temporal Calibration"))
        sections.append(f'{sec_div("sec-temporal")}<h2>{sn}. Temporal Calibration</h2>')

        if has_time_offsets:
            sections.append(f"<h3>Time Offset</h3>")
            sections.append(f"<p>Reference camera: <strong>{result.reference_camera}</strong></p><table>")
            sections.append("<tr><th>Camera</th><th>dt (seconds)</th></tr>")
            for cam_id, dt in result.time_offsets.items():
                sections.append(f"<tr><td>cam{cam_id}</td><td class='num'>{dt:.9f}</td></tr>")
            sections.append("</table>")

        if has_rs:
            sections.append("<h3>Rolling Shutter Readout</h3>")
            sections.append("<table><tr><th>Camera</th><th>Readout Time (s)</th></tr>")
            for cam in result.cameras:
                sections.append(f"<tr><td>cam{cam.camera_id}</td>"
                                f"<td class='num'>{cam.rolling_shutter_readout:.6f}</td></tr>")
            sections.append("</table>")

        sections.append("</div>")

    # ========== 3D CAMERA POSES & BOARD ==========
    scene_spec = (views or {}).get("scene")
    if scene_spec or plot_3d_fig is not None:
        sn = next_sec()
        toc_entries.append(("sec-3d", f"{sn}. Calibration Process" if scene_spec else f"{sn}. 3D Camera Poses"))
        if scene_spec:
            from mvis.analysis.views import assets_html, scene_html
            sections.append(f'{sec_div("sec-3d")}<h2>{sn}. Calibration Process</h2>')
            if not (views or {}).get("rig"):   # the rig section put the player in already
                sections.append(assets_html())
            sections.append(scene_html(scene_spec, lead="The solved path of the reference camera around the board. Press Play "
                                       "or drag the bar to follow it, use the buttons to change the view, and drag to rotate. "
                                       "Click a camera on the right to put it on the frustum."))
        else:
            sections.append(f'{sec_div("sec-3d")}<h2>{sn}. 3D Camera Poses & Board</h2>')
            sections.append(f'<div class="plot-container">{_plotly_to_html(plot_3d_fig)}</div>')

        # (the static "Corner Coverage" companion was dropped 2026-10-25: the
        # interactive plot above marks solver-kept / dropped corners itself)

        sections.append("</div>")

    # ========== REPROJECTION ERROR ANALYSIS ==========
    sn = next_sec()
    toc_entries.append(("sec-reproj", f"{sn}. Reprojection Errors"))
    sections.append(f'{sec_div("sec-reproj")}<h2>{sn}. Reprojection Error Analysis</h2>')
    sections.append(f"""<h3>Summary</h3><table>
        <tr><th>Camera</th><th>Mean (px)</th><th>Median (px)</th><th>Std (px)</th>
            <th>P90 (px)</th><th>P95 (px)</th>
            <th>Max (px)</th><th>{unit_label}s</th><th>Observations</th></tr>""")
    for err in result.reprojection_errors:
        tags = err.num_observations // corners_per_tag
        sections.append(f"""<tr><td>cam{err.camera_id}</td>
            <td class="num">{err.mean:.3f}</td><td class="num">{err.median:.3f}</td>
            <td class="num">{f"{err.std:.3f}" if err.std else "—"}</td><td class="num">{err.p90:.3f}</td>
            <td class="num">{err.p95:.3f}</td>
            <td class="num">{f"{err.max:.3f}" if err.max else "—"}</td><td class="num">{tags:,}</td>
            <td class="num">{err.num_observations:,}</td></tr>""")
    if is_chessboard:
        sections.append("<p class='note'>Corners = individual chessboard corner observations.</p>")
    else:
        sections.append(f"<p class='note'>Tags = Observations / {corners_per_tag} "
                        f"(corners per tag).</p>")
    sections.append("</table></div>")

    # Per-camera error plots
    for cam_id, edata in sorted(error_data.items()):
        sec_id = f"sec-cam-{cam_id}"
        toc_entries.append((sec_id, f"  cam{cam_id} Errors"))
        sections.append(f'{sec_div(sec_id)}<h3>cam{cam_id} — Error Analysis</h3>')

        # Prefer robust (median + p95) plot
        if "frame_median" in edata and "frame_p95" in edata:
            fig = plot_per_frame_robust(
                edata["frame_median"], edata["frame_p95"], cam_id)
            sections.append(f'<div class="plot-container"><img src="data:image/png;base64,{fig_b64(fig)}"></div>')
        elif "frame_rms" in edata:
            fig = plot_per_frame_rms(edata["frame_rms"], cam_id)
            sections.append(f'<div class="plot-container"><img src="data:image/png;base64,{fig_b64(fig)}"></div>')

        if "tag_errors" in edata and edata["tag_errors"]:
            fig = plot_error_per_tag_robust(edata["tag_errors"], num_tags, cam_id,
                                           is_chessboard=is_chessboard)
            sections.append(f'<div class="plot-container"><img src="data:image/png;base64,{fig_b64(fig)}"></div>')

        if "error_vectors" in edata and len(edata["error_vectors"]) > 0:
            fig = plot_error_histogram_uv(edata["error_vectors"], cam_id)
            sections.append(f'<div class="plot-container"><img src="data:image/png;base64,{fig_b64(fig)}"></div>')

            fig = plot_error_vectors_dual(
                edata["error_vectors"],
                edata.get("error_vectors_by_frame"),
                cam_id,
            )
            sections.append(f'<div class="plot-container"><img src="data:image/png;base64,{fig_b64(fig)}"></div>')
        elif "errors" in edata and len(edata["errors"]) > 0:
            fig = plot_error_histogram(edata["errors"], cam_id)
            sections.append(f'<div class="plot-container"><img src="data:image/png;base64,{fig_b64(fig)}"></div>')

        # one row: Kalibr-style scatter (coloured by image index) beside the corner heatmap
        row = []
        if edata.get("error_vectors_by_frame"):
            row.append(plot_reprojection_scatter(edata["error_vectors_by_frame"], cam_id))
        if "corner_errors" in edata and edata["corner_errors"]:
            row.append(plot_corner_heatmap(edata["corner_errors"], board, cam_id))
        imgs = "".join(f'<div class="plot-container"><img src="data:image/png;base64,{fig_b64(f)}"></div>'
                       for f in row)
        if imgs:
            sections.append(f'<div class="plot-row">{imgs}</div>' if len(row) == 2 else imgs)

        sections.append("</div>")

    # ========== CALIBRATION DATA QUALITY ==========
    sn = next_sec()
    toc_entries.append(("sec-quality", f"{sn}. Data Quality"))
    sections.append(f'{sec_div("sec-quality")}<h2>{sn}. Calibration Data Quality</h2>')
    # folded by default (since 2026-10-25): heading + TOC anchor stay
    sections.append('<details class="appendix"><summary>Motion excitation, '
                    'assessment and feature coverage (click to expand)</summary>'
                    '<div class="appendix-content">')

    if motion_metrics:
        from mvis.analysis.motion_analysis import assess_motion_quality
        sections.append("""<h3>Motion Excitation</h3>
        <p class="desc">
        Motion diversity of the calibration board across all frames.
        Good calibration requires varied translation (different distances and positions),
        rotation (board tilted around all 3 axes), and angular velocity (fast motion for
        time offset and rolling shutter estimation).
        <em>Thresholds are empirical recommendations.</em></p>""")

        # Metrics table
        sections.append("""<table>
            <tr><th>Metric</th><th>Description</th>""")
        for cam_id in sorted(motion_metrics.keys()):
            sections.append(f"<th>cam{cam_id}</th>")
        sections.append("</tr>")

        # Board size info
        board_size_str = ""
        if hasattr(board, "_num_tags"):
            if hasattr(board, "_tag_size"):
                ts = board._tag_size
                sp = getattr(board, "_tag_spacing", 0.3)
                cols = getattr(board, "_cols", 6)
                rows = getattr(board, "_rows", 6)
                board_w_m = cols * ts * (1 + sp)
                board_h_m = rows * ts * (1 + sp)
                board_size_str = f"{board_w_m:.3f} &times; {board_h_m:.3f}"

        rows_data = [
            ("Poses", "Number of valid board poses", lambda m: f"{m.n_poses}"),]
        if board_size_str:
            rows_data.append(
                ("Board size (m)", "Physical board dimensions", lambda m: board_size_str))
        rows_data += [
            ("x range (m)", "Horizontal movement extent", lambda m: f"{m.x_range:.3f}"),
            ("y range (m)", "Vertical movement extent", lambda m: f"{m.y_range:.3f}"),
            ("z range (m)", "Depth variation", lambda m: f"{m.z_range:.3f}"),
            ("z ratio", "max(z)/min(z) — depth diversity for focal length", lambda m: f"{m.z_ratio:.2f}"),
            ("Volume (m&sup3;)", "Translation diversity (x&times;y&times;z)", lambda m: f"{m.volume:.4f}"),
            ("Inter-frame rotation (&deg;)", "Min&ndash;max angular change between consecutive frames", lambda m: f"{m.rot_angle_min:.1f} &ndash; {m.rot_angle_max:.1f}"),
            ("Rotation variation (&deg;)", "Std of inter-frame angular changes", lambda m: f"{m.rot_angle_std:.1f}"),
            ("Axis eigenvalues", "Rotation axis covariance eigenvalues &lambda;1, &lambda;2, &lambda;3", lambda m: f"{m.axis_eigenvalues[0]:.4f}, {m.axis_eigenvalues[1]:.4f}, {m.axis_eigenvalues[2]:.4f}"),
            ("Axis ratio", "&lambda;min/&lambda;max (1=isotropic rotation, 0=single-axis)", lambda m: f"{m.axis_ratio:.3f}"),
            ("&omega; median (rad/s)", "Typical angular speed", lambda m: f"{m.omega_median:.2f}"),
            ("&omega; max (rad/s)", "Peak angular speed", lambda m: f"{m.omega_max:.2f}"),
            ("&omega; std (rad/s)", "Speed variation (time offset signal)", lambda m: f"{m.omega_std:.2f}"),
            ("RS excitation (rad/s)", "Peak &radic;(&omega;<sub>x</sub>&sup2;+&omega;<sub>y</sub>&sup2;) in camera frame", lambda m: f"{m.rs_excitation:.2f}"),
        ]

        for label, desc, fmt_fn in rows_data:
            sections.append(f"<tr><td><strong>{label}</strong></td>"
                           f"<td style='font-size:12px;color:var(--text-secondary)'>{desc}</td>")
            for cam_id in sorted(motion_metrics.keys()):
                sections.append(f"<td class='num'>{fmt_fn(motion_metrics[cam_id])}</td>")
            sections.append("</tr>")
        sections.append("</table>")

        # Assessment with metric explanation
        assessment_params = [
            ("Intrinsics (fx)", "z_ratio", "intrinsics_fx",
             "z_ratio: >2.0 good, 1.5-2.0 marginal, <1.5 poor"),
            ("Distortion", "hull coverage", "distortion",
             "Hull coverage: >60% good, 30-60% marginal, <30% poor"),
            ("Rotation Diversity", "axis_ratio", "rotation_diversity",
             "&lambda;min/&lambda;max: >0.1 good, 0.03-0.1 marginal, <0.03 poor"),
            ("Time Offset", "&omega; std", "time_offset",
             "&omega; std (rad/s): >0.5 good, 0.2-0.5 marginal, <0.2 poor"),
            ("RS Readout", "RS excitation", "rs_readout",
             "Peak &radic;(&omega;x&sup2;+&omega;y&sup2;): >1.0 good, 0.3-1.0 marginal, <0.3 poor"),
        ]

        sections.append("<h3>Assessment</h3><table>"
                        "<tr><th>Parameter</th><th>Metric</th>")
        for cam_id in sorted(motion_metrics.keys()):
            sections.append(f"<th>cam{cam_id}</th>")
        sections.append("</tr>")

        for label, metric, key, desc in assessment_params:
            sections.append(f"<tr><td><strong>{label}</strong><br>"
                           f"<span style='font-size:11px;color:var(--text-muted)'>{desc}</span></td>"
                           f"<td>{metric}</td>")
            for cam_id in sorted(motion_metrics.keys()):
                grades = assess_motion_quality(motion_metrics[cam_id],
                                                calibrate_rs=True, calibrate_dt=True)
                sections.append(_grade_cell(grades.get(key, "—")))
            sections.append("</tr>")

        sections.append("</table>")

        # Rotation axis spread plot
        fig_axis = plot_rotation_axis_spread(motion_metrics)
        sections.append(
            f'<div class="plot-container">'
            f'<img src="data:image/png;base64,{fig_b64(fig_axis)}">'
            f'</div>')

        # Angular velocity timeline
        if motion_omega_data:
            fig_omega = plot_angular_velocity_timeline(motion_omega_data)
            sections.append(
                f'<div class="plot-container">'
                f'<img src="data:image/png;base64,{fig_b64(fig_omega)}">'
                f'</div>')

    # Feature coverage map
    _img_sizes = {cam.camera_id: cam.image_size for cam in result.cameras
                  if hasattr(cam, "image_size") and cam.image_size != (0, 0)}
    if not _img_sizes:
        for cid, frames_list in all_frames.items():
            if frames_list and hasattr(frames_list[0], "image_size"):
                _img_sizes[cid] = frames_list[0].image_size

    if all_frames:
        sections.append("""<h3>Feature Coverage</h3>
        <div class="desc">
        <p style="margin:0 0 6px 0;">
        All detected feature positions across all frames overlaid on a density heatmap.</p>
        <ul style="margin:4px 0;padding-left:20px;line-height:1.6;">
        <li><strong>Yellow&ndash;red heatmap</strong>: detection density
            (brighter = more detections in that region).</li>
        <li><strong>Blue dots</strong>: individual corner detections.</li>
        <li><strong>Dashed blue line</strong>: convex hull &mdash; the outermost
            boundary enclosing all detections.</li>
        </ul>
        <table style="font-size:13px;border-collapse:collapse;margin:6px 0;border:none;">
        <tr>
          <td style="padding:3px 10px 3px 0;vertical-align:top;border:none;"><em>Cell coverage</em></td>
          <td style="padding:3px 0;border:none;">Percentage of 50&times;50 px grid cells containing
              at least one detection (reveals gaps where the board was never observed).</td>
        </tr>
        <tr>
          <td style="padding:3px 10px 3px 0;vertical-align:top;border:none;"><em>Hull coverage</em></td>
          <td style="padding:3px 0;border:none;">Convex hull area as a percentage of the full image area
              (measures how much of the image extent the board reached).</td>
        </tr>
        </table>
        <p style="margin:6px 0 0 0;">
        A good calibration has high coverage in both metrics, with detections spread
        across the full image &mdash; especially in corners and edges.</p>
        </div>""")
        fig = plot_feature_coverage_multi(all_frames, _img_sizes)
        sections.append(
            f'<div class="plot-container">'
            f'<img src="data:image/png;base64,{fig_b64(fig)}">'
            f'</div>')

        fig_acc = plot_detection_accumulation(all_frames, _img_sizes,
                                              board=board)
        sections.append(
            f'<div class="plot-container">'
            f'<img src="data:image/png;base64,{fig_b64(fig_acc)}">'
            f'</div>')

    sections.append("</div></details></div>")

    # ========== DETECTION STATISTICS ==========
    sn = next_sec()
    toc_entries.append(("sec-detection", f"{sn}. Detection Statistics"))
    sections.append(f'{sec_div("sec-detection")}<h2>{sn}. Detection Statistics</h2>')
    if detection_stats:
        sections.append("""<table>
            <tr><th>Camera</th><th>Total Frames</th><th>Detected</th>
                <th>Rate</th><th>Mean Tags/Frame</th><th>Min Tags</th><th>Max Tags</th></tr>""")
        for cam_id, stats in sorted(detection_stats.items()):
            rate = stats["detected_frames"] / max(stats["total_frames"], 1) * 100
            sections.append(f"""<tr>
                <td>cam{cam_id}</td><td class="num">{stats['total_frames']}</td>
                <td class="num">{stats['detected_frames']}</td><td class="num">{rate:.0f}%</td>
                <td class="num">{stats['mean_tags']:.1f}</td><td class="num">{stats['min_tags']}</td>
                <td class="num">{stats['max_tags']}</td></tr>""")
        sections.append("</table>")

    # Per-tag detection bar chart + tag tracking history heatmap
    for cam_id, edata in sorted(error_data.items()):
        if "tag_detection_counts" in edata:
            fig = plot_detection_rate_per_tag(
                edata["tag_detection_counts"], num_tags,
                len(all_frames.get(cam_id, [])), cam_id,
                is_chessboard=is_chessboard)
            sections.append(f'<div class="plot-container"><img src="data:image/png;base64,{fig_b64(fig)}"></div>')

        # Tag/corner tracking history
        frames_cam = all_frames.get(cam_id, [])
        if frames_cam:
            per_frame_ids = []
            for f in frames_cam:
                if is_chessboard:
                    per_frame_ids.append(
                        {o.point_id for o in f.observations}
                    )
                else:
                    per_frame_ids.append(
                        {o.point_id // 4 for o in f.observations}
                    )
            fig = plot_tag_tracking_history(
                per_frame_ids, num_tags, cam_id,
                is_chessboard=is_chessboard)
            sections.append(f'<div class="plot-container">'
                            f'<img src="data:image/png;base64,{fig_b64(fig)}"></div>')

    # Sample detection images
    if detection_images:
        sections.append("<h3>Sample Detections</h3>")
        for cam_id, images in sorted(detection_images.items()):
            sections.append(f"<h4>cam{cam_id}</h4>")
            sections.append('<div class="img-grid">')
            for img in images[:5]:
                b64 = _image_to_base64(img)
                sections.append(f'<img src="data:image/png;base64,{b64}">')
            sections.append("</div>")
    sections.append("</div>")

    # ========== PIPELINE TIMING ==========
    tim = getattr(result, "timings", None)
    if tim and tim.get("sections"):
        al = next(app_letters)
        toc_entries.append(("sec-timing", f"Appendix {al} · Timing"))
        sections.append(f'{sec_div("sec-timing")}<h2>Appendix {al}: Pipeline Wall-Clock Breakdown</h2>'
                        '<details class="appendix"><summary>Stage timings (click to expand)</summary>'
                        '<div class="appendix-content">')
        try:
            fig = plot_timing_breakdown(tim)
            sections.append(
                f'<div class="plot-container">'
                f'<img src="data:image/png;base64,{fig_b64(fig)}"></div>'
            )
        except Exception as e:
            sections.append(f"<p class='note'>Timing plot failed: {e}</p>")

        # Table of stages with metadata
        sections.append(f"<p>Total: <strong>{tim.get('total_seconds', 0):.1f}s</strong></p>")
        sections.append("<table><tr><th>Stage</th><th>Seconds</th>"
                        "<th>%</th><th>Metadata</th></tr>")
        total = tim.get("total_seconds", 0.0) or 1.0
        for s in tim["sections"]:
            pct = 100.0 * s["seconds"] / total
            meta = ", ".join(f"{k}={v}" for k, v in s.get("metadata", {}).items())
            sections.append(
                f"<tr><td>{s['name']}</td>"
                f"<td class='num'>{s['seconds']:.2f}</td>"
                f"<td class='num'>{pct:.1f}%</td>"
                f"<td>{meta}</td></tr>"
            )
            for c in s.get("children", []) or []:
                cmeta = ", ".join(f"{k}={v}" for k, v in c.get("metadata", {}).items())
                cpct = 100.0 * c["seconds"] / total
                sections.append(
                    f"<tr><td>&nbsp;&nbsp;&#8627; {c['name']}</td>"
                    f"<td class='num'>{c['seconds']:.2f}</td>"
                    f"<td class='num'>{cpct:.1f}%</td>"
                    f"<td>{cmeta}</td></tr>"
                )
        sections.append("</table></div></details></div>")

    # ========== RESULT FILES (verbatim yaml, folded) ==========
    res_html = result_files_html((calibration_info or {}).get("result_files"))
    if res_html:
        al = next(app_letters)
        toc_entries.append(("sec-results", f"Appendix {al} · Result files"))
        sections.append(f'{sec_div("sec-results")}<h2>Appendix {al}: Result Files</h2>'
                        f'{res_html}</div>')

    # ========== CALIBRATION CONFIGURATION ==========
    if calibration_info:
        ci = calibration_info
        al = next(app_letters)
        toc_entries.append(("sec-config", f"Appendix {al} · Configuration"))
        sections.append(f'{sec_div("sec-config")}<h2>Appendix {al}: Calibration Configuration</h2>')
        # Global paths
        rows = []
        if ci.get("config_path"):
            rows.append(f"<tr><td>Config YAML</td><td><code>{ci['config_path']}</code></td></tr>")
        if ci.get("output_report_path"):
            rows.append(f"<tr><td>Report path</td><td><code>{ci['output_report_path']}</code></td></tr>")
        if ci.get("calibration_yaml_path"):
            rows.append(f"<tr><td>Calibration YAML</td><td><code>{ci['calibration_yaml_path']}</code></td></tr>")
        if ci.get("reference_camera") is not None:
            rows.append(f"<tr><td>Reference camera</td><td>{ci['reference_camera']}</td></tr>")
        if rows:
            sections.append("<table>" + "".join(rows) + "</table>")

        # Per-camera config table
        cams_info = ci.get("cameras_info", {}) or {}
        if cams_info:
            sections.append("""<h3>Per-Camera Source</h3>
                <table>
                <tr><th>ID</th><th>Name</th><th>Source</th><th>Path / Topic</th>
                    <th>Distortion</th><th>Frames</th><th>Hz</th><th>Duration (s)</th></tr>""")
            cam_name = {c.camera_id: getattr(c, "name", None) or "" for c in result.cameras}
            dist_model = {c.camera_id: c.distortion_model for c in result.cameras}
            for cid in sorted(cams_info.keys()):
                d = cams_info[cid]
                name = d.get("name") or cam_name.get(cid, "")
                src_type = d.get("source_type", "")
                path = d.get("topic") or d.get("source_path") or ""
                distm = d.get("distortion_model") or dist_model.get(cid, "")
                nframes = d.get("n_frames_used", "")
                hz = d.get("effective_hz", None)
                dur = d.get("duration_s", None)
                hz_s = f"{hz:.2f}" if isinstance(hz, (int, float)) else ""
                dur_s = f"{dur:.1f}" if isinstance(dur, (int, float)) else ""
                sections.append(f"""<tr>
                    <td>cam{cid}</td><td>{name}</td><td>{src_type}</td><td><code>{path}</code></td>
                    <td>{distm}</td><td class="num">{nframes}</td><td class="num">{hz_s}</td>
                    <td class="num">{dur_s}</td></tr>""")
            sections.append("</table>")
        # verbatim run config, folded (same format as the cam-IMU report)
        cfg_path = ci.get("config_path")
        if cfg_path and os.path.isfile(str(cfg_path)):
            try:
                with open(cfg_path) as f:
                    sections.append(config_fold(f.read()))
            except (OSError, UnicodeDecodeError):
                pass
        sections.append("</div>")

    # ========== APPENDIX A: LOWEST-DETECTION FRAMES ==========
    if all_detection_images and appendix_max_frames > 0:
        al = next(app_letters)
        toc_entries.append(("sec-appendix-a", f"Appendix {al} · Lowest-detection frames"))
        sections.append(f'{sec_div("sec-appendix-a")}<h2>Appendix {al}: Lowest-Detection Frames</h2>')
        appendix_cap = appendix_max_frames
        sections.append(
            '<details class="appendix"><summary>'
            f'Up to {appendix_cap} frames per camera with the fewest detected tags '
            f'(click to expand)</summary><div class="appendix-content">'
        )
        sections.append(
            f'<p class="desc">Each image shows detected corners overlaid by the detector. '
            f'Useful for spotting poorly-detected frames where the board was partially '
            f'occluded, motion-blurred, or too small in frame.</p>'
        )
        for cam_id, images in sorted(all_detection_images.items()):
            if len(images) == 0:
                continue
            frames_cam = all_frames.get(cam_id, [])
            if not frames_cam:
                continue

            cpt = 4 if hasattr(board, "_num_tags") and \
                getattr(board, "_num_tags", 0) > 0 else 1
            expected_tags = num_tags

            pairs = []
            n_available = min(len(frames_cam), len(images))
            for fi in range(n_available):
                f = frames_cam[fi]
                n_obs = len(f.observations)
                n_tags_frame = (len({o.point_id // 4 for o in f.observations})
                                if cpt == 4 else n_obs)
                pairs.append((fi, n_tags_frame, n_obs, f.timestamp))
            if not pairs:
                continue

            pairs.sort(key=lambda x: x[1])
            lowest = pairs[:appendix_cap]

            unit_lc = unit_label.lower()
            sections.append(
                f"<h3>cam{cam_id} — lowest {len(lowest)} by {unit_lc} count "
                f"(expected {expected_tags})</h3>")
            for fi, n_tags_frame, n_obs, ts in lowest:
                img = images[fi]
                b64 = _image_to_base64(img)
                pct = 100.0 * n_tags_frame / max(expected_tags, 1)
                sections.append(
                    f'<div class="appendix-item">'
                    f'<img src="data:image/png;base64,{b64}">'
                    f'<div class="caption">'
                    f'<b>f{fi}</b> &middot; ts={ts:.3f}s<br>'
                    f'{n_tags_frame}/{expected_tags} {unit_lc}s ({pct:.0f}%) &middot; '
                    f'{n_obs} obs'
                    f'</div></div>')
        sections.append("</div></details></div>")

    # ========== APPENDIX B: PER-FRAME PER-TAG ERRORS ==========
    has_heatmap = any("frame_tag_errors" in ed and ed["frame_tag_errors"]
                      for ed in error_data.values())
    if has_heatmap:
        al = next(app_letters)
        toc_entries.append(("sec-appendix-b", f"Appendix {al} · Per-frame per-{unit_label.lower()} error"))
        sections.append(f'{sec_div("sec-appendix-b")}<h2>Appendix {al}: Per-Frame Per-{unit_label} Error</h2>')
        sections.append(
            f'<details class="appendix"><summary>Per-frame per-{unit_label.lower()} error heatmaps '
            f'(click to expand)</summary><div class="appendix-content">'
        )
        for cam_id, edata in sorted(error_data.items()):
            if "frame_tag_errors" in edata and edata["frame_tag_errors"]:
                n_frames = len(all_frames.get(cam_id, []))
                fig = plot_frame_tag_heatmap(
                    edata["frame_tag_errors"], n_frames, num_tags, cam_id)
                sections.append(f'<h3>cam{cam_id}</h3>')
                sections.append(f'<div class="plot-container"><img src="data:image/png;base64,{fig_b64(fig)}"></div>')
        sections.append("</div></details></div>")

    # ========== ASSEMBLE HTML ==========
    html = render_page(_cam_only_title(title, result, calibration_info), toc_entries, sections, style)

    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"HTML report saved to {output_path}")
