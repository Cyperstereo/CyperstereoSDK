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
"""Plotly 3D visualization of camera poses and board positions.

Generates interactive 3D plots that can be rotated/zoomed in the browser.

All coordinates are rotated into the "display frame" so the board stands
upright on the YZ plane (board normal = +X), matching the Rerun live viewer
(see ``rerun_viz.to_display_frame``). The display transform is a pure
permutation ``[x, y, z] -> [z, x, y]``; it changes no distances or angles.
"""

import base64

import numpy as np
import plotly.graph_objects as go

from mvis.detection.board import Board
from mvis.detection.corner_subsample import corner_mask


# Moving-glyph axis colors (x, y, z), matching the static pose triads below
_AXIS_RGB = ("#FF0000", "#00CC00", "#0000FF")

# Camera colors for multi-camera visualization
_CAM_COLORS = [
    "#2196F3",  # blue
    "#F44336",  # red
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#795548",  # brown
    "#607D8B",  # gray
]


# Display transform: board XY plane -> YZ plane (x_disp, y_disp, z_disp) =
# (z_orig, x_orig, y_orig); one definition in analysis.display_frame.
from mvis.analysis.display_frame import (  # noqa: E402,F401  (_to_display_points: tests)
    T_DISP as _T_DISP, cube_bounds, to_display_points as _to_display_points,
    to_display_pose as _to_display_pose)


def _make_frustum(position: np.ndarray, R: np.ndarray, scale: float = 0.03):
    """Create frustum lines for a camera.

    Args:
        position: Camera position [x, y, z].
        R: 3x3 rotation matrix (columns = camera axes in world frame).
        scale: Size of the frustum.

    Returns:
        List of (xs, ys, zs) line segments.
    """
    # Image plane corners in camera frame (normalized)
    corners_cam = np.array([
        [-0.6, -0.4, 1.0],
        [0.6, -0.4, 1.0],
        [0.6, 0.4, 1.0],
        [-0.6, 0.4, 1.0],
    ]) * scale

    # Transform to world frame
    corners_world = (R @ corners_cam.T).T + position

    lines = []
    # Lines from center to corners
    for c in corners_world:
        lines.append(([position[0], c[0]], [position[1], c[1]], [position[2], c[2]]))
    # Image plane rectangle
    for i in range(4):
        c1 = corners_world[i]
        c2 = corners_world[(i + 1) % 4]
        lines.append(([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]]))

    return lines


# ---------------------------------------------------------------------------
# Calibration progress (play/scrub along the trajectory in the HTML reports)
# ---------------------------------------------------------------------------

# Moving pose glyph, in camera units (x right, y down, z = optical axis): the
# image plane spans x in +-_GLYPH_HW, y in +-hh (hh from the image aspect), with
# an "up" tick above the top edge. _PROGRESS_JS in report_shell.py mirrors
# _glyph_xyz / _image_grid for the scrub/play updates.
_GLYPH_HW = 0.6
_GLYPH_HH = 0.4                       # 3:2 when there are no images


def _glyph_corners(hh):
    hw = _GLYPH_HW
    return (np.array([[-hw, -hh, 1.0], [hw, -hh, 1.0], [hw, hh, 1.0], [-hw, hh, 1.0]]),
            np.array([[-0.2, -hh - 0.05, 1.0], [0.0, -hh - 0.22, 1.0],
                      [0.2, -hh - 0.05, 1.0]]))


def _cam_pose(p, R, cam):
    """Camera pose from the body pose (p, R) and body->camera `cam` = (R_BC, t_BC)."""
    if cam is None:
        return p, R
    R_bc, t_bc = cam
    return p + R @ t_bc, R @ R_bc


def _glyph_xyz(p, R, fr, ax, hh=_GLYPH_HH, cam=None):
    """Frustum polyline (None-separated) and image-plane quad at the camera
    pose, and 3 axis segments at the body pose."""
    pc, Rc = _cam_pose(p, R, cam)
    corners, up = _glyph_corners(hh)
    c = pc + (Rc @ (corners * fr).T).T
    u = pc + (Rc @ (up * fr).T).T
    gap = [None] * 3
    frustum = []
    for k in range(4):
        frustum += [pc, c[k], gap]
    frustum += [c[0], c[1], c[2], c[3], c[0], gap, u[0], u[1], u[2]]
    axes = [[p, p + R[:, a] * ax] for a in range(3)]
    return frustum, c, axes


def _image_grid(p, R, fr, hh, iw, ih):
    """(3, ih, iw) plot coordinates of the image-plane pixel grid (row 0 = top)."""
    x = np.linspace(-_GLYPH_HW, _GLYPH_HW, iw)
    y = np.linspace(-hh, hh, ih)
    gx, gy = np.meshgrid(x, y)
    cam = np.stack([gx, gy, np.ones_like(gx)]) * fr
    return p[:, None, None] + np.einsum("ij,jhw->ihw", R, cam)


def _ray_end(p, R, plane):
    """Where the optical axis (camera z) meets the board plane n.x = d, or p."""
    if plane is None:
        return p
    n, d = np.asarray(plane[:3], float), float(plane[3])
    z = R[:, 2]
    den = float(n @ z)
    if abs(den) < 1e-9 or (d - n @ p) / den <= 0:
        return p
    return p + (d - n @ p) / den * z


_COVIS_COLOR = "#9333EA"              # board corner seen by >= 2 cameras now
_MAX_FRUSTUM_IMAGES = 200             # per camera; the page shows the nearest


def add_progress(fig, xyz, t=None, color="#2563EB", unit="pose",
                 R=None, frustum=0.0, axis=0.0, images=None,
                 colorscale=None, plane=None, panel=None, cam=None,
                 views=None, corners=None):
    """Append a "covered" path + "current" marker for the report's progress bar.

    `xyz` (N,3) are the trajectory points in plot coordinates, in time order;
    `t` (N,) their timestamps (s) or None to count `unit`s instead. With the
    rotations `R` (N,3,3; columns = body axes in plot coordinates) a pose glyph
    rides along too: x/y/z body axes of length `axis` and, if `frustum` > 0, a
    camera frustum of that depth. `colorscale` colours the covered path by time
    (with a colour bar); `plane` (nx, ny, nz, d: the board plane n.x = d) adds
    a dashed optical-axis ray to where the camera looks.

    Cameras: `views` is a list of dict(name, color, cam, images, panel, ids) —
    `cam` = (R_BC, t_BC) body -> camera (None = the body is the camera),
    `images` (N,h,w uint8) painted on the frustum, `panel`
    (pose_images.pose_panel) for the side panel, `ids` per-pose detected board
    point ids. The frustum follows the selected view (a selector appears for
    2+ views). The single-camera arguments `cam` / `images` / `panel` are the
    one-view shorthand. `corners` = (trace index, base colours, base sizes) of
    the board-corner scatter enables the co-visibility colouring (2+ views
    with ids). Everything starts at the end of the run, so the static figure
    still shows the whole path; `report_shell.plotly_to_html` renders the
    strip that moves it.
    """
    xyz = np.asarray(xyz, float)
    if len(xyz) < 2:
        return fig
    if t is not None:
        t = np.asarray(t, float)
        t = (t - t[0]).round(4).tolist() if len(t) == len(xyz) else None
    if colorscale:
        # colour array stays full length while the JS slices x/y/z, so each
        # point keeps its own time colour as the covered path grows
        c = t if t is not None else list(range(len(xyz)))
        line = dict(color=c, colorscale=colorscale, cmin=0, cmax=c[-1], width=6,
                    showscale=True, colorbar=dict(
                        title=dict(text="t (s)" if t is not None else unit,
                                   side="top"),
                        thickness=10, len=0.55, x=0.97, y=0.45, outlinewidth=0))
    else:
        line = dict(color=color, width=6)
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines",
        line=line, name="covered", showlegend=False, hoverinfo="skip"))
    done = len(fig.data) - 1
    fig.add_trace(go.Scatter3d(
        x=xyz[-1:, 0], y=xyz[-1:, 1], z=xyz[-1:, 2], mode="markers",
        marker=dict(size=7, color="#F59E0B", line=dict(color="#111827", width=1)),
        name="current"))
    spec = dict(done=done, cur=len(fig.data) - 1, xyz=xyz.round(5).tolist(),
                unit=unit, t=t)
    if views is None and (cam is not None or images is not None or panel is not None):
        views = [dict(name="cam0", color=color, cam=cam, images=images, panel=panel)]
    views = [_view_spec(v, len(xyz), frustum) for v in (views or [])]
    if R is not None and len(R) == len(xyz):
        R = np.asarray(R, float)
        v0 = views[0] if views else dict(cam=None, images=None, hh=_GLYPH_HH)
        vcam, images, hh = v0["cam"], v0["images"], v0["hh"]
        fr_pts, quad, axes = _glyph_xyz(xyz[-1], R[-1], frustum, axis, hh, vcam)
        pc, Rc = _cam_pose(xyz[-1], R[-1], vcam)
        gcolor = v0.get("color", color)
        glyph = {}
        if any(v["images"] is not None for v in views):
            img = images if images is not None else np.zeros((1, 2, 2), np.uint8)
            ih, iw = img.shape[1:]
            g = _image_grid(pc, Rc, frustum, hh, iw, ih)
            fig.add_trace(go.Surface(
                x=g[0], y=g[1], z=g[2], surfacecolor=img[-1],
                colorscale="gray", cmin=0, cmax=255, showscale=False,
                visible=images is not None,
                lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                              roughness=1.0, fresnel=0.0),
                name="current image", hoverinfo="skip", showlegend=False))
            glyph["image"] = len(fig.data) - 1
        if frustum > 0:
            fx, fy, fz = zip(*[(None,) * 3 if q[0] is None else q for q in fr_pts])
            fig.add_trace(go.Scatter3d(
                x=fx, y=fy, z=fz, mode="lines", line=dict(color=gcolor, width=5),
                name="current frustum", showlegend=False, hoverinfo="skip"))
            glyph["frustum"] = len(fig.data) - 1
        if plane is not None and frustum > 0:
            ray = np.array([pc, _ray_end(pc, Rc, plane)])
            fig.add_trace(go.Scatter3d(
                x=ray[:, 0], y=ray[:, 1], z=ray[:, 2], mode="lines",
                line=dict(color=gcolor, width=3, dash="dash"),
                name="current ray", showlegend=False, hoverinfo="skip"))
            glyph["ray"] = len(fig.data) - 1
            spec["plane"] = [float(v) for v in plane]
        if frustum > 0 and "image" not in glyph:
            fig.add_trace(go.Mesh3d(
                x=quad[:, 0], y=quad[:, 1], z=quad[:, 2], i=[0, 0], j=[1, 2],
                k=[2, 3], color=gcolor, opacity=0.25, name="current image plane",
                showlegend=False, hoverinfo="skip"))
            glyph["plane"] = len(fig.data) - 1
        for a, (seg, c) in enumerate(zip(axes, _AXIS_RGB)):
            seg = np.array(seg)
            fig.add_trace(go.Scatter3d(
                x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
                line=dict(color=c, width=7), name="current " + "xyz"[a],
                showlegend=False, hoverinfo="skip"))
            glyph["xyz"[a]] = len(fig.data) - 1
        spec.update(R=R.reshape(len(R), 9).round(5).tolist(), fr=frustum,
                    ax=axis, glyph=glyph)
    if views:
        for v in views:
            imgs = v.pop("images")
            if imgs is not None:
                # evenly spaced subset keeps long runs light (~3 kB per image)
                keep = np.unique(np.linspace(0, len(imgs) - 1, min(
                    len(imgs), _MAX_FRUSTUM_IMAGES)).round().astype(int))
                v.update(img=base64.b64encode(np.ascontiguousarray(
                    imgs[keep], np.uint8).tobytes()).decode("ascii"),
                    img_idx=keep.tolist(),
                    iw=int(imgs.shape[2]), ih=int(imgs.shape[1]))
            if v["cam"] is not None:
                v["cam"] = (v["cam"][0].reshape(9).round(6).tolist()
                            + v["cam"][1].round(6).tolist())
            if v.get("panel"):
                v["panel"] = _pack_uv(v["panel"])
        spec["views"] = views
        if corners is not None and sum(v.get("ids") is not None for v in views) >= 2:
            idx, base_c, base_s = corners
            spec["covis"] = dict(trace=int(idx), color=list(base_c),
                                 size=[float(x) for x in base_s], both=_COVIS_COLOR)
            c, sz = _covis_style(views, len(xyz) - 1, list(base_c), list(base_s))
            fig.data[idx].marker.color = c
            fig.data[idx].marker.size = sz
    fig.update_layout(meta=dict(mvis_progress=spec))
    return fig


def _pack_uv(panel):
    """Panel corner pixels as base64 int16 (uvb) + per-step offsets (uvo):
    about half the bytes of the JSON lists for long multi-camera runs."""
    flat = [np.asarray(u, np.int16) for u in panel["uv"]]
    off = np.concatenate([[0], np.cumsum([len(u) for u in flat])]).astype(int)
    p = {k: v for k, v in panel.items() if k != "uv"}
    p["uvb"] = base64.b64encode(np.concatenate(flat or [np.zeros(0, np.int16)])
                                .astype("<i2").tobytes()).decode("ascii")
    p["uvo"] = off.tolist()
    return p


def _view_spec(v, n, frustum):
    """Normalise one camera view for add_progress (drops unusable parts)."""
    cam = v.get("cam")
    if cam is not None:
        cam = (np.asarray(cam[0], float), np.asarray(cam[1], float))
    images = v.get("images")
    if images is not None and (frustum <= 0 or len(images) != n):
        images = None
    panel = v.get("panel")
    if panel is not None and len(panel.get("uv", ())) != n:
        panel = None
    ids = v.get("ids")
    if ids is not None and len(ids) != n:
        ids = None
    hh = (_GLYPH_HW * images.shape[1] / images.shape[2]
          if images is not None else _GLYPH_HH)
    return dict(name=v.get("name", "cam"), color=v.get("color", "#2563EB"),
                cam=cam, images=images, panel=panel, hh=hh,
                ids=[list(map(int, x)) for x in ids] if ids is not None else None)


def _covis_style(views, i, base_c, base_s):
    """Board-corner colours/sizes at pose i: one camera -> its colour,
    2+ cameras -> _COVIS_COLOR (mirrors covis() in report_shell._PROGRESS_JS)."""
    seen = {}
    for k, v in enumerate(views):
        if v.get("ids") is None:
            continue
        for pid in v["ids"][i]:
            seen.setdefault(pid, []).append(k)
    c, s = list(base_c), list(base_s)
    for pid, ks in seen.items():
        if 0 <= pid < len(c):
            c[pid] = _COVIS_COLOR if len(ks) > 1 else views[ks[0]]["color"]
            s[pid] = 6.0 if len(ks) > 1 else 4.5
    return c, s


def progress_spec(fig):
    """The progress spec `add_progress` stored on `fig`, or None."""
    meta = fig.layout.meta
    return meta.get("mvis_progress") if isinstance(meta, dict) else None


def plot_camera_poses(
    board_poses: list[np.ndarray],
    board: Board,
    extrinsics: list | None = None,
    camera_ids: list[int] | None = None,
    title: str = "Camera Poses & Board",
    times: list[float] | None = None,
    images: np.ndarray | None = None,
    corner_subsample: str = "all",
    panel: dict | None = None,
    views: list | None = None,
) -> go.Figure:
    """Create interactive 3D plot of camera poses and board points.

    Args:
        board_poses: List of 4x4 SE(3) matrices (T_board_cam in GTSAM convention,
                     i.e., camera position in board frame).
        board: Board object for 3D point positions.
        extrinsics: List of (cam_id, 4x4 T_cam_ref) for multi-camera.
                    If provided, board_poses are for the reference camera,
                    and other cameras are shown using their extrinsics.
        camera_ids: Camera IDs for labeling.
        title: Plot title.
        times: Timestamps (s) parallel to board_poses; label the progress
               bar in time instead of pose count.
        images: (N,h,w) uint8 per-pose camera images (pose_images.py) shown
                on the moving frustum's image plane.
        corner_subsample: solver corner subsampling (e.g. "top_left"); kept
                corners are drawn solid, dropped ones faint.
        panel: per-pose camera views + detected corners (pose_images.pose_panel)
               for the side panel that follows the progress bar.
        views: multi-camera form of `images` / `panel`: one dict per camera
               (ref first) with camera_id, images, panel and ids (per-pose
               detected point ids). Adds the camera selector, one side-panel
               card per camera and the co-visibility colouring of the board.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    ref_poses_disp = [_to_display_pose(T) for T in board_poses]
    ref_xyz = np.array([T[:3, 3] for T in ref_poses_disp])
    corner_trace = len(fig.data)            # _add_board's first trace = corners
    board_plane = _add_board(fig, board, ref_xyz, corner_subsample)
    side = float(np.sign(np.median(ref_xyz[:, 0]))) if len(ref_xyz) else -1.0
    side = side or -1.0

    _add_camera_poses(fig, ref_poses_disp, cam_id=0, color=_CAM_COLORS[0],
                      name="Camera 0 (ref)")

    # Additional cameras via extrinsics
    if extrinsics:
        for i, (cam_id, T_cam_ref) in enumerate(extrinsics):
            T_cam_ref_cam_i = np.linalg.inv(T_cam_ref)
            cam_poses_disp = []
            for T_board_cam_ref in board_poses:
                T_board_cam_i = T_board_cam_ref @ T_cam_ref_cam_i
                cam_poses_disp.append(_to_display_pose(T_board_cam_i))

            color = _CAM_COLORS[(i + 1) % len(_CAM_COLORS)]
            _add_camera_poses(fig, cam_poses_disp, cam_id=cam_id, color=color,
                              name=f"Camera {cam_id}")

    # moving glyph sized to the trajectory so the frustum image stays readable
    extent = float(np.ptp(ref_xyz, axis=0).max()) if len(ref_xyz) else 0.0
    fr = float(np.clip((0.15 if images is not None else 0.08) * extent, 0.045, 0.25))
    ext_by_id = {cid: T for cid, T in (extrinsics or [])}
    pview = None
    if views:
        pview = []
        for k, v in enumerate(views):
            cid = v.get("camera_id", k)
            T_off = (np.linalg.inv(np.asarray(ext_by_id[cid], float))
                     if cid in ext_by_id else None)      # ref camera -> camera cid
            pview.append(dict(
                name=f"cam{cid}", color=_CAM_COLORS[k % len(_CAM_COLORS)],
                cam=None if T_off is None else (T_off[:3, :3], T_off[:3, 3]),
                images=v.get("images"), panel=v.get("panel"), ids=v.get("ids")))
    cm = fig.data[corner_trace].marker
    add_progress(fig, ref_xyz, t=times, color=_CAM_COLORS[0],
                 R=[T[:3, :3] for T in ref_poses_disp], frustum=fr,
                 axis=0.8 * fr, images=images, colorscale=_TIME_COLORSCALE,
                 plane=board_plane, panel=panel, views=pview,
                 corners=(corner_trace, list(cm.color), list(cm.size)))

    axis = dict(showbackground=True, backgroundcolor="#F8FAFC",
                gridcolor="#E2E8F0", zerolinecolor="#CBD5E1",
                title_font=dict(size=12, color="#475569"),
                tickfont=dict(size=10, color="#64748B"))
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15, color="#1E293B")),
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif"),
        scene=dict(
            # Axis names reflect display-frame orientation:
            #   display x = depth (board normal points toward -x)
            #   display y = board width
            #   display z = board height
            xaxis=dict(axis, title="depth (m)"),
            yaxis=dict(axis, title="board width (m)"),
            zaxis=dict(axis, title="board height (m)"),
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6 * side, y=-1.1, z=0.7),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
            ),
        ),
        legend=dict(orientation="h", x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.7)",
                    font=dict(size=11), itemsizing="constant"),
        height=820,
        margin=dict(l=0, r=24, t=44, b=0),
    )

    return fig


def _add_camera_poses(
    fig: go.Figure,
    poses: list[np.ndarray],
    cam_id: int,
    color: str,
    name: str,
    frustum_scale: float = 0.03,
):
    """Add camera pose traces to a plotly figure.

    Args:
        fig: Plotly figure to add to.
        poses: List of 4x4 T_board_cam matrices.
        cam_id: Camera ID.
        color: Color for this camera.
        name: Legend name.
        frustum_scale: Size of camera frustums.
    """
    positions = np.array([T[:3, 3] for T in poses])

    # Camera positions + path: faint context; the progress overlay (ref
    # camera) draws the time-coloured path and the moving pose on top
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="markers",
        marker=dict(size=2, color=color, opacity=0.45),
        name=name,
    ))
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
        mode="lines",
        line=dict(color="#94A3B8", width=1.5),
        showlegend=False,
        opacity=0.5,
        hoverinfo="skip",
    ))

    # Every N-th frustum (~15) as ONE None-separated trace (was ~7 traces each)
    step = max(1, len(poses) // 15)
    xs, ys, zs = [], [], []
    for i in range(0, len(poses), step):
        T = poses[i]
        for lx, ly, lz in _make_frustum(T[:3, 3], T[:3, :3], frustum_scale):
            xs += list(lx) + [None]
            ys += list(ly) + [None]
            zs += list(lz) + [None]
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="lines",
        line=dict(color=color, width=1.5),
        opacity=0.35, name=f"{name} frustums", showlegend=False,
        hoverinfo="skip",
    ))


# Covered-path time colours (light -> deep blue, readable on the light scene)
_TIME_COLORSCALE = [[0.0, "#BAE6FD"], [0.5, "#3B82F6"], [1.0, "#1E3A8A"]]


def _add_board(fig, board, cam_xyz, corner_subsample="all", T=None):
    """Board as a light plate with dark tags (AprilGrid), its outline and the
    corners (solver-kept solid, dropped faint); returns the board plane
    (nx, ny, nz, d) with n.x = d in plot coordinates.

    `T` (4x4) maps board-frame points to plot coordinates (default: the
    cam-only display permutation). The first trace added is the corner scatter
    — callers and tests rely on it being fig.data[0] in the cam-only plot. The
    plate sits a hair behind the tags as seen from the cameras (`cam_xyz`).
    """
    T = _T_DISP if T is None else np.asarray(T, float)
    to_plot = lambda q: (T[:3, :3] @ np.asarray(q, float).T).T + T[:3, 3]
    n, o = T[:3, 2] / np.linalg.norm(T[:3, 2]), T[:3, 3]
    cam_xyz = np.asarray(cam_xyz, float).reshape(-1, 3)
    side = float(np.sign(np.median((cam_xyz - o) @ n))) if len(cam_xyz) else -1.0
    side = side or -1.0

    orig = np.asarray(board.all_points_3d(), float)
    pts = to_plot(orig)
    has_tags = getattr(board, "_num_tags", 0) > 0 and len(orig) % 4 == 0
    kept = _corner_kept_mask(len(orig), corner_subsample, has_tags)
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=np.where(kept, 3.0, 1.5),
                    color=np.where(kept, "#0F172A", "#94A3B8")),
        name=("Board corners" if kept.all()
              else f"Board corners ({int(kept.sum())} used / {len(kept)})"),
        hoverinfo="skip",
    ))

    lo, hi = orig[:, :2].min(0), orig[:, :2].max(0)
    m = 0.06 * float((hi - lo).max())               # plate margin
    plate = np.array([[lo[0] - m, lo[1] - m, 0], [hi[0] + m, lo[1] - m, 0],
                      [hi[0] + m, hi[1] + m, 0], [lo[0] - m, hi[1] + m, 0]])
    pd = to_plot(plate) - side * 0.002 * n          # behind the tags
    fig.add_trace(go.Mesh3d(
        x=pd[:, 0], y=pd[:, 1], z=pd[:, 2], i=[0, 0], j=[1, 2], k=[2, 3],
        color="#F1F5F9", opacity=1.0, flatshading=True, name="board",
        hoverinfo="skip", showlegend=False))
    ring = np.vstack([pd, pd[:1]])
    fig.add_trace(go.Scatter3d(
        x=ring[:, 0], y=ring[:, 1], z=ring[:, 2], mode="lines",
        line=dict(color="#334155", width=3), name="Board Outline",
        showlegend=False, hoverinfo="skip"))

    if has_tags:
        # tag t = corners 4t..4t+3 in ring order (bl, br, tr, tl)
        q = np.arange(len(pts) // 4) * 4
        tp = pts + side * 0.001 * n
        fig.add_trace(go.Mesh3d(
            x=tp[:, 0], y=tp[:, 1], z=tp[:, 2],
            i=np.concatenate([q, q]), j=np.concatenate([q + 1, q + 2]),
            k=np.concatenate([q + 2, q + 3]),
            color="#1E293B", opacity=1.0, flatshading=True, name="tags",
            hoverinfo="skip", showlegend=False))
    return (float(n[0]), float(n[1]), float(n[2]), float(n @ o))


# ---------------------------------------------------------------------------
# Matplotlib static figures (for HTML report)
# ---------------------------------------------------------------------------

def plot_stereo_extrinsics_3d_interactive(
    extrinsics: list,
    reference_camera: int,
    frustum_scale: float = 0.03,
):
    """Plotly interactive 3D stereo camera visualization with frustums.

    Each camera rendered as a wireframe pyramid (frustum) with coloured
    XYZ axis arrows. Baseline dashed line between cameras. Fully
    rotatable / zoomable in the browser.

    Coordinates are in the **reference camera frame** (no display
    transform) so axes have their natural meaning:
    X=red(right), Y=green(down), Z=blue(forward/optical axis).

    Returns:
        plotly ``go.Figure``.
    """
    fig = go.Figure()

    # Ref camera at origin in its own frame — no display transform.
    cam_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cam_data[reference_camera] = (np.zeros(3), np.eye(3))

    for ext in extrinsics:
        T = np.asarray(ext.T_to_from, dtype=np.float64)
        T_cam_in_ref = np.linalg.inv(T)
        cam_data[ext.from_camera] = (T_cam_in_ref[:3, 3], T_cam_in_ref[:3, :3])

    axis_colors = ["red", "green", "blue"]
    axis_names = ["X (right)", "Y (down)", "Z (forward)"]

    for cid, (pos, R) in cam_data.items():
        is_ref = (cid == reference_camera)
        color = _CAM_COLORS[cid % len(_CAM_COLORS)]
        label = f"cam{cid}" + (" (ref)" if is_ref else "")

        # Frustum wireframe.
        frustum_lines = _make_frustum(pos, R, scale=frustum_scale)
        for i, (xs, ys, zs) in enumerate(frustum_lines):
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=3),
                name=label if i == 0 else None,
                showlegend=(i == 0),
                legendgroup=f"cam{cid}",
                hoverinfo="skip",
            ))

        # Camera center marker.
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode="markers+text",
            marker=dict(size=5, color=color, symbol="diamond"),
            text=[label],
            textposition="top center",
            textfont=dict(size=10, color=color),
            showlegend=False,
            legendgroup=f"cam{cid}",
        ))

        # Short axis arrows (X=red, Y=green, Z=blue).
        arrow_len = frustum_scale * 0.6
        for ax_idx in range(3):
            end = pos + R[:, ax_idx] * arrow_len
            fig.add_trace(go.Scatter3d(
                x=[pos[0], end[0]], y=[pos[1], end[1]], z=[pos[2], end[2]],
                mode="lines",
                line=dict(color=axis_colors[ax_idx], width=4),
                name=f"{axis_names[ax_idx]}" if (cid == reference_camera and ax_idx == 0) else None,
                showlegend=False,
                legendgroup=f"cam{cid}",
                hovertext=f"cam{cid} {axis_names[ax_idx]}",
                hoverinfo="text",
            ))

    # Baseline dashed line between cameras.
    ref_pos = cam_data[reference_camera][0]
    for cid, (pos, _) in cam_data.items():
        if cid == reference_camera:
            continue
        baseline_mm = float(np.linalg.norm(pos - ref_pos)) * 1000.0
        mid = 0.5 * (ref_pos + pos)
        fig.add_trace(go.Scatter3d(
            x=[ref_pos[0], pos[0]], y=[ref_pos[1], pos[1]], z=[ref_pos[2], pos[2]],
            mode="lines",
            line=dict(color="black", width=2, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter3d(
            x=[mid[0]], y=[mid[1]], z=[mid[2]],
            mode="text",
            text=[f"{baseline_mm:.1f}mm"],
            textfont=dict(size=12, color="black"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title="Stereo Camera Poses (interactive — ref camera frame)",
        scene=dict(
            xaxis_title="X (right)",
            yaxis_title="Y (down)",
            zaxis_title="Z (forward)",
            aspectmode="data",
            # Default view: baseline horizontal, Z (forward) on left,
            # X (right) at bottom, Y (down) going back-right.
            camera=dict(
                eye=dict(x=-0.8, y=-0.5, z=-1.5),
                up=dict(x=0, y=-1, z=0),  # Y-down = "up" in screen
            ),
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, t=40, b=0),
        height=500,
    )
    fig.add_annotation(
        text="Camera frame: X=red(right) Y=green(down) Z=blue(forward/optical)",
        xref="paper", yref="paper", x=0.01, y=0.01,
        showarrow=False, font=dict(size=10, color="grey"),
    )
    return fig


def _draw_camera_frustum(ax, position, R, scale=0.03,
                          label="", color="#2196F3", alpha=0.7):
    """Draw a camera frustum (pyramid + axis arrows) in matplotlib 3D.

    The frustum is a 4-sided pyramid from the camera center to an
    image-plane rectangle, plus short axis arrows (X=red right, Y=green
    down, Z=blue forward).

    Args:
        ax: matplotlib 3D axis.
        position: camera center [x, y, z] in display frame.
        R: 3×3 rotation (columns = camera axes in display frame).
        scale: frustum depth in metres.
        label: text label for the legend.
        color: frustum wireframe colour.
        alpha: line opacity.
    """
    # Image-plane corners in camera frame (3:2 aspect).
    corners_cam = np.array([
        [-0.6, -0.4, 1.0],
        [ 0.6, -0.4, 1.0],
        [ 0.6,  0.4, 1.0],
        [-0.6,  0.4, 1.0],
    ]) * scale

    # Transform to display frame.
    corners_world = (R @ corners_cam.T).T + position

    # Pyramid edges: center → each corner.
    for c in corners_world:
        ax.plot([position[0], c[0]], [position[1], c[1]],
                 [position[2], c[2]],
                 color=color, linewidth=0.8, alpha=alpha)
    # Image-plane rectangle.
    for i in range(4):
        c1, c2 = corners_world[i], corners_world[(i + 1) % 4]
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], [c1[2], c2[2]],
                 color=color, linewidth=1.5, alpha=alpha)

    # Short axis arrows at the center.
    axis_len = scale * 0.6
    for ax_idx, ax_color in enumerate(["red", "green", "blue"]):
        end = position + R[:, ax_idx] * axis_len
        ax.plot([position[0], end[0]], [position[1], end[1]],
                 [position[2], end[2]],
                 color=ax_color, linewidth=2.5, alpha=0.9)

    # Camera center dot + label.
    ax.scatter([position[0]], [position[1]], [position[2]],
                c=color, s=40, zorder=5, edgecolors="black", linewidths=0.5)
    ax.text(position[0], position[1], position[2],
             f"  {label}", fontsize=8, color=color)


def plot_stereo_extrinsics_3d(
    extrinsics: list,
    reference_camera: int,
    axis_scale_m: float = 0.03,
):
    """Matplotlib 3D + top-down YZ inset of camera poses as frustums.

    Each camera is drawn as a pyramid (image-plane rectangle + 4 edges
    to the center) with short axis arrows (X=red, Y=green, Z=blue).

    Coordinates follow the board-display convention from ``_T_DISP``.

    Returns:
        matplotlib Figure (1 3D axis + 1 2D inset).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    # Show cameras in the reference camera frame directly (no display
    # transform) so X=right, Y=down, Z=forward have their natural meaning.
    cam_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cam_data[reference_camera] = (np.zeros(3), np.eye(3))

    for ext in extrinsics:
        T = np.asarray(ext.T_to_from, dtype=np.float64)
        T_cam_in_ref = np.linalg.inv(T)
        cam_data[ext.from_camera] = (T_cam_in_ref[:3, 3], T_cam_in_ref[:3, :3])

    # Draw frustums.
    for cid, (pos, R) in cam_data.items():
        is_ref = (cid == reference_camera)
        color = _CAM_COLORS[cid % len(_CAM_COLORS)]
        label = f"cam{cid}" + (" (ref)" if is_ref else "")
        _draw_camera_frustum(ax3d, pos, R, scale=axis_scale_m,
                              label=label, color=color)

    # Baseline dashed lines + labels.
    ref_pos = cam_data[reference_camera][0]
    for cid, (pos, _) in cam_data.items():
        if cid == reference_camera:
            continue
        ax3d.plot([ref_pos[0], pos[0]], [ref_pos[1], pos[1]],
                   [ref_pos[2], pos[2]],
                   linestyle="--", color="black", linewidth=1)
        baseline_mm = float(np.linalg.norm(pos - ref_pos)) * 1000.0
        mid = 0.5 * (ref_pos + pos)
        ax3d.text(mid[0], mid[1], mid[2], f"{baseline_mm:.1f}mm",
                   fontsize=9, fontweight="bold")

    ax3d.set_xlabel("X (right)")
    ax3d.set_ylabel("Y (down)")
    ax3d.set_zlabel("Z (forward)")
    ax3d.set_title("Stereo camera poses (ref camera frame)")
    ax3d.text2D(0.02, 0.02,
                 "Camera frame: X=red(right) Y=green(down) Z=blue(forward)",
                 transform=ax3d.transAxes, fontsize=7, color="grey")
    # Invert Y so "down" visually goes downward in the plot.
    ax3d.invert_yaxis()
    _cube_equal_aspect(ax3d, [p for p, _ in cam_data.values()], axis_scale_m)

    # XZ-plane top-down inset (bird's eye: looking down the Y axis).
    for cid, (pos, R) in cam_data.items():
        color = _CAM_COLORS[cid % len(_CAM_COLORS)]
        is_ref = (cid == reference_camera)
        ax2d.plot(pos[0], pos[2], "o", color=color, markersize=8,
                   markeredgecolor="black",
                   label=f"cam{cid}" + (" (ref)" if is_ref else ""))
        # Short forward arrow in XZ plane.
        fwd = R[:, 2] * axis_scale_m * 0.8
        ax2d.annotate("", xy=(pos[0] + fwd[0], pos[2] + fwd[2]),
                        xytext=(pos[0], pos[2]),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5))
    for cid, (pos, _) in cam_data.items():
        if cid == reference_camera:
            continue
        ax2d.plot([ref_pos[0], pos[0]], [ref_pos[2], pos[2]],
                   "k--", linewidth=1)
    ax2d.set_xlabel("X (right)")
    ax2d.set_ylabel("Z (forward)")
    ax2d.set_aspect("equal", adjustable="datalim")
    ax2d.set_title("Top-down (XZ plane, looking along Y-down)")
    ax2d.legend(loc="upper right", fontsize=8)
    ax2d.grid(True, alpha=0.3)
    ax2d.legend(loc="upper right", fontsize=8)
    ax2d.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig



def _cube_equal_aspect(ax, points_iter, min_range):
    """Set xlim/ylim/zlim to a cube around the points for equal aspect."""
    pts = np.asarray(list(points_iter))
    if pts.size == 0:
        return
    center = pts.mean(axis=0)
    half = max(np.max(np.ptp(pts, axis=0)) / 2.0, min_range * 3.0)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def _corner_kept_mask(n_points: int, mode: str, has_tags: bool) -> np.ndarray:
    """Return a boolean mask of shape (n_points,) indicating kept corners.

    For AprilGrid boards (has_tags=True), filter is based on
    ``point_id % 4`` (tag-intrinsic corner index), through the solver's
    `corner_mask` (an unknown mode raises). For chessboards or
    mode="all", all points are kept.
    """
    if mode == "all" or not has_tags:
        return np.ones(n_points, dtype=bool)
    return corner_mask(np.arange(n_points), mode)


# ===========================================================================
# Cam-IMU 3D plots
# ===========================================================================
# Two interactive plots for the joint cam-IMU report:
#   plot_cam_imu_scene  — the recovered world trajectory + board/landmarks
#   plot_cam_imu_rig    — the body-frame sensor layout (cameras + IMUs)
# Both reuse the display permutation (board upright) and _make_frustum above.

import gtsam  # noqa: E402  (lazy-friendly; plotly module already heavy)

_X = gtsam.symbol_shorthand.X
_B0 = gtsam.symbol('b', 0)
_Lm = lambda i: gtsam.symbol('l', i)          # feature-mode world landmark
_AXIS_COLORS = ("#E53935", "#43A047", "#1E88E5")   # x, y, z


def _scene_layout(fig, title, axis_labels=("z", "x", "y")):
    # default labels match the _T_DISP board-frame permutation; pass true
    # ("x","y","z") labels for unpermuted (gravity-aligned) scenes.
    fig.update_layout(
        title=title, height=560, margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(aspectmode="data",
                   xaxis_title=axis_labels[0], yaxis_title=axis_labels[1],
                   zaxis_title=axis_labels[2]),
        legend=dict(orientation="h", yanchor="bottom", y=0.0))


def _equal_aspect(fig, pts):
    """Force an equal-aspect cube around the given (N,3) display points."""
    pts = np.asarray(pts, float)
    if pts.size == 0:
        return
    c, r = cube_bounds(pts, 1e-3, pad=1.05)
    fig.update_layout(scene=dict(
        aspectmode="cube",
        xaxis=dict(range=[c[0] - r, c[0] + r]),
        yaxis=dict(range=[c[1] - r, c[1] + r]),
        zaxis=dict(range=[c[2] - r, c[2] + r])))


def _traj_positions(values):
    """World positions of the trajectory states X(0..K), as (N,3)."""
    pos = []
    k = 0
    while values.exists(_X(k)):
        pos.append(np.asarray(values.atPose3(_X(k)).translation(), float))
        k += 1
    return np.array(pos) if pos else np.zeros((0, 3))


def _traj_rotations(values, n):
    """World orientations (3x3) of the trajectory states X(0..n-1)."""
    return np.array([values.atPose3(_X(k)).rotation().matrix() for k in range(n)])


def _landmark_points(values, max_n=20000):
    """Feature-mode world landmarks _L(id) present in `values`, as (N,3)."""
    pts = []
    for key in values.keys():
        if gtsam.Symbol(key).chr() == ord('l'):
            try:
                pts.append(np.asarray(values.atPoint3(key), float))
            except Exception:
                continue
            if len(pts) >= max_n:
                break
    return np.array(pts) if pts else np.zeros((0, 3))


def plot_cam_imu_scene(values, board=None, T_BinW=None, landmarks=None,
                       title="Trajectory & Scene", t_states=None,
                       base_cam=None, views=None, state_idx=None) -> "go.Figure":
    """World-frame scene: the recovered IMU trajectory plus the calibration
    board (board mode) or the landmark cloud (feature mode).

    `values` is the converged GTSAM Values; `board` + `T_BinW` give board-point
    geometry in the world; `landmarks` (N,3) overrides auto-extraction;
    `t_states` (s) labels the progress bar in time instead of state count.
    `views` (cam_imu_report._cam_views) = one dict per camera: name, cam =
    (R, t) of T_CtoI, and per-step images / panel / ids — each camera rides
    the moving IMU triad rigidly (frustum + image + view ray, selector,
    side-panel cards, board co-visibility). `state_idx` thins the progress
    timeline to those states (views are per thinned step). `base_cam` =
    dict(T_CtoI, images, panel) is the one-camera shorthand.
    """
    fig = go.Figure()
    allpts = []

    # NOTE: no _T_DISP permutation here. That transform is for the CAMERA-ONLY
    # plots whose world frame IS the board frame (board on z=0, stood up like a
    # wall by the permutation). The cam-IMU world is GRAVITY-ALIGNED (g_inW =
    # [0,0,-g], z up), so raw coordinates already display naturally — with the
    # permutation the real vertical mapped to display-x and an upright board
    # rendered lying "on the ground".
    pos = _traj_positions(values)
    Rs = _traj_rotations(values, len(pos)) if len(pos) else None
    if views is None and base_cam is not None:
        T_ci = np.asarray(base_cam["T_CtoI"], float)
        views = [dict(name="cam0", cam=(T_ci[:3, :3], T_ci[:3, 3]),
                      images=base_cam.get("images"), panel=base_cam.get("panel"))]
    views = [dict(v, color=v.get("color", _CAM_COLORS[k % len(_CAM_COLORS)]))
             for k, v in enumerate(views or [])]
    if views and len(pos):
        t0 = np.asarray(views[0]["cam"][1], float)
        allpts.append(np.array([p + R @ t0 for p, R in zip(pos, Rs)]))

    # board in world (board mode) -> else landmark cloud (feature mode)
    plane = None
    Tb = None
    if board is not None:
        Tb = T_BinW if T_BinW is not None else (
            values.atPose3(_B0) if values.exists(_B0) else None)
    corner_trace = None
    if board is not None and Tb is not None:
        T = Tb.matrix() if hasattr(Tb, "matrix") else np.asarray(Tb, float)
        corner_trace = len(fig.data)
        plane = _add_board(fig, board, allpts[0] if allpts else pos, T=T)
        bp = np.asarray(board.all_points_3d(), float)
        allpts.append((T[:3, :3] @ bp.T).T + T[:3, 3])
    else:
        if board is not None:
            bpts = np.asarray(board.all_points_3d(), float)
        else:
            bpts = (np.asarray(landmarks, float) if landmarks is not None
                    else _landmark_points(values))
        if len(bpts):
            allpts.append(bpts)
            fig.add_trace(go.Scatter3d(
                x=bpts[:, 0], y=bpts[:, 1], z=bpts[:, 2], mode="markers",
                marker=dict(size=2, color="#444"),
                name="board" if board is not None else "landmarks"))

    if len(pos):
        allpts.append(pos)
        # faint full path for context; the progress overlay draws the
        # time-coloured covered part and the moving rig on top
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="lines",
            line=dict(color="#94A3B8", width=1.5), opacity=0.6,
            name="IMU trajectory", hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(
            x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]], mode="markers",
            marker=dict(size=5, color="#16A34A"), name="start"))
        extent = max(float(np.ptp(pos, axis=0).max()), 1e-3)
        has_img = any(v.get("images") is not None for v in views)
        fr = (float(np.clip((0.15 if has_img else 0.08) * extent, 0.03, 0.25))
              if views else 0.0)
        idx = (np.asarray(state_idx, int) if state_idx is not None
               else np.arange(len(pos)))
        ts = (np.asarray(t_states, float)[idx]
              if t_states is not None and len(t_states) == len(pos) else None)
        corners = None
        if corner_trace is not None:
            cm = fig.data[corner_trace].marker
            corners = (corner_trace, list(cm.color), list(cm.size))
        add_progress(fig, pos[idx], t=ts,
                     unit="step" if len(idx) < len(pos) else "state", R=Rs[idx],
                     axis=(0.8 * fr if views else max(0.25 * extent, 0.03)),
                     frustum=fr, colorscale=_TIME_COLORSCALE, plane=plane,
                     views=views or None, corners=corners)

    _styled_scene(fig, title, ("x (m)", "y (m)", "z up (m)"),
                  eye=dict(x=1.4, y=-1.4, z=0.8))
    if allpts:
        _equal_aspect(fig, np.concatenate(allpts))
    return fig


def _styled_scene(fig, title, axis_titles, eye):
    """Light report styling shared by the 3D trajectory plots."""
    axis = dict(showbackground=True, backgroundcolor="#F8FAFC",
                gridcolor="#E2E8F0", zerolinecolor="#CBD5E1",
                title_font=dict(size=12, color="#475569"),
                tickfont=dict(size=10, color="#64748B"))
    fig.update_layout(
        title=dict(text=title, x=0.02, font=dict(size=15, color="#1E293B")),
        paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, -apple-system, Segoe UI, sans-serif"),
        scene=dict(xaxis=dict(axis, title=axis_titles[0]),
                   yaxis=dict(axis, title=axis_titles[1]),
                   zaxis=dict(axis, title=axis_titles[2]),
                   aspectmode="data",
                   camera=dict(eye=eye, up=dict(x=0, y=0, z=1))),
        legend=dict(orientation="h", x=0.02, y=0.98,
                    bgcolor="rgba(255,255,255,0.7)", font=dict(size=11),
                    itemsizing="constant"),
        height=820,
        margin=dict(l=0, r=24, t=44, b=0),
    )


def plot_cam_imu_rig(cameras, imus=None, title="Sensor Rig (base-IMU frame)") -> "go.Figure":
    """Body-frame rig: base IMU triad at the origin, each camera as a frustum at
    its T_C->I, each aux IMU as a triad at its T_IstoI, lever arms drawn + labeled.
    """
    fig = go.Figure()
    allpts = [np.zeros((1, 3))]

    def triad(origin, R, scale, name):
        for i in range(3):
            tip = origin + R[:, i] * scale
            allpts.append(tip[None, :])          # keep tips inside the ranges
            fig.add_trace(go.Scatter3d(
                x=[origin[0], tip[0]], y=[origin[1], tip[1]], z=[origin[2], tip[2]],
                mode="lines", line=dict(color=_AXIS_COLORS[i], width=6),
                showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(
            x=[origin[0]], y=[origin[1]], z=[origin[2]], mode="markers+text",
            marker=dict(size=3, color="#222"), text=[name], textposition="top center",
            showlegend=False, hoverinfo="skip"))

    triad(np.zeros(3), np.eye(3), 0.04, "imu0")

    for c, e in enumerate(cameras or []):
        cal = e["calib"]
        p = np.asarray(cal.T.translation(), float)
        R = np.asarray(cal.T.rotation().matrix(), float)
        col = _CAM_COLORS[c % len(_CAM_COLORS)]
        for xs, ys, zs in _make_frustum(p, R, scale=0.03):
            allpts.append(np.column_stack([xs, ys, zs]))   # keep corners inside
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                       line=dict(color=col, width=3),
                                       showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter3d(
            x=[0, p[0]], y=[0, p[1]], z=[0, p[2]], mode="lines",
            line=dict(color=col, width=2, dash="dot"),
            name=f"cam{c}  |{1e3 * np.linalg.norm(p):.0f} mm|"))
        allpts.append(p[None, :])

    for m in (imus or []):
        if "T_IstoI" in m:
            T = m["T_IstoI"]
            p = np.asarray(T.translation(), float)
            R = np.asarray(T.rotation().matrix(), float)
            triad(p, R, 0.04, m.get("name", "aux"))
            fig.add_trace(go.Scatter3d(
                x=[0, p[0]], y=[0, p[1]], z=[0, p[2]], mode="lines",
                line=dict(color="#888", width=2, dash="dot"),
                name=f"{m.get('name', 'aux')}  |{1e3 * np.linalg.norm(p):.0f} mm|"))
            allpts.append(p[None, :])

    fig.update_layout(title=title, height=560, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                      legend=dict(orientation="h", yanchor="bottom", y=0.0))
    _equal_aspect(fig, np.concatenate(allpts))
    return fig
