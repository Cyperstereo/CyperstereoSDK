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
"""Visualize a CalibScenario — the rig trajectory + the placed board in 3D, and
a sample camera view (the projected board in the image).

Headless (Agg): pass `save=<path>` to render a PNG.
"""

from __future__ import annotations

import numpy as np
import gtsam


def _board_face(board, T_BinW) -> np.ndarray:
    """The 4 outer corners of the (planar) board, in the world frame."""
    pB = np.asarray(board.all_points_3d(), float)
    xs, ys = pB[:, 0], pB[:, 1]
    corners_B = [[xs.min(), ys.min(), 0.0], [xs.max(), ys.min(), 0.0],
                 [xs.max(), ys.max(), 0.0], [xs.min(), ys.max(), 0.0]]
    return np.array([np.asarray(T_BinW.transformFrom(gtsam.Point3(*c)), float)
                     for c in corners_B])


def _draw_frustum(ax, pos, R, scale=0.12, color="0.4", lw=0.6):
    """Small camera frustum (optical axis = +z_cam) at a pose."""
    x, y, z = R[:, 0], R[:, 1], R[:, 2]
    d = 2.0 * scale
    corners = [pos + d * z + sx * scale * x + sy * scale * y
               for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    loop = np.array(corners + [corners[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=color, lw=lw)
    for c in corners:
        seg = np.array([pos, c])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, lw=lw * 0.7)


def _equal_aspect(ax, pts):
    """Give a 3D axis equal scaling on x/y/z around `pts`, so geometry is not
    visually distorted by matplotlib's independent per-axis autoscaling. The half-
    range is floored at 0.1 m to keep near-degenerate point sets viewable."""
    from mvis.analysis.display_frame import cube_bounds
    c, r = cube_bounds(pts, 0.1)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)


def plot_scenario(scenario, times=None, n_frustums=8, image_time=None,
                  save=None, figsize=(13, 6)):
    """Render a scenario: left = 3-D trajectory + board (world frame); right = the
    board as seen by the camera in one frame. Returns the matplotlib Figure;
    `save=<path.png>` writes it (headless Agg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    sc = scenario
    if times is None:
        times = np.linspace(sc.traj.start_time, sc.traj.end_time, 200)
    times = np.asarray(times, float)

    cam_pos, cam_R = [], []
    for t in times:
        R_GtoI, p = sc.traj.get_pose(t)
        T_CtoW = gtsam.Pose3(gtsam.Rot3(R_GtoI.T), gtsam.Point3(*p)).compose(sc.T_CtoI)
        cam_pos.append(np.asarray(T_CtoW.translation(), float))
        cam_R.append(T_CtoW.rotation().matrix())
    cam_pos = np.array(cam_pos)
    pts = sc.points_inW

    fig = plt.figure(figsize=figsize)

    # --- 3-D: trajectory + board ---
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2], "-",
            color="tab:blue", lw=1.5, label="camera path")
    ax.scatter(*cam_pos[0], color="green", s=45, label="start")
    for i in np.linspace(0, len(times) - 1, n_frustums).astype(int):
        _draw_frustum(ax, cam_pos[i], cam_R[i])
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="tab:red", s=8,
               label="board corners")
    ax.add_collection3d(Poly3DCollection([_board_face(sc.board, sc.T_BinW)],
                                         alpha=0.25, facecolor="tab:orange",
                                         edgecolor="k"))
    _equal_aspect(ax, np.vstack([cam_pos, pts]))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("trajectory + board  (world frame, z-up)")
    ax.legend(loc="upper right", fontsize=8)

    # --- 2-D: one camera frame ---
    ax2 = fig.add_subplot(1, 2, 2)
    tf = float(times[len(times) // 2]) if image_time is None else float(image_time)
    cam = sc.camera([tf])
    w, h = sc.image_size
    ax2.add_patch(plt.Rectangle((0, 0), w, h, fill=False, edgecolor="k"))
    if len(cam.uv[0]):
        ax2.scatter(cam.uv[0][:, 0], cam.uv[0][:, 1], c="tab:red", s=14)
    ax2.set_xlim(-20, w + 20)
    ax2.set_ylim(h + 20, -20)                 # image coords: v increases downward
    ax2.set_aspect("equal")
    ax2.set_xlabel("u"); ax2.set_ylabel("v")
    ax2.set_title(f"camera view @ t={tf:.2f}s  ({len(cam.ids[0])} corners)")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
    return fig
