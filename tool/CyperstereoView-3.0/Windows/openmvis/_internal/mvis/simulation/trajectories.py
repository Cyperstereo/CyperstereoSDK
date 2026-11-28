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
"""Load + shrink the repo's canonical sim trajectories (e.g. sine_3d) to a
tight workspace, for multi-board and other small-scene studies."""

import os

import numpy as np

from mvis.simulation.trajectory_io import load_trajectory
from mvis.simulation.bspline_se3 import BsplineSE3

# sine_3d ships inside the package; the few other pose-only trajectories the tests use live in
# the repo's tests/data/sim_trajectories in a source checkout. Rig examples with a full setup are
# in `mvis.simulation.examples`.
PACKAGED_DIR = os.path.join(os.path.dirname(__file__), "data", "trajectories")
SINE3D_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "tests", "data", "sim_trajectories")


def trajectory_path(name):
    """Path of trajectory ``<name>.txt``: the packaged copy if present, else the repo's."""
    packaged = os.path.join(PACKAGED_DIR, f"{name}.txt")
    return packaged if os.path.exists(packaged) else os.path.join(SINE3D_DIR, f"{name}.txt")


def load_sine3d(name="sine_3d", window=(0.0, 70.0), target_span=1.0, dt=0.05):
    """Load ``<name>.txt`` into a BsplineSE3, then (if ``target_span``) RECENTER it
    to the origin and scale its translation about the windowed centroid so the max
    peak-to-peak position span over the window equals ``target_span`` metres.
    Rotations are untouched, so the yaw sweep / gyro excitation is preserved.
    Returns the recentered+scaled BsplineSE3.

    Default ``sine_3d``, 70 s window -> ~2 yaw turns, ~1 m span centred on origin.
    """
    traj = load_trajectory(trajectory_path(name), dt=dt)
    if not target_span:
        return traj
    t_lo = traj.start_time + float(window[0])
    t_hi = traj.start_time + float(window[1])
    ts = np.arange(t_lo, t_hi, 0.1)
    P = np.array([traj.get_pose(t)[1] for t in ts])
    centroid = P.mean(0)
    span = float((P.max(0) - P.min(0)).max())
    s = float(target_span) / span
    ctrl = [Q.copy() for Q in traj.ctrl]
    for Q in ctrl:
        Q[:3, 3] = s * (Q[:3, 3] - centroid)     # recenter to origin + scale
    return BsplineSE3(traj.t0, traj.dt, ctrl)
