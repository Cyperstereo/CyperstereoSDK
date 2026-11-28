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
"""Simulator examples: a real motion and a full rig setup, ready to simulate.

Each example is a folder with `trajectory.txt` (`t theta_ItoG(3) p_IinG(3)`, 200 Hz, the base IMU
in a gravity-aligned world), `setup.yaml` (cameras, IMUs, board) and a README. They come from our
digital twins and calibration-motion studies; every truth is the real solve moved off by the fixed
seeded perturbation (`mvis.simulation.twin.perturb`), so no real device calibration
ships. They are regenerated from the twins by maintainer tooling, not part of the package.

    from mvis.simulation.examples import load_example, simulate
    ex = load_example("virig_4imus_3cams")
    data = simulate(ex, seed=0)
    result = run_cam_imu_calibration_multi(data["cams"], ex.board_points_flat, data["t_imu"],
                                           data["w_m"], data["a_m"], aux_imus=data["aux_imus"] or None)

The wheel ships `twin_gs` only; a source checkout also has the others under `data/sim_examples/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gtsam
import numpy as np
import yaml

from mvis.simulation.bspline_se3 import BsplineSE3
from mvis.simulation.trajectory_io import load_trajectory

PACKAGED_DIR = Path(__file__).resolve().parent / "data" / "examples"
REPO_DIR = Path(__file__).resolve().parents[3] / "data" / "sim_examples"
# the order of the docs: own dataset twins first, then public, calibration motions, regression twins
EXAMPLES = ("virig_4imus_3cams", "elp_t265", "tum_vi_raw", "guide06c_60", "guide60b",
            "twin_gs", "twin_rs")
EDGE_S = 0.2              # keep simulated samples this far inside the trajectory (s)


@dataclass
class SimExample:
    """One example: the base-IMU trajectory and the rig setup (`setup.yaml` as a dict)."""
    name: str
    path: Path
    setup: dict
    traj: BsplineSE3           # T_ItoW of the base IMU
    board_points_flat: np.ndarray   # (N, 3) board corners as the solver sees them (z = 0)
    board_points: np.ndarray   # (N, 3) true corners: the flat board with its measured heights

    @property
    def T_BinW(self) -> np.ndarray:
        return np.asarray(self.setup["board"]["T_BinW"], float)

    def board_points_world(self) -> np.ndarray:
        T = self.T_BinW
        return self.board_points @ T[:3, :3].T + T[:3, 3]


def example_dir(name: str) -> Path:
    """Folder of example `name`: the packaged copy if present, else the source checkout's."""
    for root in (PACKAGED_DIR, REPO_DIR):
        if (root / name / "setup.yaml").exists():
            return root / name
    raise FileNotFoundError(f"no simulator example {name!r} (known: {', '.join(available())})")


def available() -> list[str]:
    """Names of the examples found, packaged or in the source checkout, in `EXAMPLES` order."""
    found = {p.parent.name for root in (PACKAGED_DIR, REPO_DIR) if root.is_dir()
             for p in root.glob("*/setup.yaml")}
    return [n for n in EXAMPLES if n in found] + sorted(found - set(EXAMPLES))


def board_points_flat(board: dict) -> np.ndarray:
    """(N, 3) corners of the `board` block of a setup, from the MVIS board model."""
    from mvis.core.config import BoardConfig
    from mvis.detection.board_factory import create_board
    keys = ("board_type", "rows", "cols", "square_size", "tag_size", "tag_spacing", "tag_family")
    cfg = BoardConfig(**{k: board[k] for k in keys if k in board})
    return np.asarray(create_board(cfg).all_points_3d(), float)


def load_example(name: str, dt: float | None = None) -> SimExample:
    """Load example `name`: the trajectory as a `BsplineSE3` (control spacing `dt`, default the
    file's 5 ms) and the setup."""
    d = example_dir(name)
    with open(d / "setup.yaml") as f:
        setup = yaml.safe_load(f)
    flat = board_points_flat(setup["board"])
    true = flat.copy()
    if setup["board"].get("heights_m") is not None:
        true[:, 2] = np.asarray(setup["board"]["heights_m"], float)
    return SimExample(name=name, path=d, setup=setup, traj=load_trajectory(str(d / "trajectory.txt"), dt=dt),
                      board_points_flat=flat, board_points=true)


def _imu_intr(imu: dict):
    from mvis.imu import ImuModel, parse_imu_model
    return ImuModel.from_tangent(np.asarray(imu["intrinsics_xi24"], float), parse_imu_model(imu["model"]))


def _times(t_lo, t_hi, rate):
    from mvis.simulation.imu_measurements import imu_times
    return imu_times(t_lo, t_hi, float(rate))


def simulate(ex: SimExample, seed: int = 0, noise: bool = True, t0: float | None = None,
             t1: float | None = None) -> dict:
    """Simulate the example's cameras and IMUs over [t0, t1] (default: the trajectory less `EDGE_S`
    at each end). Returns `cams` (camera dicts for `run_cam_imu_calibration_multi`, each with the
    true intrinsics as `cal_vec`), `t_imu`, `w_m`, `a_m` (base IMU) and `aux_imus`.

    Each camera samples at its rate with its td and rolling-shutter readout; frames whose exposure
    leaves the trajectory are dropped. The base accel lags the gyro by t_ga; an aux IMU sample
    stamped t was measured at t + td (gyro) and t + td - t_ga (accel). `noise=False` gives exact
    measurements; otherwise corners get `corner_sigma_px` and every IMU the base IMU's noise
    densities (as in the twins).
    """
    from mvis.simulation.camera_measurements import simulate_camera
    from mvis.simulation.from_result import cal_from_vec
    from mvis.simulation.imu_measurements import simulate_imu

    s = ex.setup
    lo, hi = ex.traj.start_time + EDGE_S, ex.traj.end_time - EDGE_S
    t_lo = lo if t0 is None else max(lo, float(t0))
    t_hi = hi if t1 is None else min(hi, float(t1))
    if t_hi - t_lo < 1.0:
        raise ValueError(f"{ex.name}: window [{t_lo}, {t_hi}] is under 1 s")
    g = np.asarray(s["world"]["g_inW"], float)
    pts_W = ex.board_points_world()

    cams = []
    for c, cam in enumerate(s["cameras"]):
        td, readout = float(cam["td_s"]), float(cam["readout_s"])
        reach = (min(0.0, readout), max(0.0, readout))
        t_cam = _times(t_lo, t_hi, cam["rate_hz"])
        t_cam = t_cam[(t_cam + td + reach[0] > lo) & (t_cam + td + reach[1] < hi)]
        model = cam["model"]
        intr = np.asarray(cam["intrinsics"], float)
        st = simulate_camera(ex.traj, t_cam, cal_from_vec(intr, model == "equidistant", model),
                             gtsam.Pose3(np.asarray(cam["T_CtoI"], float)), pts_W, tuple(cam["image_size"]),
                             td=td, sigma_px=float(cam["corner_sigma_px"]) if noise else 0.0,
                             readout=readout, rolling_mode=int(cam.get("rolling_mode", 0)), seed=1000 + seed + c)
        keep = [k for k, ids in enumerate(st.ids) if len(ids)]
        cams.append(dict(t_cam=np.asarray(t_cam)[keep], ids_list=[np.asarray(st.ids[k], int) for k in keep],
                         uv_list=[np.asarray(st.uv[k], float) for k in keep], cal_vec=intr,
                         is_fisheye=model == "equidistant", model=model, image_size=tuple(cam["image_size"]),
                         pin_intrinsics=False, estimate_readout=readout != 0.0,
                         rolling_mode=int(cam.get("rolling_mode", 0)), frames=None,
                         retime_sensor_clock=False))

    imu = s["imu"]
    nk = dict(imu["noise"]) if noise else {}
    t_imu = _times(t_lo, t_hi, imu["rate_hz"])
    kw = dict(g_inG=g, b_g0=imu["b_g"], b_a0=imu["b_a"], seed=seed + 100, **nk)
    gyro = simulate_imu(ex.traj, t_imu, _imu_intr(imu), **kw)
    accel = simulate_imu(ex.traj, t_imu, _imu_intr(imu), td=-float(imu["t_ga_s"]), **kw)

    aux = []
    for i, a in enumerate(s.get("aux_imus") or []):
        td, T = float(a["td_s"]), np.asarray(a["T_IstoI"], float)
        t = _times(t_lo, t_hi, a["rate_hz"])
        t = t[(t + td - abs(float(a["t_ga_s"])) > lo) & (t + td + abs(float(a["t_ga_s"])) < hi)]
        kw = dict(g_inG=g, R_IstoI=T[:3, :3], p_IsinI=T[:3, 3], b_g0=a["b_g"], b_a0=a["b_a"],
                  seed=seed + 200 + i, **nk)
        ag = simulate_imu(ex.traj, t, _imu_intr(a | {"model": imu["model"]}), td=td, **kw)
        aa = simulate_imu(ex.traj, t, _imu_intr(a | {"model": imu["model"]}), td=td - float(a["t_ga_s"]), **kw)
        aux.append(dict(t_imu=ag.t, w_m=ag.w_m, a_m=aa.a_m, name=a["name"], gyro_only=bool(a.get("gyro_only"))))
    return dict(cams=cams, t_imu=gyro.t, w_m=gyro.w_m, a_m=accel.a_m, aux_imus=aux)
