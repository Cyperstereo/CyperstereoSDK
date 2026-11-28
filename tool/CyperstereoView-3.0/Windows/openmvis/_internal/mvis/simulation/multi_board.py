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
"""Multi-board simulation scenario: N known boards ringing a trajectory, with
board-tagged corner IDs so each detection knows which board it belongs to.

Supports two camera regimes (set by `cal`/`is_fisheye`/`image_size`):
  * t265 fisheye -> wide FoV, all boards co-visible every frame (direct multi-target);
  * narrow pinhole -> boards drop in/out one at a time (IMU-bridge case).
Use `forward_T_CtoI()` so the optical axis looks horizontally OUTWARD at the ring
(an upright mount points the camera ~71 deg up and barely sees the boards).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import gtsam

from mvis.simulation.imu_measurements import GRAVITY
from mvis.simulation.scenario import _ImuStreams
from mvis.simulation.camera_measurements import (
    board_points_in_global, simulate_camera, CamStream)

BOARD_ID_STRIDE = 100000

# Captured ground truth of a real VI-Rig 4imus_3cams april_01 calibration (a real
# solve, intrinsics pinned), shipped with the package so the default config works in
# any checkout and from any cwd. The per-sensor frame_times/sample_times replay
# lists were dropped (only the states, board pose and rig extrinsics are read
# here); still loadable with `gt_from_json`.
DEFAULT_GT_FILE = str(Path(__file__).resolve().parent / "data" / "gt_3cam.json")


def board_of(gid):
    """Board index carried by a board-tagged global corner ID (the inverse of the
    `b * BOARD_ID_STRIDE + local` tagging; pairs with `local_of`)."""
    return np.asarray(gid) // BOARD_ID_STRIDE


def local_of(gid):
    """Board-local corner index inside a board-tagged global ID (see `board_of`)."""
    return np.asarray(gid) % BOARD_ID_STRIDE


def place_boards_ring(board, *, radius=1.5, center=(0, 0, 0), directions=4,
                      tilt=0.0) -> list[gtsam.Pose3]:
    """Place `directions` copies of `board` evenly around a circle of `radius`
    in the XY plane, each centroid `radius` from `center` and oriented so its
    surface normal (+z_B) points back toward `center`. `tilt` (rad) lifts each
    board's normal toward +z_W for variety."""
    center = np.asarray(center, float)
    centroid_B = np.asarray(board.all_points_3d(), float).mean(0)
    poses = []
    for k in range(directions):
        ang = 2 * np.pi * k / directions
        out = np.array([np.cos(ang), np.sin(ang), 0.0])      # board center dir
        center_W = center + radius * out
        n = -out * np.cos(tilt) + np.array([0, 0, 1.0]) * np.sin(tilt)  # +z_B back at center
        up = np.array([0, 0, 1.0])
        x_B = np.cross(up, n); x_B /= np.linalg.norm(x_B)
        y_B = np.cross(n, x_B)
        R_BinW = np.column_stack([x_B, y_B, n])
        t_BinW = center_W - R_BinW @ centroid_B
        poses.append(gtsam.Pose3(gtsam.Rot3(R_BinW), gtsam.Point3(*t_BinW)))
    return poses


# Forward-looking camera mount: optical axis (z_cam) -> body +x, so the camera
# looks ~horizontally outward at the ring (NOT up). Without this the axis sits
# ~71 deg above horizontal (verified) and only the fisheye rim catches boards.
FWD_MOUNT = gtsam.Rot3(np.array([[0, 0, 1.], [-1, 0, 0], [0, -1, 0]]))


def forward_T_CtoI(err_rotvec=(0.02, -0.03, 0.05), t=(0.05, -0.02, 0.01)) -> gtsam.Pose3:
    """Forward-looking cam-IMU extrinsic = FWD_MOUNT composed with a small
    mounting error, at lever `t`. Optical axis ends up ~10 deg below horizontal."""
    R = FWD_MOUNT.compose(gtsam.Rot3.Expmap(np.asarray(err_rotvec, float)))
    return gtsam.Pose3(R, gtsam.Point3(*t))


@dataclass
class MultiBoardScenario(_ImuStreams):
    """Ground truth for an N-board scenario + the measurement generators that
    realize it (camera, base IMU, optional aux IMU).

    The single-board `CalibScenario` analogue, differing in one contract: corner
    IDs are **board-tagged** (`b * BOARD_ID_STRIDE + local`), so a detection
    identifies its board without a separate association step. Whether the boards
    are co-visible is set by the camera model, not by this class — see the module
    docstring on the fisheye vs pinhole regimes.
    """
    traj: object
    boards: list
    T_BinW_list: list
    cal: object
    T_CtoI: gtsam.Pose3
    image_size: tuple
    imu_intr: object
    b_g: np.ndarray
    b_a: np.ndarray
    is_fisheye: bool = False
    g_inW: np.ndarray = field(default_factory=lambda: GRAVITY.copy())
    td_cam: float = 0.0
    sigma_px: float = 0.0
    sigma_w: float = 0.0      # IMU gyro / accel white-noise densities + bias RW
    sigma_a: float = 0.0
    sigma_bg: float = 0.0
    sigma_ba: float = 0.0
    readout: float = 0.0      # rolling-shutter total top-to-bottom readout (s); 0 = global
    rolling_mode: int = 0     # 0 top / 1 middle / 2 bottom frame-time reference
    aux: object = None        # optional 2nd IMU (mvis.simulation.scenario.AuxImu)

    def board_points(self, b) -> np.ndarray:
        """Board `b`'s corners in its OWN board frame (the metric target geometry)."""
        return np.asarray(self.boards[b].all_points_3d(), float)

    def points_inW(self, b) -> np.ndarray:
        """Board `b`'s corners in the world frame, i.e. `board_points(b)` carried
        through that board's placement `T_BinW_list[b]`."""
        return board_points_in_global(self.boards[b], self.T_BinW_list[b])

    def _all_points_inW(self):
        """Stacked world points + their global (board-tagged) IDs."""
        P, gid = [], []
        for b in range(len(self.boards)):
            pts = self.points_inW(b)
            n = len(pts)
            P.append(pts)
            gid.append(b * BOARD_ID_STRIDE + np.arange(n))
        return np.vstack(P), np.concatenate(gid)

    def camera(self, times, seed: int = 0) -> CamStream:
        """Project all boards; remap local point indices to board-tagged IDs."""
        P, gid = self._all_points_inW()
        raw = simulate_camera(self.traj, times, self.cal, self.T_CtoI, P,
                              self.image_size, td=self.td_cam,
                              sigma_px=self.sigma_px, readout=self.readout,
                              rolling_mode=self.rolling_mode, seed=seed)
        ids = [gid[np.asarray(idx, int)] if len(idx) else np.array([], int)
               for idx in raw.ids]
        return CamStream(raw.t, ids, raw.uv)


def visibility_audit(stream: CamStream, n_boards: int) -> dict:
    """Report per-board frame counts, # frames seeing >1 board, and each board's
    first/last frame window. Co-visibility is EXPECTED with the fisheye and
    ABSENT with the pinhole — this reports it, it does not assert a value."""
    per_board = [0] * n_boards
    co_visible = 0
    windows = [[None, None] for _ in range(n_boards)]
    for f, ids in enumerate(stream.ids):
        if not len(ids):
            continue
        present = set(int(b) for b in np.unique(board_of(np.asarray(ids, int))))
        if len(present) > 1:
            co_visible += 1
        for b in present:
            per_board[b] += 1
            if windows[b][0] is None:
                windows[b][0] = f
            windows[b][1] = f
    return {'per_board_frames': per_board, 'co_visible_frames': co_visible,
            'windows': [tuple(w) for w in windows]}


# --- canonical real-rig intrinsics ------------------------------------------
T265_FISHEYE = np.array([288.74401, 288.91833, 416.14622, 405.97177,
                         -0.00777028, 0.04423187, -0.04124983, 0.00730763])  # Cal3Fisheye (virig 3cams)
PINHOLE_RADTAN = np.array([400., 400., 320., 240.,
                           -0.28, 0.07, 0.0005, -0.0004])                    # Cal3DS2 (generic)
# real ELP cam0 (left) from the 4imus_4cams April rig (narrow pinhole, 640x480)
ELP_PINHOLE = np.array([427.42363, 426.76205, 338.89618, 268.30866,
                        -0.36494510, 0.09841813, 0.00032057, 0.00144252])    # Cal3DS2


@dataclass
class MultiBoardSimConfig:
    """Single source of truth for a multi-board cam-IMU SIMULATION. `build()`
    constructs the scenario; `simulate()` returns everything `run_multi_board_
    calibration` needs.

    DEFAULT = the realistic single-board case: real ELP camera intrinsics on the
    REAL captured calibration trajectory + real board pose, ONE board + ONE IMU,
    1 px corner + physical MVIS IMU noise. This converges nicely under realistic
    noise (a real calibration motion excites all rotation axes). Use `.fisheye()`
    / `.pinhole()` for the synthetic N-board sine_3d demos.
    """
    # --- trajectory: real captured calibration trajectory (GT json) by default;
    # set traj_file="" + a traj_name to use the synthetic sine_3d instead. ---
    traj_name: str = "sine_3d"
    traj_window: tuple = (0.0, 70.0)
    target_span: float = 0.0          # workspace span (m); 0/None = keep real scale
    traj_dt: float = 0.05
    traj_file: str = DEFAULT_GT_FILE   # real trajectory from this GT json (packaged)
    gt_cam_index: int = 0             # which GT camera's extrinsic to reuse (mount="real")
    # --- boards ---
    n_boards: int = 1
    board_rows: int = 7
    board_cols: int = 7               # board width = (cols-1)*board_square
    board_square: float = 0.10        # m
    board_placement: str = "real"     # "real" (GT board pose) | "ring" (N facing inward) | "facing" (1 in front)
    board_depth: float = 2.5          # facing mode: board distance in front of cam (m)
    ring_radius: float = 1.5          # m
    ring_center: tuple = (0.0, 0.0, 0.0)
    board_tilt: float = 0.0           # ring-normal tilt toward +z (rad)
    mount: str = "real"               # "real" (GT T_CtoI) | "forward" (ring) | "frontal" (facing)
    # --- camera ---
    is_fisheye: bool = False
    camera_model: str | None = None   # distortion model; None = from is_fisheye (radtan/equidistant)
    cam_intrinsics: tuple = tuple(ELP_PINHOLE)    # [fx,fy,cx,cy,dist...] real 4cam ELP
    image_size: tuple = (640, 480)
    # --- cam-IMU extrinsic (GT): forward-looking mount + lever ---
    mount_err_rotvec: tuple = (0.02, -0.03, 0.05)
    lever: tuple = (0.05, -0.02, 0.01)            # cam-IMU translation (m)
    # --- IMU intrinsics + bias (GT) --- (calyx = estimator default). calyx gyro
    # Dw is FULL 3x3 (9 params, diag at 0/4/8); accel Da is triangular (6 params,
    # diag at 0/2/5); Tg is 9.
    imu_model: str = "calyx"
    imu_dw: tuple = (1.01, 0.02, -0.01, 0.005, 0.99, 0.015, -0.01, 0.008, 1.02)
    imu_da: tuple = (0.98, -0.02, 1.01, 0.01, -0.01, 0.99)
    imu_tg: tuple = tuple([0.0] * 9)
    imu_R_rotvec: tuple = (0.0, 0.0, 0.0)
    b_g: tuple = (0.0, 0.0, 0.0)
    b_a: tuple = (0.0, 0.0, 0.0)
    # --- optional 2nd IMU (lever arm gives differential α×r specific force) ---
    aux_enable: bool = False
    aux_lever: tuple = (0.10, -0.05, 0.02)   # NON-ZERO base->aux translation (m)
    aux_rotvec: tuple = (0.05, -0.03, 0.08)  # base->aux rotation
    aux_td: float = 0.0
    # --- sampling / noise ---
    cam_hz: float = 10.0
    imu_hz: float = 200.0
    duration: float = 80.0            # calibration window length (s); real default traj is 80 s
    sigma_px: float = 1.0             # corner pixel noise (0 = noise-free)
    # IMU measurement noise densities = the PHYSICAL MVIS rig densities (what the
    # sim-recovery estimator actually uses via calib_cfg, NOT the loose 0.01
    # CamImuCalibConfig placeholder). `estimator_config()` weights the solve to match.
    imu_sigma_w: float = 0.00016968
    imu_sigma_a: float = 0.003
    imu_sigma_bg: float = 1.9393e-05
    imu_sigma_ba: float = 0.003
    intr_perturb_px: float = 0.0      # perturb the camera-intrinsics SEED (fx,fy,cx,cy) by N(0,this) px
    # rolling shutter: total readout time (s) and frame-time reference. 0 = global
    # shutter. The real ELP cam is rolling: line_delay 3.636e-5 s/row * 480 = 17.45 ms.
    readout: float = 0.0
    rolling_mode: int = 0
    seed: int = 1

    @classmethod
    def fisheye(cls, **kw):
        """Synthetic demo: t265 fisheye + 4x 1 m boards on a sine_3d ring -> all
        co-visible (direct chaining). NOTE: sine_3d's roll/pitch is weak, so this
        is fragile under realistic IMU noise (the real-trajectory default isn't)."""
        base = dict(traj_name="sine_3d", traj_file="", target_span=1.0,
                    n_boards=4, board_rows=11, board_cols=11, board_square=0.10,
                    board_placement="ring", mount="forward", is_fisheye=True,
                    cam_intrinsics=tuple(T265_FISHEYE), image_size=(848, 800))
        base.update(kw)
        return cls(**base)

    @classmethod
    def pinhole(cls, **kw):
        """Synthetic demo: narrow pinhole + 4x 0.28 m boards on a sine_3d ring ->
        non-co-visible (IMU-bridge case)."""
        base = dict(traj_name="sine_3d", traj_file="", target_span=1.0,
                    n_boards=4, board_rows=7, board_cols=7, board_square=0.04,
                    board_placement="ring", mount="forward", is_fisheye=False,
                    cam_intrinsics=tuple(PINHOLE_RADTAN), image_size=(640, 480))
        base.update(kw)
        return cls(**base)

    # ----- named config templates (the canonical scenarios we run) -----
    @property
    def gt_source(self) -> str:
        """How ground truth is obtained: `captured` (real GT json via traj_file),
        `synthetic` (sine_3d, GT defined directly). [`grounded` (live bag) deferred.]"""
        return "captured" if self.traj_file else "synthetic"

    @classmethod
    def realistic_single_board(cls, **kw):
        """DEFAULT: real ELP cam + real captured trajectory + real board, realistic noise."""
        return cls(**kw)

    @classmethod
    def synthetic_multiboard_fisheye(cls, **kw):
        """Synthetic 4-board fisheye sine_3d demo (noise-free hypothesis test)."""
        return cls.fisheye(**kw)

    @classmethod
    def synthetic_multiboard_pinhole(cls, **kw):
        """Synthetic 4-board narrow-pinhole sine_3d demo (IMU-bridge case)."""
        return cls.pinhole(**kw)

    @classmethod
    def rolling_shutter(cls, **kw):
        """Realistic single-board + the real ELP rolling shutter (17.45 ms readout)."""
        return cls(readout=0.0174519, rolling_mode=0, **kw)

    @classmethod
    def two_imu(cls, **kw):
        """Realistic single-board + a 2nd IMU at a non-zero lever arm."""
        return cls(aux_enable=True, **kw)

    # -- builders --
    def board(self):
        """The `Chessboard` target these settings describe (rows/cols/square size)."""
        from mvis.detection.chessboard import Chessboard
        return Chessboard(self.board_rows, self.board_cols, self.board_square)

    def cal(self):
        """Simulator calibration object for the TRUE camera intrinsics — gtsam
        `Cal3Fisheye` / `Cal3DS2` for equidistant / radtan, else a CameraModel
        adapter (`from_result.ModelCal`) for `camera_model`. (`simulate()` feeds
        the estimator a separately perturbed SEED vector, not this.)"""
        from mvis.simulation.from_result import cal_from_vec
        return cal_from_vec(self.cam_intrinsics, self.is_fisheye, self.camera_model)

    def T_CtoI(self):
        """The ground-truth camera->IMU extrinsic for a synthetic mount: `frontal`
        applies `mount_err_rotvec` about identity (camera faces the board), anything
        else uses the outward-looking ring mount. The `mount == "real"` case is not
        handled here — `build()` reads that pose from the captured GT instead."""
        if self.mount == "frontal":      # standard small-rotation mount (real/facing)
            return gtsam.Pose3(gtsam.Rot3.Expmap(np.asarray(self.mount_err_rotvec, float)),
                               gtsam.Point3(*self.lever))
        return forward_T_CtoI(self.mount_err_rotvec, self.lever)   # ring (outward)

    def _gt(self):
        """Load the captured ground truth referenced by `traj_file` (`gt_source ==
        "captured"` only)."""
        from mvis.simulation.from_result import gt_from_json
        return gt_from_json(self.traj_file)

    def trajectory(self, g=None):
        """The continuous-time trajectory to simulate along: a B-spline fitted to the
        captured GT states when `traj_file` is set (pass `g` to reuse an already-loaded
        GT), otherwise the named analytic `sine_3d` path. Returns a spline exposing
        `start_time`/`end_time`."""
        if self.traj_file:               # real captured calibration trajectory
            from mvis.simulation.from_result import fit_bspline_from_states
            g = g or self._gt()
            spline, _fit_err = fit_bspline_from_states(
                np.asarray(g.t_states, float),
                [np.asarray(T, float) for T in g.T_states], self.traj_dt)
            return spline
        from mvis.simulation.trajectories import load_sine3d
        return load_sine3d(self.traj_name, window=self.traj_window,
                           target_span=self.target_span, dt=self.traj_dt)

    def imu_intrinsics(self):
        """Ground-truth base-IMU intrinsics: the `imu_dw`/`imu_da`/`imu_tg` scale-
        misalignment + g-sensitivity terms and the `imu_R_rotvec` frame rotation,
        packed under the selected `imu_model` convention (rpng/kalibr/calyx — the
        conventions differ in which triangle of Dw/Da is parameterized)."""
        from mvis.imu import ImuModel, RPNG, KALIBR, CALYX
        m = {"rpng": RPNG, "kalibr": KALIBR, "calyx": CALYX}[self.imu_model]
        R = gtsam.Rot3.Expmap(np.asarray(self.imu_R_rotvec, float)).matrix()
        return ImuModel(x_dw=np.asarray(self.imu_dw, float),
                        x_da=np.asarray(self.imu_da, float),
                        x_tg=np.asarray(self.imu_tg, float), R=R, model=m)

    def model_name(self):
        """Distortion-model name: `camera_model`, else the one matching
        `is_fisheye`, in the spelling the config layer and output files use
        (`equidistant` / `radtan`)."""
        return self.camera_model or ("equidistant" if self.is_fisheye else "radtan")

    def estimator_config(self, **over):
        """CamImuCalibConfig weighted to MATCH the sim noise (sigma_px + IMU
        densities) and the GT IMU model, so the estimator is statistically
        consistent with the data. Thin wrapper over the shared `matched_calib_cfg`."""
        from mvis.simulation.noise import ImuNoiseDensities, matched_calib_cfg
        noise = ImuNoiseDensities(self.imu_sigma_w, self.imu_sigma_bg,
                                  self.imu_sigma_a, self.imu_sigma_ba)
        return matched_calib_cfg(noise, imu_model=self.imu_model,
                                 sigma_px=self.sigma_px, **over)

    def aux(self):
        """The 2nd IMU (`AuxImu`) at a non-zero lever, or None. Identity intrinsics."""
        if not self.aux_enable:
            return None
        from mvis.simulation.scenario import AuxImu
        from mvis.imu import create_imu_model
        from mvis.imu import RPNG, CALYX, KALIBR
        m = {"rpng": RPNG, "kalibr": KALIBR, "calyx": CALYX}[self.imu_model]
        T_IstoI = gtsam.Pose3(gtsam.Rot3.Expmap(np.asarray(self.aux_rotvec, float)),
                              gtsam.Point3(*self.aux_lever))
        return AuxImu(T_IstoI=T_IstoI, intr=create_imu_model(m),
                      b_g=np.zeros(3), b_a=np.zeros(3), td=self.aux_td)

    def build(self) -> "MultiBoardScenario":
        """Assemble the ground-truth `MultiBoardScenario` these settings describe.

        Resolves the three "real or synthetic" choices independently: the trajectory
        (captured B-spline vs `sine_3d`), the cam-IMU extrinsic (`mount == "real"`
        reads the captured GT, else `T_CtoI()`), and the board placement
        (`real` = the captured board pose, `facing` = one board placed in front of
        the camera at `board_depth`, otherwise `n_boards` on the ring). Ground truth
        only — no sampling happens here; see `simulate()`.
        """
        g = self._gt() if self.traj_file else None
        traj = self.trajectory(g)
        board = self.board()
        # cam-IMU extrinsic: real (from GT), frontal, or forward ring mount
        if self.mount == "real" and g is not None:
            T_CtoI = gtsam.Pose3(np.asarray(g.cams[self.gt_cam_index]['T_CtoI'], float))
        else:
            T_CtoI = self.T_CtoI()
        # board placement: real GT pose, frontal facing, or ring
        if self.board_placement == "real" and g is not None:
            poses = [gtsam.Pose3(np.asarray(g.T_BinW, float))]    # REAL board pose
        elif self.board_placement == "facing":   # ONE board synthetically in front
            from mvis.simulation.camera_measurements import place_board_facing_camera
            t0 = traj.start_time + self.traj_window[0]
            poses = [place_board_facing_camera(traj, t0, self.board_depth, board, T_CtoI)]
        else:                                    # N boards on the ring
            poses = place_boards_ring(board, radius=self.ring_radius,
                                      center=self.ring_center, directions=self.n_boards,
                                      tilt=self.board_tilt)
        return MultiBoardScenario(
            traj=traj, boards=[board] * len(poses), T_BinW_list=poses,
            cal=self.cal(), is_fisheye=self.is_fisheye, T_CtoI=T_CtoI,
            image_size=tuple(self.image_size), imu_intr=self.imu_intrinsics(),
            b_g=np.asarray(self.b_g, float), b_a=np.asarray(self.b_a, float),
            sigma_px=self.sigma_px, aux=self.aux(),
            sigma_w=self.imu_sigma_w, sigma_a=self.imu_sigma_a,
            sigma_bg=self.imu_sigma_bg, sigma_ba=self.imu_sigma_ba,
            readout=self.readout, rolling_mode=self.rolling_mode)

    def simulate(self):
        """Build the scenario and sample sensors -> (scenario, cam_dict, imu,
        board_points, gt, aux_imus). cam_dict + imu + board_points (+ aux_imus)
        feed `run_multi_board_calibration`; gt has the ground-truth params.
        `aux_imus` is [] unless `aux_enable` (then one dict with t_imu/w_m/a_m)."""
        from mvis.simulation.imu_measurements import imu_times
        scn = self.build()
        t0 = scn.traj.start_time
        dur = min(self.duration, scn.traj.end_time - t0 - 0.2)   # cap to traj length
        cam = scn.camera(imu_times(t0, t0 + dur, self.cam_hz), seed=self.seed)
        t_imu = imu_times(t0, t0 + dur, self.imu_hz)
        imu = scn.base_imu(t_imu, seed=self.seed)
        # the SEED intrinsics fed to the calibrator (the sim used the TRUE ones);
        # optionally perturb fx,fy,cx,cy by N(0, intr_perturb_px) so the solve must
        # recover the camera intrinsics from an imperfect prior.
        cal_seed = np.asarray(self.cam_intrinsics, float).copy()
        if self.intr_perturb_px:
            rng = np.random.RandomState(self.seed + 7)
            cal_seed[:4] += self.intr_perturb_px * rng.standard_normal(4)
        cam_dict = dict(t_cam=cam.t, ids_list=cam.ids, uv_list=cam.uv,
                        cal_vec=cal_seed, model=self.model_name(),
                        is_fisheye=self.is_fisheye, image_size=tuple(self.image_size),
                        rolling_mode=self.rolling_mode,
                        estimate_readout=bool(self.readout))   # rolling shutter -> estimate tr
        board_points = {b: scn.board_points(b) for b in range(len(scn.boards))}
        aux_imus = []
        if self.aux_enable:
            a = scn.aux_imu(t_imu, seed=self.seed + 1)
            aux_imus = [dict(t_imu=a.t, w_m=a.w_m, a_m=a.a_m)]
        gt = dict(T_CtoI=scn.T_CtoI, board_poses=scn.T_BinW_list,
                  imu_intr=scn.imu_intr, cam_intrinsics=np.asarray(self.cam_intrinsics, float),
                  aux=scn.aux, readout=self.readout)
        return scn, cam_dict, imu, board_points, gt, aux_imus


# canonical name for the unified sim config (one pipeline, many gt_sources)
SimConfig = MultiBoardSimConfig
