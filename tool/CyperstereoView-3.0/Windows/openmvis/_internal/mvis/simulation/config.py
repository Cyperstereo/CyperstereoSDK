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
"""Simulation-study config — first-class home for the grounded sim studies.

`SimStudyConfig` fully specifies one grounded simulation run (grounding → GT
trajectory → simulated sensors → recovery) as one dataclass, so a study is a
value that can be built, varied and passed to `mvis.simulation.run`.

The default `bag` / `camchain` name the VI-Rig 4imus_3cams recording under
`$MVIS_DATASETS` (or `~/Datasets`) rather than a hardcoded absolute path; set
them to your own recording and camera prior (a `*_calib.yaml` from `mvis cam`
or a Kalibr camchain) for any other rig.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from mvis.core.paths import datasets_root
from mvis.simulation.noise import (
    MVIS_IMU_NOISE, ImuNoiseDensities, matched_calib_cfg)

# Root for the MVIS calibration datasets the grounded studies read (`core.paths.datasets_root`,
# read once at import).
DATASET_ROOT = datasets_root()
_DS = os.path.join(DATASET_ROOT, "mvis_4imus_3cams")


@dataclass
class SimStudyConfig:
    """Settings for one GROUNDED simulation study — the variant whose ground truth
    is derived from a real recording rather than invented.

    The study runs in four stages, matching the field groups below: (1) calibrate a
    real bag window to get a trustworthy GT rig, (2) fit a GT B-spline trajectory to
    it, (3) resample synthetic camera + IMU streams along that trajectory at known
    noise levels, (4) re-solve and compare against the GT that produced the data.
    The last group is the sweep spec — how many seeds, and whether to grid over
    injected time offset and/or state rate.

    Contrast `MultiBoardSimConfig`, whose ground truth is synthetic or captured; this
    one is the `gt_source == "grounded"` path, so it needs the dataset present.
    """
    # --- 1. grounding (real data -> ground truth) ---
    bag: str = os.path.join(_DS, "data_arucotags", "virig_4imus_3cams_aurco_16.bag")
    camchain: str = os.path.join(_DS, "virig_4imus_3cams_april_01-camchain.yaml")
    cam_topic: str = "/blackfly/image_raw"
    imu_topic: str = "/gx3_25/data"
    detector: str = "aruco"       # "aruco" (ArucoFeatureDetector) | "april" (AprilGrid corners as features)
    aruco_dict: str = "DICT_6X6_250"
    t0: float = 78.0              # window start (s) — richest moving slice t[78-122]
    window: float = 44.0
    ground_hz: float = 8.0        # grounding-calib state rate (<= real_cam_hz)
    ground_pin_intr: bool = True  # pin cam intrinsics in the GROUNDING calib (robust GT)
    sfm_mode: str = "general"     # "general" (essential) | "planar" (homography, for a planar board)
    real_cam_hz: float = 30.0     # measured aurco_16 blackfly rate (for validation)

    # --- 2. trajectory (GT B-spline) ---
    knot_dt: float = 0.0          # 0 = grounding spacing; else resample via from_pose_fn

    # --- 3. simulated sensors ---
    cam_hz: float = 20.0          # sim camera rate (MUST be int multiple of each state_hz)
    imu_hz: float = 400.0         # sim IMU rate (>= ~10-15x max state_hz)
    sigma_px: float = 0.5
    sigma_w: float = MVIS_IMU_NOISE.sigma_w    # MVIS densities (mvis.simulation.noise)
    sigma_bg: float = MVIS_IMU_NOISE.sigma_bg
    sigma_a: float = MVIS_IMU_NOISE.sigma_a
    sigma_ba: float = MVIS_IMU_NOISE.sigma_ba
    td_inject: float | None = None  # None = inject GT td; 0.0 = off; else explicit (s)

    # --- 4. recovery / estimator ---
    state_hz: float = 8.0
    model: str = "radtan"         # CAMERA distortion model
    imu_model: str = "calyx"      # IMU-intrinsic model family (kalibr|rpng|calyx);
                                  # matches the project-wide CamImuCalibConfig default
    pin_intr: bool = False        # cam intrinsics+distortion (False = FREE)
    pin_td: bool = False          # pin recovery td at the injected/GT value
    pin_imu: bool = False         # fix IMU intrinsics (Dw/Da) at identity

    # --- sweep spec (drives main's control flow) ---
    n_seeds: int = 5
    td_list_ms: list = field(default_factory=lambda: [0.0, 1, 2, 5, 10, 20])
    td_sweep: bool = False        # sweep td_list_ms (single state rate)
    state_hz_list: list = field(default_factory=list)  # non-empty => state_hz x td grid

    # cached / derived
    cache: str = "/tmp/mvis_aruco16_detcache"

    # ----- construction from a shared cam-IMU run config -----
    @classmethod
    def from_cam_imu_config(cls, cfg, **overrides):
        """Build a SimStudyConfig that SHARES a ``CamImuRunConfig``'s dataset pointing — bag
        (``input.path``), camchain (``camera_input.path``), and the first cam/imu topic — instead
        of re-declaring it. So one config drives calibrate -> roundtrip -> repeatability ->
        simulate. Sim-only knobs (sfm_mode, rates, noise, window, recovery) keep their defaults
        unless passed in ``overrides``. Expects a ROS1 config (the sim loads a bag)."""
        base = {"bag": cfg.input.path}
        ci = getattr(cfg, "camera_input", None)
        if ci is not None and getattr(ci, "path", None):
            base["camchain"] = ci.path
        if cfg.input.cam_topics:
            base["cam_topic"] = cfg.input.cam_topics[0]
        if cfg.input.imu_topics:
            base["imu_topic"] = cfg.input.imu_topics[0]
        base.update(overrides)
        return cls(**base)

    # ----- construction from the legacy env vars (CLI back-compat) -----
    @classmethod
    def from_env(cls, n_seeds=None):
        """Build from the legacy `DETECTOR`/`T0`/`STATEHZ`/… env vars the study
        scripts were driven by, so those invocations keep working. `DETECTOR=april`
        switches to an AprilGrid bag (`BAGNUM`) with planar-homography SfM; anything
        else takes the ArUco defaults. Explicit `n_seeds` overrides `n_seeds=5`."""
        g = os.environ.get
        det = g("DETECTOR", "aruco")
        ds = {}
        if det == "april":                    # AprilGrid corners fed as features
            num = g("BAGNUM", "01")
            ds = dict(detector="april",
                      bag=os.path.join(_DS, f"virig_4imus_3cams_april_{num}.bag"),
                      t0=float(g("T0", "5")), window=float(g("WINDOW", "45")),
                      sfm_mode=g("SFM", "planar"))   # planar board -> homography SfM
        else:
            ds = dict(t0=float(g("T0", "78")), window=float(g("WINDOW", "44")),
                      sfm_mode=g("SFM", "general"))
        return cls(
            **ds,
            ground_hz=float(g("GROUNDHZ", "8.0")),
            ground_pin_intr=(g("GPININTR", "1") != "0"),
            knot_dt=float(g("KNOTDT", "0")),
            cam_hz=float(g("CAMHZ", "20")),
            imu_hz=float(g("IMUHZ", "400")),
            td_inject=(0.0 if g("TDOFF", "0") == "1" else None),
            state_hz=float(g("STATEHZ", "8.0")),
            imu_model=g("IMUMODEL", "calyx"),
            pin_intr=g("PININTR", "0") == "1",
            pin_td=g("PINTD", "0") == "1",
            pin_imu=g("PINIMU", "0") == "1",
            n_seeds=(n_seeds if n_seeds is not None else 5),
            td_list_ms=[float(x) for x in g("TDLIST", "0,1,2,5,10,20").split(",")],
            td_sweep=g("TDSWEEP", "0") == "1",
            state_hz_list=[float(x) for x in g("STATEHZLIST", "").split(",") if x],
        )

    # ----- rate-coupling validation (warn on the footguns we hit) -----
    def validate(self):
        """Check the three rate couplings that have silently corrupted studies before,
        printing a `[config WARN]` line per problem and returning True when clean:
        grounding above the real camera rate (it cannot go faster than the frames),
        `cam_hz` not an integer multiple of a target `state_hz` (states alias to a
        LOWER effective rate than requested), and `imu_hz` under 10x the max state
        rate (preintegration factors go sample-starved). Warns only — never raises,
        so a deliberate configuration can still run."""
        warns = []
        if self.ground_hz > self.real_cam_hz + 1e-6:
            warns.append(f"ground_hz {self.ground_hz} > real cam {self.real_cam_hz}Hz "
                         "(grounding uses real frames — capped at the real rate)")
        targets = [self.state_hz] + list(self.state_hz_list)
        for hz in targets:
            ratio = self.cam_hz / hz
            if abs(ratio - round(ratio)) > 1e-6:
                warns.append(f"cam_hz {self.cam_hz} is NOT an integer multiple of "
                             f"state_hz {hz} (ratio {ratio:.2f}) — states will ALIAS "
                             f"to cam_hz/round(ratio) = {self.cam_hz/round(ratio):.1f}Hz")
        max_state = max(targets)
        if self.imu_hz < 10 * max_state:
            warns.append(f"imu_hz {self.imu_hz} < 10x max state_hz {max_state} "
                         "(IMU preintegration factors may be sample-starved)")
        for w in warns:
            print(f"  [config WARN] {w}", flush=True)
        return not warns

    def calib_cfg(self, pin_td=False, state_hz=None):
        """Build the pipeline CamImuCalibConfig for one solve (shared helper)."""
        noise = ImuNoiseDensities(self.sigma_w, self.sigma_bg, self.sigma_a, self.sigma_ba)
        return matched_calib_cfg(
            noise, imu_model=self.imu_model, sigma_px=self.sigma_px,
            imu_intrinsics=("pinned" if self.pin_imu else "dw_da"),
            state_hz=(self.state_hz if state_hz is None else state_hz),
            z_min_frac=0.6, sigma_td_prior=(1e-6 if pin_td else 0.0),
            init_kwargs={"scale_rel_tol": 0.6, "keyframe_hz": 2.0})
