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
"""Build a ground-truth simulation scenario from a real CamImuCalibResult:
gtsam cal objects, a B-spline trajectory through the solved states, and the
extracted GT parameter set (Task 2)."""
import json
from mvis.core.gravity import of_result
import numpy as np
import gtsam
from dataclasses import dataclass, asdict

from mvis.core.camera_models import GTSAM, check_model, legacy_model
from mvis.simulation.bspline_se3 import (
    BsplineSE3, _log_se3, _exp_se3, _inv_se3)


def gt_cam_model(cam: dict) -> str:
    """Distortion model of a GT cam dict: its 'model' key, else (GT JSON written
    before the key existed) the legacy `is_fisheye` flag, which raises unless
    intr8 is 8 long."""
    m = cam.get('model')
    if m:
        return check_model(str(m))
    return legacy_model(bool(cam.get('is_fisheye', False)), cam.get('intr8'))


class ModelCal:
    """A non-gtsam `CameraModel` (fov/ds/eucm/ucm/omni_radtan/kb4) behind the gtsam
    Cal3 `uncalibrate(xy)` API the simulator projects through (`xy` = the
    normalized point X/Z, Y/Z of a z > 0 camera point). Out-of-model-range
    points raise RuntimeError, like gtsam's cheirality, so the simulator drops
    them.

    Limitation: project_points gates z > z_near before calling uncalibrate, so a
    custom-model GT is simulated over the front hemisphere only (no rays past
    ~87 deg off-axis), even when the model's valid domain reaches beyond 90 deg."""

    def __init__(self, model: str, cal_vec):
        from mvis.camera.camera_model import build_camera_model   # lazy: heavy import
        self.model = model
        self._m = build_camera_model(model, np.asarray(cal_vec, float))

    def uncalibrate(self, xy):
        uv = self._m.project(np.array([float(xy[0]), float(xy[1]), 1.0]))
        if not np.isfinite(uv).all():
            raise RuntimeError(f"{self.model} projection out of model range")
        return np.asarray(uv, float)


def cal_from_vec(cal_vec, is_fisheye, model=None):
    """Simulator cal (anything with `uncalibrate(xy)`) from a cal_vec.
    radtan: [fx,fy,cx,cy,k1,k2,p1,p2] -> Cal3DS2(fx,fy,s=0,cx,cy,k1,k2,p1,p2).
    equidistant: [fx,fy,cx,cy,k1,k2,k3,k4] -> Cal3Fisheye(...). Any other
    `model` -> a `ModelCal` projecting through its CameraModel. `model`
    defaults from `is_fisheye`, which needs an 8-vector (a longer one is a
    custom model that must be named — raise, never simulate it as radtan)."""
    v = [float(x) for x in cal_vec]
    model = check_model(model or legacy_model(is_fisheye, v))
    if model not in GTSAM:
        return ModelCal(model, v)
    if len(v) != 8:
        raise ValueError(f"cal_from_vec: gtsam {model} needs an 8-vector, got {len(v)}")
    fx, fy, cx, cy = v[0], v[1], v[2], v[3]
    d = v[4:]
    if model == "equidistant":
        return gtsam.Cal3Fisheye(fx, fy, 0.0, cx, cy, d[0], d[1], d[2], d[3])
    return gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, d[0], d[1], d[2], d[3])


def fit_bspline_from_states(t_states, T_states, dt):
    """Fit a BsplineSE3 through discrete T_ItoG (4x4) poses by sampling a
    piecewise-geodesic interpolation of the states at spacing `dt`. Returns
    (spline, max_pose_err_m) where the error is the max position gap between the
    fitted spline and the input states over the interior (fit-quality gate)."""
    ts = np.asarray(t_states, float)
    Ts = [np.asarray(T, float) for T in T_states]
    order = np.argsort(ts)
    ts = ts[order]
    Ts = [Ts[i] for i in order]

    def pose_fn(t):
        """Pose at `t` by SE(3) geodesic interpolation between the bracketing
        states, clamped to the end poses outside `[ts[0], ts[-1]]`."""
        if t <= ts[0]:
            return Ts[0]
        if t >= ts[-1]:
            return Ts[-1]
        j = int(np.searchsorted(ts, t, side="right"))   # ts[j-1] <= t < ts[j]
        T0, T1 = Ts[j - 1], Ts[j]
        alpha = (t - ts[j - 1]) / (ts[j] - ts[j - 1])
        return T0 @ _exp_se3(alpha * _log_se3(_inv_se3(T0) @ T1))

    spline = BsplineSE3.from_pose_fn(pose_fn, float(ts[0]), float(ts[-1]), float(dt))
    # fit-quality: position error at the interior states (skip the first/last
    # knot where the spline is only padded-queryable)
    errs = []
    for t, T in zip(ts[1:-1], Ts[1:-1]):
        _R, p = spline.get_pose(float(t))
        errs.append(float(np.linalg.norm(p - T[:3, 3])))
    max_err = max(errs) if errs else 0.0
    return spline, max_err


@dataclass
class GroundTruth:
    """A complete rig + trajectory ground truth, in plain JSON-serializable types.

    Deliberately holds lists and 4x4 nested lists rather than gtsam objects so it
    can round-trip through `gt_to_json`/`gt_from_json` and be handed to a later
    simulation run. This is what makes the digital twin possible: a real
    `CamImuCalibResult` becomes the GT that a synthetic dataset is generated from,
    so the recovery can be scored against the rig that actually produced it.
    """
    cams: list            # [{intr8, T_CtoI(4x4), td, is_fisheye, model, image_size}]
    imus: list            # [{name, is_base, xi24, T_IstoI(4x4)|None, td}]
    b_g: list
    b_a: list
    T_BinW: list          # 4x4
    t_states: list
    T_states: list        # [4x4, ...]
    imu_model: str
    g_inW: list
    board_points_local: object = None   # array, set by the caller for `synth.build_synth`; not in the JSON


def _P(pose):             # gtsam.Pose3 -> 4x4 list
    """gtsam.Pose3 -> a plain nested 4x4 list (JSON-serializable, see `GroundTruth`)."""
    return np.asarray(pose.matrix(), float).tolist()


def capture_gt(result):
    """Extract a self-consistent GT set from one real CamImuCalibResult."""
    X = gtsam.symbol_shorthand.X
    cams = []
    for e in result.cameras:
        cal = e["calib"]
        cams.append(dict(
            intr8=[float(v) for v in np.asarray(cal.intr, float)],
            T_CtoI=_P(cal.T), td=float(cal.td),
            is_fisheye=bool(cal.is_fisheye), model=str(cal.model),
            image_size=list(e["image_size"])))
    imus = [dict(name="base", is_base=True,
                 xi24=[float(v) for v in np.asarray(result.imu_intr_xi, float)],
                 T_IstoI=None, td=0.0)]
    for m in (result.imus or []):
        imus.append(dict(
            name=m.get("name", "aux"), is_base=False,
            xi24=[float(v) for v in np.asarray(m["intr_xi24"], float)],
            T_IstoI=_P(m["T_IstoI"]), td=float(m["td"])))
    ts = np.asarray(result.t_states, float)
    T_states = [_P(result.values.atPose3(X(k))) for k in range(len(ts))]
    return GroundTruth(
        cams=cams, imus=imus,
        b_g=[float(v) for v in np.asarray(result.b_g, float)],
        b_a=[float(v) for v in np.asarray(result.b_a, float)],
        T_BinW=_P(result.T_BinW),
        t_states=[float(t) for t in ts], T_states=T_states,
        imu_model=str(getattr(result, "imu_model", "calyx")),
        g_inW=[0.0, 0.0, -of_result(result)])


def gt_to_json(gt, path):
    """Write a `GroundTruth` to `path` as JSON (all but `board_points_local` are plain types)."""
    d = asdict(gt)
    del d["board_points_local"]
    with open(path, "w") as f:
        json.dump(d, f)


def gt_from_json(path):
    """Read back a `GroundTruth` written by `gt_to_json`."""
    with open(path) as f:
        return GroundTruth(**json.load(f))


def ground_truth_from_result(result, image_sizes) -> "GroundTruth":
    """Capture a `CamImuCalibResult` as a `GroundTruth` for digital-twin simulation:
    the recovered IMU-world trajectory (`T_ItoW` states), board pose, EVERY camera
    ({intr8, T_CtoI, td, readout, is_fisheye, model, image_size}), and base + aux IMUs
    ({xi24, T_IstoI, td}). `image_sizes` is a per-camera (w, h) list (the result does
    not store it). World is gravity-aligned, g_inW = [0, 0, -|g|] with the solve's |g| (9.81 by default)."""
    from mvis.calibration.pipeline_cam_imu import X

    t_states = np.asarray(result.t_states, float)
    T_states = [result.values.atPose3(X(k)).matrix().tolist() for k in range(len(t_states))]
    cams = []
    for c, e in enumerate(result.cameras):
        cal = e["calib"]
        cams.append(dict(intr8=np.asarray(cal.intr, float).tolist(),
                         T_CtoI=cal.T.matrix().tolist(), td=float(cal.td),
                         readout=float(cal.tr), is_fisheye=bool(cal.is_fisheye),
                         model=str(cal.model), image_size=list(image_sizes[c])))
    imus = [dict(name="imu0", is_base=True,
                 xi24=np.asarray(result.imu_intr_xi, float).tolist(), T_IstoI=None, td=0.0)]
    for m in (getattr(result, "imus", None) or []):
        T = m.get("T_IstoI")
        imus.append(dict(name=m.get("name", "aux"), is_base=False,
                         xi24=np.asarray(m.get("intr_xi24", np.zeros(24)), float).tolist(),
                         T_IstoI=(T.matrix().tolist() if T is not None else None),
                         td=float(m.get("td", 0.0))))
    return GroundTruth(
        cams=cams, imus=imus,
        b_g=np.asarray(result.b_g, float).tolist(), b_a=np.asarray(result.b_a, float).tolist(),
        T_BinW=result.T_BinW.matrix().tolist(),
        t_states=t_states.tolist(), T_states=T_states,
        imu_model=str(getattr(result, "imu_model", "calyx")),
        g_inW=[0.0, 0.0, -of_result(result)])
