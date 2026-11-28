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
"""CalibrationResult -> EstimateStates: the writer that
lets a run's solved values land in the db, so reports can read load(path,
run_id) instead of *_full.yaml.

Pure mapping, no sqlite: build the EstimateStates here, set db.run /
db.estimate_states, and let DatabaseManager.save() persist. Conventions:
result.board_poses hold T_board_cam (X_cam = T . X_board, output_writer's
full-YAML key); CamOnlyBaseState wants T_cam_to_board (p_board = T . p_cam,
mvisdatabase.py:195), so each pose is INVERTED here. Frames whose pose is
None (diverged / unsolved) are skipped, matching write_full_yaml.
"""
from __future__ import annotations

import numpy as np

from mvis.mvisdb.mvisdatabase import (
    CalibProvenance, CameraIntrinsicsState, CameraModel, CamOnlyBaseState,
    EstimateStates, RelativeCalib, ReprojStats, SensorState, SensorType,
)

__all__ = ["estimates_from_calibration_result", "calibration_result_from_estimates",
           "estimates_from_cam_imu_result",
           "mat_to_pose7", "pose7_to_mat"]


def mat_to_pose7(T: np.ndarray) -> np.ndarray:
    """4x4 SE(3) -> (7,) [qw qx qy qz tx ty tz] (state-blob convention)."""
    T = np.asarray(T, np.float64)
    R = T[:3, :3]
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(1.0 + R[i, i] - R[j, j] - R[k, k], 0.0)) * 2
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q /= np.linalg.norm(q)
    if q[0] < 0:                       # canonical sign: qw >= 0
        q = -q
    return np.concatenate([q, T[:3, 3]])


def _invert_se3(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, np.float64)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def estimates_from_calibration_result(result) -> EstimateStates:
    """Map a solved cam-only CalibrationResult into the EstimateStates shape.

    Every value is marked CalibProvenance.ESTIMATED — this mapper exists for
    solver output only; priors/pins never come through here. sigmas stay None
    until the optimizer exports marginals (open item).
    """
    ref = result.reference_camera
    es = EstimateStates(reference_stream=(SensorType.CAM, ref))

    rel_by_cam = {}
    for e in result.extrinsics:
        if e.to_camera != ref:
            continue
        rel_by_cam[e.from_camera] = RelativeCalib(
            T_x_to_ref=np.asarray(e.T_to_from, np.float64),
            td_x_to_ref_s=result.time_offsets.get(e.from_camera),
            provenance=CalibProvenance.ESTIMATED)

    err_by_cam = {e.camera_id: e for e in result.reprojection_errors}
    for cam in result.cameras:
        cid = cam.camera_id
        err = err_by_cam.get(cid)
        es.sensor_states.append(SensorState(
            sensor=(SensorType.CAM, cid),
            intrinsics=CameraIntrinsicsState(
                intrinsics=np.asarray(cam.intrinsics, np.float64),
                distortion=np.asarray(cam.distortion, np.float64),
                camera_model=CameraModel(cam.distortion_model),
                readout_s=cam.rolling_shutter_readout or None,
                provenance=CalibProvenance.ESTIMATED),
            relative=None if cid == ref else rel_by_cam.get(cid),
            reproj=(ReprojStats(err.rms, err.mean, err.median, err.p90,
                                err.p95, err.num_observations)
                    if err is not None else None)))
    if result.overall_rms:
        es.overall_rms = float(result.overall_rms)

    ts, poses7 = [], []
    for i, T in enumerate(result.board_poses):
        if T is None:
            continue
        M = T.matrix() if hasattr(T, "matrix") else np.asarray(T, np.float64)
        ts.append(result.board_pose_times[i]
                  if i < len(result.board_pose_times) else 0.0)
        poses7.append(mat_to_pose7(_invert_se3(M)))
    if ts:
        es.base_state = CamOnlyBaseState(
            t=np.asarray(ts, np.float64),
            T_cam_to_board=np.asarray(poses7, np.float64))
    return es


def pose7_to_mat(p: np.ndarray) -> np.ndarray:
    """(7,) [qw qx qy qz tx ty tz] -> 4x4 SE(3). Inverse of mat_to_pose7."""
    qw, qx, qy, qz, tx, ty, tz = np.asarray(p, np.float64)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])
    T[:3, 3] = [tx, ty, tz]
    return T


def _so3_log(R: np.ndarray) -> np.ndarray:
    """gtsam-backed log, imported here so mvisdb stays importable without gtsam at arrays-only sites."""
    from mvis.calibration.factors.lie_groups import log_so3
    return log_so3(R)


def calibration_result_from_estimates(es: EstimateStates,
                                      image_sizes: dict | None = None):
    """EstimateStates -> CalibrationResult: the read-side inverse
    of estimates_from_calibration_result, so reports can render from
    load(path, run_id) without the *_full.yaml. board_poses come back in the
    report's T_board_cam convention (stored T_cam_to_board re-inverted).
    image_size is not part of the estimate: pass ``image_sizes`` (camera_id ->
    (w, h), e.g. DatabaseManager.stream_resolutions) to restore it from the
    capture's streams — kb4's valid_domain needs it; cameras without an entry
    keep the (0, 0) default.
    Lazy core import keeps mvisdb importable without mvis.core at arrays-only
    call sites, same rule as bridge.py."""
    from mvis.core.types import (
        CalibrationResult, CameraParams, ExtrinsicTransform,
    )
    ref = es.reference_stream[1]
    cameras, extrinsics, time_offsets = [], [], {}
    for ss in sorted(es.sensor_states, key=lambda s: s.sensor[1]):
        if ss.sensor[0] is not SensorType.CAM:
            continue
        cid = ss.sensor[1]
        i = ss.intrinsics
        sz = (image_sizes or {}).get(cid)
        cameras.append(CameraParams(
            camera_id=cid, intrinsics=i.intrinsics, distortion=i.distortion,
            distortion_model=i.camera_model.value,
            image_size=((int(sz[0]), int(sz[1])) if sz is not None
                        else (0, 0)),
            rolling_shutter_readout=i.readout_s or 0.0))
        if ss.relative is not None:
            T = np.asarray(ss.relative.T_x_to_ref, np.float64)
            extrinsics.append(ExtrinsicTransform(
                from_camera=cid, to_camera=ref, T_to_from=T,
                so3=_so3_log(T[:3, :3]), translation=T[:3, 3].copy()))
            if ss.relative.td_x_to_ref_s is not None:
                time_offsets[cid] = ss.relative.td_x_to_ref_s
    result = CalibrationResult(cameras=cameras, extrinsics=extrinsics,
                               reference_camera=ref, time_offsets=time_offsets)
    from mvis.core.types import ReprojectionErrorStats
    for ss in sorted(es.sensor_states, key=lambda s: s.sensor[1]):
        if ss.sensor[0] is SensorType.CAM and ss.reproj is not None:
            rp = ss.reproj
            result.reprojection_errors.append(ReprojectionErrorStats(
                camera_id=ss.sensor[1], rms=rp.rms or 0.0, mean=rp.mean or 0.0,
                median=rp.median or 0.0, p90=rp.p90 or 0.0, p95=rp.p95 or 0.0,
                num_observations=rp.n_obs or 0))
    if es.overall_rms is not None:
        result.overall_rms = es.overall_rms
    bs = es.base_state
    if isinstance(bs, CamOnlyBaseState):
        result.board_pose_times = [float(t) for t in bs.t]
        result.board_poses = [_invert_se3(pose7_to_mat(p))
                              for p in bs.T_cam_to_board]
    return result


def estimates_from_cam_imu_result(result, *, state_hz: float | None = None) -> EstimateStates:
    """Map a solved cam-IMU CamImuCalibResult into the EstimateStates shape.

    reference_stream = (IMU, 0). Sensor states: base camera (index 0
    of result.cameras) and extras (T_CtoC0 composed onto the base's T_CtoI so
    every camera's relative is w.r.t. the reference IMU), base IMU xi24 with
    marginal sigmas, extra IMUs from result.imus. world_frame is the
    gravity-aligned gauge. base_state (trajectory + bias knots) is DEFERRED:
    the result carries endpoint biases + drift, not per-knot arrays, and
    fabricating knots from a linear model would misrepresent solver output.
    """
    from mvis.core.gravity import of_result
    from mvis.mvisdb.mvisdatabase import (
        ImuIntrinsicsModel, ImuIntrinsicsState, WorldFrame,
    )
    es = EstimateStates(reference_stream=(SensorType.IMU, 0),
                        state_hz=state_hz,
                        world_frame=WorldFrame(g_magnitude=of_result(result)))

    # base_state: per-knot trajectory + velocity + biases, extracted EXACTLY
    # from the converged Values (X(k) pose, _Vk(k) = [v, b_g, b_a] 9-vector —
    # verified against build_joint_graph's key scheme). Skipped when the
    # result carries no values (e.g. a stub in tests).
    values = getattr(result, "values", None)
    t_states = getattr(result, "t_states", None)
    if values is not None and t_states is not None and len(t_states):
        import gtsam
        _X = gtsam.symbol_shorthand.X
        _Vk = lambda k: gtsam.symbol('v', k)
        K = len(t_states)
        T7 = np.empty((K, 7)); V = np.empty((K, 3))
        BG = np.empty((K, 3)); BA = np.empty((K, 3))
        for k in range(K):
            T7[k] = mat_to_pose7(values.atPose3(_X(k)).matrix())
            vb = np.asarray(values.atVector(_Vk(k)), np.float64)
            V[k], BG[k], BA[k] = vb[:3], vb[3:6], vb[6:9]
        from mvis.mvisdb.mvisdatabase import CamImuBaseState
        es.base_state = CamImuBaseState(
            t=np.asarray(t_states, np.float64),
            T_imu_to_world=T7, v=V, b_g=BG, b_a=BA)

    # base IMU
    es.sensor_states.append(SensorState(
        sensor=(SensorType.IMU, 0),
        intrinsics=ImuIntrinsicsState(
            xi24=np.asarray(result.imu_intr_xi, np.float64),
            model=ImuIntrinsicsModel(result.imu_model),
            provenance=CalibProvenance.ESTIMATED,
            sigmas=(np.asarray(result.imu_intr_sigma, np.float64)
                    if result.imu_intr_sigma is not None else None))))

    # extra IMUs (relative to the reference IMU)
    for i, entry in enumerate(result.imus, start=1):
        T = entry["T_IstoI"]
        M = T.matrix() if hasattr(T, "matrix") else np.asarray(T, np.float64)
        es.sensor_states.append(SensorState(
            sensor=(SensorType.IMU, i),
            intrinsics=ImuIntrinsicsState(
                xi24=np.zeros(24), model=ImuIntrinsicsModel(result.imu_model),
                provenance=CalibProvenance.PRIOR_FIXED),
            relative=RelativeCalib(
                T_x_to_ref=M, td_x_to_ref_s=float(entry.get("td", 0.0)),
                provenance=CalibProvenance.ESTIMATED)))

    # cameras: every entry's calib is a full solved bundle whose T IS
    # T_CtoI (T_CtoC0 in the result is derived from these for reporting;
    # verified at pipeline_cam_imu.py:841 — T_rel = sols[0].T^-1 . sols[c].T)
    for cid, entry in enumerate(result.cameras):
        calib = entry["calib"]
        intr = np.asarray(calib.intr, np.float64)
        M = calib.T.matrix()
        es.sensor_states.append(SensorState(
            sensor=(SensorType.CAM, cid),
            intrinsics=CameraIntrinsicsState(
                intrinsics=intr[:4], distortion=intr[4:],
                camera_model=CameraModel(calib.model),
                readout_s=calib.tr or None,
                provenance=CalibProvenance.ESTIMATED,
                sigmas=(np.asarray(entry.get("calib_sigma"), np.float64)
                        if entry.get("calib_sigma") is not None else None)),
            relative=RelativeCalib(
                T_x_to_ref=M, td_x_to_ref_s=float(calib.td),
                provenance=CalibProvenance.ESTIMATED)))
    return es
