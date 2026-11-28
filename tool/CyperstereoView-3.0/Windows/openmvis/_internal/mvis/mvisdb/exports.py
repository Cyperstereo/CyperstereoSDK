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
"""Export views: deployable artifacts rendered FROM the database.

The calib YAML (and later the Kalibr camchain) stop being things only the
solve can write: any solved run renders them via load-side mapping —
state(run_id) -> calibration_result_from_estimates -> the SAME OutputWriter
serialization the pipeline uses, so the export is guarded by construction
against drifting from the writer.

Known-lossy fields (documented, not silently wrong): per-camera `max` error
is not part of the persisted estimate and renders as 0. CameraParams.image_size
is not in the estimate either, but the capture's camera table stores each
stream's resolution (res_w/res_h), so the calib view restores it from there —
kb4's `valid_domain` (corner_radius_px, covers_image) is computed from it.
Only a capture without a stored resolution renders image_size as 0 (and those
two valid_domain fields as null); `normalise_calib_yaml_pair` documents how the
comparison treats that case.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["export_calib_yaml", "export_kalibr_camchain",
           "normalise_calib_yaml_pair"]


def export_kalibr_camchain(manager, db_path, run_id: int, out_path) -> None:
    """Render the Kalibr camchain-imucam FROM a solved cam-IMU run.

    Rebuilds the serializer's expected entries (CameraCalib + T_CtoC0 +
    image_size) from state(run_id) and the capture's stream resolutions,
    then uses the SAME to_kalibr_camchain_imucam the pipeline writer uses.
    T_CtoC0 is recomposed from the stored T_CtoI matrices, so downstream
    T_cn_cnm1 matches the writer to float-recomposition precision (~1e-12),
    not byte-for-byte — the shadow compares it with a tolerance."""
    import gtsam
    import numpy as np
    import yaml
    from types import SimpleNamespace
    from mvis.calibration.factors.camera_calib import CameraCalib
    from mvis.io.cam_imu_output import to_kalibr_camchain_imucam
    from mvis.mvisdb.mvisdatabase import SensorType
    es = manager.state(db_path, run_id)
    db = manager.load(db_path)
    res_by_cam = {s.camera_id: s.info.resolution for s in db.streams}
    cam_states = sorted((s for s in es.sensor_states
                         if s.sensor[0] is SensorType.CAM),
                        key=lambda s: s.sensor[1])
    cams, T0 = [], None
    for ss in cam_states:
        i = ss.intrinsics
        T = np.asarray(ss.relative.T_x_to_ref, np.float64)
        cal = CameraCalib(
            gtsam.Pose3(T), ss.relative.td_x_to_ref_s or 0.0,
            i.readout_s or 0.0,
            np.concatenate([i.intrinsics, i.distortion]),
            model=i.camera_model.value)
        entry = {"calib": cal}
        res = res_by_cam.get(ss.sensor[1])
        if res is not None:
            entry["image_size"] = tuple(res)
        if T0 is None:
            T0 = gtsam.Pose3(T)
        else:
            entry["T_CtoC0"] = T0.inverse().compose(gtsam.Pose3(T))
        cams.append(entry)
    # build first: a non-exportable model raises before the file opens
    chain = to_kalibr_camchain_imucam(SimpleNamespace(cameras=cams))
    with open(str(Path(out_path)), "w") as f:
        yaml.safe_dump(chain, f, sort_keys=False)


def export_calib_yaml(manager, db_path, run_id: int, out_path) -> None:
    """Render `<run>_calib.yaml` from a solved run in the capture db."""
    from mvis.io.output_writer import OutputWriter
    from mvis.mvisdb.estimates import calibration_result_from_estimates
    es = manager.state(db_path, run_id)
    result = calibration_result_from_estimates(
        es, image_sizes=manager.stream_resolutions(db_path))
    OutputWriter.write_calibration_yaml(result, str(Path(out_path)))


# valid_domain fields derived from image_size (mvis.camera.kb_domain.kb_valid_domain)
_IMAGE_DERIVED_DOMAIN_KEYS = ("corner_radius_px", "covers_image")
# writer-only diagnostics (not estimate states): dropped on both sides
_DIAGNOSTIC_TOP_KEYS = ("board_shape", "board_point_bias", "corner_accounting")


def normalise_calib_yaml_pair(writer: dict, view: dict) -> None:
    """Normalisation, in place, of a (writer calib YAML, export view)
    pair loaded with yaml.safe_load, so they compare with ``==``.

    Drops the documented lossy / derived fields on both sides: per-camera
    reprojection `max` (not persisted), extrinsic `so3` (derived from
    T_to_from, which is compared exactly; the view's log map matches only to
    ~1e-13), `image_size`, and the writer-only diagnostics
    `rolling_shutter_readout_sigma` (per camera), `board_shape`,
    `board_point_bias` and `corner_accounting` (the estimate store holds parameters, not sigmas or
    board diagnostics). The view restores image_size from the capture's
    stream resolution, so kb4's image-derived valid_domain fields are compared
    strictly; only for a camera whose view image_size is absent or 0 (the
    capture stored no resolution) are corner_radius_px / covers_image dropped
    too, since the view cannot compute them."""
    wc, vc = writer.get("cameras", []), view.get("cameras", [])
    for cw, cv in zip(wc, vc):
        sz = cv.get("image_size")
        if not sz or min(int(x) for x in sz) <= 0:
            for c in (cw, cv):
                vd = c.get("valid_domain")
                if isinstance(vd, dict):
                    for k in _IMAGE_DERIVED_DOMAIN_KEYS:
                        vd.pop(k, None)
    for d in (writer, view):
        # solve diagnostics the estimate store does not hold (formal readout
        # sigma from the marginals; board-shape estimate and per-corner bias)
        for k in _DIAGNOSTIC_TOP_KEYS:
            d.pop(k, None)
        for cam in d.get("cameras", []):
            cam.pop("image_size", None)
            cam.pop("rolling_shutter_readout_sigma", None)
        for e in d.get("reprojection_errors", {}).get("per_camera", []):
            for k in ("max", "std", "inlier"):   # writer-only statistics, not in the store
                e.pop(k, None)
        for tr in d.get("extrinsics", {}).get("transforms", []):
            tr.pop("so3", None)
