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
"""Build a pinned cam-IMU camera prior from several camera-only solves of the same rig.

Inputs and output are camera-only result dicts (`*_calib.yaml`): `median_prior` takes the
parsed YAMLs of several `mvis cam` solves of one rig and returns the per-camera median
intrinsics / distortion and the chordal-mean extrinsics, to be written as the pinned
camera prior of a cam-IMU config (the `cameras[].path` intrinsics file with
`free_intrinsics: false`); `per_camera_prior` does the same per camera group.
"""
from __future__ import annotations

import statistics as st

import numpy as np
from scipy.spatial.transform import Rotation


def _chordal_mean(Rs):
    """Chordal mean rotation. Not mvis.recalib.result_metrics.mean_rot: that one averages before the
    SVD and differs in the last bits (5e-15 at most), which would change a prior's bytes."""
    U, _, Vt = np.linalg.svd(sum(Rs))
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def median_prior(results: list[dict], source: str) -> dict:
    """Per camera the median of every intrinsic and distortion coefficient, per extrinsic the chordal
    mean rotation and median translation, per camera time offset the median. Reprojection errors are
    not carried over (they describe one solve)."""
    ref = results[0]
    cams = []
    for i, c0 in enumerate(ref["cameras"]):
        cs = [r["cameras"][i] for r in results]
        assert all(c["id"] == c0["id"] and c["model"] == c0["model"] for c in cs), "camera sets differ"
        cams.append({
            "id": c0["id"], "model": c0["model"],
            "intrinsics": {k: float(st.median(c["intrinsics"][k] for c in cs)) for k in ("fx", "fy", "cx", "cy")},
            "distortion": {"model": c0["distortion"]["model"],
                           "coeffs": [float(st.median(c["distortion"]["coeffs"][j] for c in cs))
                                      for j in range(len(c0["distortion"]["coeffs"]))]},
            "image_size": list(c0["image_size"]),
            "rolling_shutter_readout": float(st.median(c.get("rolling_shutter_readout") or 0.0 for c in cs)),
        })
    transforms = []
    for k, t0 in enumerate(ref["extrinsics"]["transforms"]):
        Ts = [np.array(r["extrinsics"]["transforms"][k]["T_to_from"]) for r in results]
        assert all(r["extrinsics"]["transforms"][k]["from"] == t0["from"] for r in results)
        R = _chordal_mean([T[:3, :3] for T in Ts])
        p = np.median(np.array([T[:3, 3] for T in Ts]), axis=0)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        transforms.append({"from": t0["from"], "to": t0["to"], "T_to_from": T.tolist(),
                           "so3": Rotation.from_matrix(R).as_rotvec().tolist(), "translation": p.tolist()})
    out = {"source": source, "cameras": cams,
           "extrinsics": {"reference_camera": ref["extrinsics"]["reference_camera"], "transforms": transforms}}
    if ref.get("time_offsets"):
        offs = []
        for k, o0 in enumerate(ref["time_offsets"]["offsets"]):
            offs.append({"camera": o0["camera"],
                         "dt": float(st.median(r["time_offsets"]["offsets"][k]["dt"] for r in results))})
        out["time_offsets"] = {"reference_camera": ref["time_offsets"]["reference_camera"], "offsets": offs}
    return out


def per_camera_prior(seed: dict, groups: dict[int, list[dict]], source: str) -> dict:
    """Seed calibration with camera k replaced by the median of its single-camera solves."""
    out = {"source": source, **{k: v for k, v in seed.items() if k not in ("source", "reprojection_errors")}}
    out["cameras"] = [dict(c) for c in seed["cameras"]]
    for k, results in groups.items():
        med = median_prior([{"cameras": r["cameras"], "extrinsics": {"reference_camera": 0, "transforms": []}}
                            for r in results], "")["cameras"][0]
        assert med["model"] == out["cameras"][k]["model"], f"camera {k}: model differs from the seed"
        med["id"] = out["cameras"][k]["id"]
        out["cameras"][k] = med
    return out
