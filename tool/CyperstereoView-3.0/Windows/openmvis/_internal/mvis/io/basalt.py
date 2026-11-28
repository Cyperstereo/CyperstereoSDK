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
"""Read Basalt calibration outputs and convert them to MVIS conventions (no solving).

Basalt's `basalt_calibrate` / `basalt_calibrate_imu` write calibration.json (cereal,
{"value0": {...}}); the comparison described in doc/comparison_basalt.md also keeps a
basalt_status.json per run next to it. `camera_only` / `camera_imu` read one run directory
and return MVIS-convention dicts for a side-by-side with an MVIS solve. Conventions, checked
against the Basalt sources (spline_linearize.h, basalt-headers calib_bias.hpp) and on TUM-VI imu1:

  time offset   cam_time_offset_ns: the image stamped t_cam is evaluated at the IMU (spline) pose
                t_cam + offset. MVIS camera td: the image stamped t_cam sees the IMU pose at
                t_cam + td. Same sign: td_ms = offset * 1e-6. One offset for all cameras.
  extrinsics    T_imu_cam[i] = T_i_c (camera pose in the IMU frame; Hamilton qx qy qz qw, metres),
                so the camera centre in the IMU frame is its translation (MVIS p_CinI) and the
                rotation is MVIS's cam rotvec log(R_i_c). In camera-only runs the "IMU" is cam0.
  intrinsics    fx fy cx cy first for every model; Basalt kb4 k1..k4 = MVIS dist[0..3] of kb4 and of equidistant alike
                (one polynomial; the two agree below 90 deg, and only kb4 goes past it);
                pinhole-radtan8 k1 k2 p1 p2 k3 k4 k5 k6 (+ rpmax, kept apart).
  gyro          calib_gyro_bias = [b(3), S(9) column-major]: w = (I + S_g) w_m - b. MVIS:
                w_m = Dw^-1 R^T w, i.e. w = R Dw w_m. Basalt's I + S_g is a full 3x3 with no
                separate rotation = the MVIS CALYX Dw; as the KALIBR variant it splits into
                R (orthogonal) times Dw (lower-triangular).
  accel         calib_accel_bias = [b(3), S(6) lower-triangular]: a = (I + S_a) a_m - b, so
                Da = I + S_a, lower-triangular = the MVIS KALIBR Da (calyx Da is upper-triangular;
                the two pin different IMU frames, which is why Basalt's R_i_c agrees with the MVIS
                kalibr variant to ~0.003 rad but with calyx only to ~0.008 rad on TUM-VI imu1).
  not modelled  gyro-accel time offset t_ga, g-sensitivity Tg, rolling-shutter readout.
"""
import json
import os

import numpy as np
from scipy.linalg import rq
from scipy.spatial.transform import Rotation

from mvis.io.result_schema import calyx_dw_decomposed

STATUS = "basalt_status.json"


def read_calibration(path):
    """The calibration dict of a Basalt calibration.json (cereal's value0 wrapper removed)."""
    with open(path) as f:
        d = json.load(f)
    return d.get("value0", d)


def _status(d):
    p = os.path.join(d, STATUS)
    return json.load(open(p)) if os.path.exists(p) else None


def pose_mm_rotvec(T):
    """(translation mm, log(R) rad) of a Basalt pose dict."""
    p = np.array([T["px"], T["py"], T["pz"]], float) * 1000
    rv = Rotation.from_quat([T["qx"], T["qy"], T["qz"], T["qw"]]).as_rotvec()
    return p, rv


def _cam_entry(c):
    vals = dict(c["intrinsics"])
    e = {"model": c["camera_type"],
         "intr": [float(vals.pop(k)) for k in ("fx", "fy", "cx", "cy")]}
    if "rpmax" in vals:
        e["rpmax"] = float(vals.pop("rpmax"))
    e["dist"] = [float(v) for v in vals.values()]
    return e


def gyro_matrix(calib):
    """(bias, I + S_g) from calib_gyro_bias."""
    g = np.asarray(calib["calib_gyro_bias"], float)
    return g[:3], np.eye(3) + g[3:].reshape(3, 3, order="F")


def accel_matrix(calib):
    """(bias, I + S_a) from calib_accel_bias (S_a lower-triangular)."""
    a = np.asarray(calib["calib_accel_bias"], float)
    S = np.zeros((3, 3))
    S[:, 0] = a[3:6]
    S[1, 1], S[2, 1], S[2, 2] = a[6], a[7], a[8]
    return a[:3], np.eye(3) + S


def rotation_times_lower(A):
    """A = R L with R in SO(3) and L lower-triangular with a positive diagonal (the MVIS kalibr
    gyro split R Dw). From the RQ decomposition of A^T: A^T = U Q -> A = Q^T U^T."""
    U, Q = rq(np.asarray(A, float).T)
    s = np.sign(np.diag(U))
    s[s == 0] = 1.0
    U, Q = U * s[None, :], Q * s[:, None]          # U s s Q = U Q, diag(U) > 0
    R, L = Q.T, U.T
    if np.linalg.det(R) < 0:                       # a reflection cannot be a mounting rotation
        raise ValueError("gyro matrix has a negative determinant")
    return R, L


# ---------- camera-only runs (basalt_calibrate) ------------------------------------------------
def camera_only(d):
    """One Basalt camera-only run -> cams (model, intr, dist, p_mm/rotvec in cam0, baseline_mm to
    cam0), or None when the run is not ok or left no calibration.json."""
    st, path = _status(d), os.path.join(d, "calibration.json")
    if not st or st.get("status") != "ok" or not os.path.exists(path):
        return None
    calib = read_calibration(path)
    cams = [_cam_entry(c) for c in calib["intrinsics"]]
    poses = [pose_mm_rotvec(T) for T in calib["T_imu_cam"]]
    # Camera-only: T_imu_cam is relative to cam0, but cam0 need not be exactly the identity.
    p0, rv0 = poses[0]
    R0 = Rotation.from_rotvec(rv0)
    for c, (p, rv) in zip(cams, poses):
        c["p_mm"] = (R0.inv().apply(p - p0)).tolist()
        c["rotvec"] = (R0.inv() * Rotation.from_rotvec(rv)).as_rotvec().tolist()
    for c in cams[1:]:
        c["baseline_mm"] = float(np.linalg.norm(c["p_mm"]))
    return {"file": path, "cams": cams, "status": st}


# ---------- camera + IMU runs (basalt_calibrate_imu) -------------------------------------------
def camera_imu(d):
    """One Basalt camera + IMU run, or None when not ok / no calibration.json.

    cams: model, intr/dist (as solved: fixed unless --opt-intrinsics), td_ms, p_mm, lever_mm,
          rotvec (log R_i_c), baseline_mm to cam0.
    imu:  bias_gyro, bias_accel, Dw_full (= calyx Dw) + its calyx decomposition, Dw/R_rotvec of
          the kalibr split, Da (lower-triangular), imu_rate_hz.
    """
    st, path = _status(d), os.path.join(d, "calibration.json")
    if not st or st.get("status") != "ok" or not os.path.exists(path):
        return None
    calib = read_calibration(path)
    td_ms = float(calib["cam_time_offset_ns"]) * 1e-6
    cams = []
    for c, T in zip(calib["intrinsics"], calib["T_imu_cam"]):
        e = _cam_entry(c)
        p, rv = pose_mm_rotvec(T)
        e.update(td_ms=td_ms, p_mm=p.tolist(), lever_mm=float(np.linalg.norm(p)), rotvec=rv.tolist())
        cams.append(e)
    for e in cams[1:]:
        e["baseline_mm"] = float(np.linalg.norm(np.subtract(e["p_mm"], cams[0]["p_mm"])))

    bg, G = gyro_matrix(calib)
    ba, A = accel_matrix(calib)
    R, L = rotation_times_lower(G)
    imu = {"bias_gyro": bg.tolist(), "bias_accel": ba.tolist(),
           "Dw_full": G.tolist(), "calyx": calyx_dw_decomposed(G),
           "Dw_kalibr": L.tolist(), "R_gyro_rotvec": Rotation.from_matrix(R).as_rotvec().tolist(),
           "Da": A.tolist(), "imu_rate_hz": float(calib.get("imu_update_rate", 0.0))}
    summary = st.get("summary") or {}
    ran = [p for p in summary.get("phases", []) if p["reason"] != "skipped"]
    return {"dir": d, "cams": cams, "imu": imu, "status": st,
            "mean_reprojection_px": ran[-1]["mean_reprojection"] if ran else None}


def mvis_keys(run, variant="kalibr"):
    """Flat `cam<i>.<metric>` / `imu.<metric>` keys for one camera_imu() run, the same
    names an MVIS cam-IMU result is summarised with, for a side-by-side table. variant picks the MVIS gyro reading Dw_diag is compared with:
    'kalibr' = diag of the lower-triangular split, 'calyx' = diag of the full 3x3."""
    out = {}
    for i, c in enumerate(run["cams"]):
        out[f"cam{i}.lever_mm"] = c["lever_mm"]
        out[f"cam{i}.td_ms"] = c["td_ms"]
        if "baseline_mm" in c:
            out[f"cam{i}.baseline_mm"] = c["baseline_mm"]
    dw = run["imu"]["Dw_kalibr"] if variant == "kalibr" else run["imu"]["Dw_full"]
    for j in range(3):
        out[f"base.Dw_diag[{j}]"] = float(dw[j][j])
        out[f"base.Da_diag[{j}]"] = float(run["imu"]["Da"][j][j])
    return out
