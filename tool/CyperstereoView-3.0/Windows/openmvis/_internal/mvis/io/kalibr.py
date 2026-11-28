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
"""Unified Kalibr interop — the single import surface for reading/writing/converting between
MVIS and Kalibr's ``camchain`` / ``camchain-imucam`` formats.

Public API
----------
- :func:`read_camchain` — Kalibr ``camchain`` dict → per-camera intrinsics/distortion/resolution.
- :func:`read_camchain_imucam` — ``camchain-imucam`` → the above PLUS the per-camera ``T_cam_imu``
  (and ``T_cn_cnm1``) — the reader that closes the write↔read loop.
- :func:`write_camchain_imucam` — a cam-IMU result → a Kalibr ``camchain-imucam`` dict.
- :func:`kalibr_model_to_internal` / :func:`pack_intrinsics_kalibr` — the low-level model-name +
  intrinsics pack/unpack primitives (MVIS ``model`` + flat cal_vec ↔ Kalibr
  ``camera_model``/``distortion_model``/``intrinsics``/``distortion_coeffs``).

Design note
-----------
The read/write **implementations** live in ``io.camera_input`` (camchain reader) and
``io.cam_imu_output`` (camchain-imucam writer) and are RE-EXPORTED here, not moved: the writer
depends on ``cam_imu_output``'s general result-serialization helpers (``_T_cam_imu``/``_img_size``)
and the reader on ``camera_input``'s camera-model helpers, so moving them would create a circular
import between those two modules and this one. ``io/kalibr.py`` is therefore the **facade** — import
Kalibr I/O from here; the primitives stay with the serializers they belong to.

The other two Kalibr surfaces are class-shaped (not camchain dicts) and stay where they are:
- board-target Kalibr YAML — ``board.apriltag_board``'s ``from_kalibr_yaml`` classmethod;
- the Kalibr IMU-intrinsics model — ``imu.kalibr.KalibrImu``.

Reading finished Kalibr runs (results txt, camchains, imu yaml; no solving) also lives here:
:func:`parse_results`, :func:`camera_only`, :func:`camera_imu`, plus :func:`norm_topic` and
:func:`centre_in_imu_mm`. Two camchain readers deliberately keep different intrinsics conventions:

- :func:`_cam_entry` (behind ``camera_only`` / ``camera_imu``) keeps Kalibr's raw ``intrinsics`` and
  ``distortion_coeffs`` order for every model;
- :func:`parse_kalibr_camchain` normalises ds, eucm and omni to ``[fu, fv, pu, pv]`` and moves the
  projection parameters into ``distortion``; omni-none becomes MVIS ``ucm`` in alpha form.

Conventions of the run readers, all checked against the MVIS sources:

  time offset   Kalibr timeshift_cam_imu: t_imu = t_cam + shift. MVIS camera td: the image stamped
                t_cam sees the IMU pose at t_cam + td (camera_imu_factor.py). Same sign: compared as is.
                Kalibr aux IMU time_offset (w.r.t. imu0) vs MVIS aux td: same sign.
  extrinsics    Kalibr T_cam_imu (and results T_ci) maps imu0 points into the camera, so the camera
                centre in the IMU frame is p = -R^T t. MVIS stores the same T_cam_imu. Kalibr T_i_b
                (imu0 -> imuN) gives the aux IMU position in imu0 as -R^T t, MVIS p_IsinI.
  baseline      Kalibr T_cn_cnm1 maps cam_{n-1} -> cam_n; chained to cam0 for a camera-to-first-camera
                distance.
  distortion    same coefficient order on both sides (equidistant k1..k4, radtan k1 k2 p1 p2).
  readout       Kalibr pinhole-radtan-rs line_delay is per image row: tr = line_delay * rows.
  IMU model     Kalibr scale-misalignment (IccSensors.py): w_m = M_g C_gyro_i w + A C_gyro_i a + b,
                a_m = M_a a + b, with w and a in the IMU (accel) frame, so A acts on the specific force in
                the gyro frame. MVIS kalibr model: w_m = Dw^-1 R^T w + Tg a + b, a in the IMU frame. So
                Dw = inv(M_g), Da = inv(M_a), R^T = C_gyro_i and Tg = A C_gyro_i.

:func:`invert_scaled_misaligned_imu` undoes that Kalibr IMU model on one measurement (used by
the bag-rewriting study that applies a Kalibr calibration to a bag).

Gap (TODO, no consumer yet): there is NO Kalibr ``imu.yaml`` (noise-density) read/write — only the
``KalibrImu`` intrinsic-model convention. Add ``read_imu_yaml``/``write_imu_yaml`` here if a
consumer needs to import/export Kalibr IMU noise parameters.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import yaml

from mvis.io.camera_input import _from_kalibr as read_camchain
from mvis.io.camera_input import kalibr_cam_keys
from mvis.io.camera_input import _norm_model as kalibr_model_to_internal
from mvis.io.cam_imu_output import _kalibr_pack as pack_intrinsics_kalibr
from mvis.io.cam_imu_output import to_kalibr_camchain_imucam as write_camchain_imucam

__all__ = [
    "read_camchain", "read_camchain_imucam", "write_camchain_imucam",
    "kalibr_model_to_internal", "pack_intrinsics_kalibr",
    "parse_results", "centre_in_imu_mm", "norm_topic", "camera_only", "camera_imu",
    "parse_kalibr_camchain", "invert_scaled_misaligned_imu",
]


def read_camchain_imucam(doc: dict) -> list[dict]:
    """Read a Kalibr ``camchain-imucam`` dict into per-camera entries — the intrinsics/distortion
    /resolution from :func:`read_camchain` PLUS the cam-IMU extrinsics the camera-input reader
    drops: ``T_cam_imu`` (4x4) and, when present, ``T_cn_cnm1`` (stereo baseline). The camera
    order matches :func:`read_camchain` (Kalibr ``cam0, cam1, …`` by number)."""
    cams = read_camchain(doc)
    for cam, key in zip(cams, kalibr_cam_keys(doc)):
        cc = doc[key]
        if "T_cam_imu" in cc:
            cam["T_cam_imu"] = [[float(x) for x in row] for row in cc["T_cam_imu"]]
        if "T_cn_cnm1" in cc:
            cam["T_cn_cnm1"] = [[float(x) for x in row] for row in cc["T_cn_cnm1"]]
    return cams


def _mat(lines, i, n):
    txt = " ".join(lines[i:i + n]).replace("[", " ").replace("]", " ")
    return np.array([float(x) for x in txt.split()]).reshape(n, -1)


def parse_results(path):
    """Parse a Kalibr `*-results-imucam.txt` into numpy arrays."""
    L = open(path).read().splitlines()
    out = {"cams": {}, "imus": []}
    cur_imu, sect = None, None
    for i, l in enumerate(L):
        s = l.strip()
        m = re.match(r"Reprojection error \((cam\d)\) \[px\]:\s+mean ([\d.e-]+), median ([\d.e-]+)"
                     r"(?:, std:?\s*([\d.e-]+))?", s)
        if m:
            out["cams"].setdefault(m.group(1), {}).update(reproj_mean=float(m.group(2)), reproj_median=float(m.group(3)),
                                                          **({"reproj_std": float(m.group(4))} if m.group(4) else {}))
        m = re.match(r"(Gyroscope|Accelerometer) error \((imu\d)\) \[[^\]]+\]:\s+mean ([\d.e-]+), median ([\d.e-]+)", s)
        if m:
            out.setdefault("resid", {})[(m.group(2), m.group(1))] = (float(m.group(3)), float(m.group(4)))
        m = re.match(r"Transformation \((cam\d)\):", s)
        if m:
            cam = m.group(1)
            j = next(k for k in range(i, i + 6) if L[k].strip().startswith("T_ci"))
            out["cams"].setdefault(cam, {})["T_ci"] = _mat(L, j + 1, 4)
        m = re.match(r"timeshift (cam\d) to imu0: \[s\]", s)
        if m:
            out["cams"].setdefault(m.group(1), {})["shift_ms"] = float(L[i + 1]) * 1000
        m = re.match(r"^IMU(\d):$", s)
        if m:
            cur_imu = {"idx": int(m.group(1))}
            out["imus"].append(cur_imu)
            sect = None
        if cur_imu is not None:
            if s.startswith("T_ib"):
                cur_imu["T_ib"] = _mat(L, i + 1, 4)
            elif s.startswith("time offset with respect to IMU0"):
                cur_imu["delay_ms"] = float(re.search(r"([-\d.e]+) \[s\]", s).group(1)) * 1000
            elif s == "Gyroscope:":
                sect = "g"
            elif s == "Accelerometer:" and "M_g" in cur_imu:
                sect = "a"
            elif s == "M:" and sect:
                cur_imu["M_" + sect] = _mat(L, i + 1, 3)
            elif s.startswith("A [") and sect == "g":
                cur_imu["A"] = _mat(L, i + 1, 3)
            elif s.startswith("C_gyro_i") and sect == "g":
                cur_imu["C_gyro_i"] = _mat(L, i + 1, 3)
    return out


# ---------- geometry helpers --------------------------------------------------------------------
def centre_in_imu_mm(T):
    """Point-mapping imu -> sensor transform: the sensor origin in the IMU frame, mm."""
    T = np.asarray(T, float)
    return -T[:3, :3].T @ T[:3, 3] * 1000


def norm_topic(t):
    """Match key for a sensor: ROS topic without /image_raw, with a leading slash (MVIS euroc rows name
    sensors 'cam0' / 'imu0', Kalibr bags '/cam0/image_raw' / '/imu0')."""
    t = (t or "").removesuffix("/image_raw")
    return "/" + t.lstrip("/") if t else t


def _cam_entry(c):
    e = {"topic": norm_topic(c.get("rostopic")), "model": c.get("distortion_model"),
         "intr": [float(x) for x in c["intrinsics"]], "dist": [float(x) for x in c.get("distortion_coeffs", [])],
         "rows": int(c["resolution"][1])}
    if c.get("line_delay") is not None:
        e["tr_ms"] = float(c["line_delay"]) * e["rows"] * 1000
    return e


def _chain_to_cam0(cams_raw):
    """T_cam_c <- cam0 for every camera of a camchain (identity for cam0)."""
    Ts, T = [], np.eye(4)
    for i, c in enumerate(cams_raw):
        if i:
            T = np.asarray(c["T_cn_cnm1"], float) @ T
        Ts.append(T.copy())
    return Ts


def _status_ok(d):
    p = os.path.join(d, "kalibr_status.json")
    return os.path.exists(p) and bool(json.load(open(p)).get("ok"))


# ---------- camera-only runs (kalibr_calibrate_cameras / pinhole-radtan-rs) --------------------
def camera_only(d, pattern="*-camchain.yaml"):
    """One Kalibr camera-only run -> list of cameras (cam0..), each with baseline_mm to cam0, or None
    when the run failed (kalibr_status ok false) or left no camchain."""
    if not _status_ok(d):
        return None
    hits = [h for h in glob.glob(os.path.join(d, pattern)) if "imucam" not in h]
    if not hits:
        return None
    assert len(hits) == 1, f"ambiguous Kalibr result: {hits}"
    doc = yaml.safe_load(open(hits[0]))
    raw = [doc[k] for k in sorted(doc, key=lambda k: int(k[3:])) if k.startswith("cam")]
    cams = [_cam_entry(c) for c in raw]
    for c, T in zip(cams[1:], _chain_to_cam0(raw)[1:]):
        c["baseline_mm"] = float(np.linalg.norm(T[:3, 3]) * 1000)
    return {"file": hits[0], "cams": cams}


# ---------- camera + IMU runs (kalibr_calibrate_imu_camera) ------------------------------------
def camera_imu(d):
    """One Kalibr camera + IMU run (input-*.yaml outputs + results txt), or None if absent.

    cams: topic, intr/dist (the run's fixed input), td_ms (timeshift), p_mm and lever_mm to imu0,
          baseline_mm to cam0 (from the fixed input chain), reproj_mean / reproj_median / reproj_std
          (std None in a results file without it).
    imus: topic, Dw = inv(M_g), Da = inv(M_a), Tg = A C_gyro_i, and for imu1..: p_mm, lever_mm, td_ms.
    """
    cc_p, im_p = os.path.join(d, "input-camchain-imucam.yaml"), os.path.join(d, "input-imu.yaml")
    res_p = os.path.join(d, "input-results-imucam.txt")
    if not (os.path.exists(cc_p) and os.path.exists(im_p) and os.path.exists(res_p)):
        return None
    cc, im, res = yaml.safe_load(open(cc_p)), yaml.safe_load(open(im_p)), parse_results(res_p)
    # runs on a retimed bag (run_imu.py retime: camera stamps pre-shifted by the MVIS td) report only the residual
    # timeshift; add the applied shift back (recorded in comparison.json) to get the camera->IMU offset
    cmp_p = os.path.join(d, "comparison.json")
    applied = (json.load(open(cmp_p))["kalibr"].get("timeshift_applied_ms", {}) if os.path.exists(cmp_p) else {})
    cams = []
    for k in sorted((k for k in cc if k.startswith("cam")), key=lambda k: int(k[3:])):
        c = cc[k]
        e = _cam_entry(c)
        p = centre_in_imu_mm(c["T_cam_imu"])
        e.update(p_mm=p.tolist(), lever_mm=float(np.linalg.norm(p)),
                 td_ms=float(c["timeshift_cam_imu"]) * 1000 + applied.get(k, 0.0),
                 **{f: res["cams"].get(k, {}).get(f) for f in ("reproj_mean", "reproj_median", "reproj_std")})
        cams.append(e)
    for e in cams[1:]:
        e["baseline_mm"] = float(np.linalg.norm(np.subtract(e["p_mm"], cams[0]["p_mm"])))
    imus = []
    for k in sorted((k for k in im if k.startswith("imu")), key=lambda k: int(k[3:])):
        b = im[k]
        g = b["gyroscopes"]
        e = {"name": k, "topic": norm_topic(b.get("rostopic")), "model": b.get("model"),
             "Dw": np.linalg.inv(np.asarray(g["M"], float)).tolist(),
             "Da": np.linalg.inv(np.asarray(b["accelerometers"]["M"], float)).tolist(),
             "Tg": (np.asarray(g.get("A", np.zeros((3, 3))), float)
                    @ np.asarray(g.get("C_gyro_i", np.eye(3)), float)).tolist()}
        if k != "imu0":
            p = centre_in_imu_mm(b["T_i_b"])
            e.update(p_mm=p.tolist(), lever_mm=float(np.linalg.norm(p)), td_ms=float(b.get("time_offset", 0.0)) * 1000)
        imus.append(e)
    return {"dir": d, "cams": cams, "imus": imus}


# ---------- camchain with normalised intrinsics (Kalibr side-by-side, doc/comparison_kalibr.md) -----
def parse_kalibr_camchain(camchain: dict) -> list[dict]:
    """Extract per-camera params from Kalibr camchain YAML, with ds, eucm and omni intrinsics
    normalised to ``[fu, fv, pu, pv]`` (:func:`_cam_entry` keeps them raw).

    The conversion is :func:`read_camchain`'s (the cam-IMU camera-input reader): ds / eucm move
    their projection parameters into ``distortion``, omni-radtan gives ``[xi, k1, k2, p1, p2]``,
    omni-none becomes MVIS ``ucm`` in alpha form, and the coefficients are fitted to the model's
    length. ``model`` stays Kalibr's ``camera_model`` (``ucm`` for omni-none). Cameras come in
    number order (cam2 before cam10)."""
    cameras = []
    for key in kalibr_cam_keys(camchain):
        cam = camchain[key]
        cm = cam.get("camera_model", "")
        s = read_camchain({key: {**cam, "rostopic": cam.get("rostopic", ""),
                                 "resolution": cam.get("resolution") or [0, 0]}})[0]
        if s["distortion_model"] == "ucm" and cm == "omni":
            cm = "ucm"
        cameras.append({
            "camera_id": int(key.replace("cam", "")),
            "model": cm,
            "distortion_model": cam.get("distortion_model", ""),
            "intrinsics": s["intrinsics"],
            "distortion": s["distortion_coeffs"],
            "resolution": cam.get("resolution", []),
            "T_cn_cnm1": cam.get("T_cn_cnm1"),  # extrinsic to previous cam
        })
    return cameras


# ---------- Kalibr scale-misalignment IMU model, inverted ------
def invert_scaled_misaligned_imu(w_m, a_m, b_g, b_a, M_g, M_a, C_gyro_i, A):
    """Corrected (w, a) in the IMU (accel) frame from one raw gyro / accel measurement.

    Kalibr ``IccScaledMisalignedImu`` on the base IMU (C_i_b = I, r_b = 0)::

        a_m = M_a a + b_a
        w_m = M_g (C_gyro_i w) + A (C_gyro_i a) + b_g

    so ``a = M_a^-1 (a_m - b_a)`` and ``w = C_gyro_i^-1 M_g^-1 (w_m - A C_gyro_i a - b_g)``
    (A acts on the specific force in the gyro frame). Each argument is a length-3 vector or a
    3x3 matrix; returns two length-3 arrays.
    """
    M_g, M_a, C, A = (np.asarray(m, float) for m in (M_g, M_a, C_gyro_i, A))
    a = np.linalg.inv(M_a) @ (np.asarray(a_m, float) - np.asarray(b_a, float))
    w = np.linalg.inv(C) @ (np.linalg.inv(M_g) @ (np.asarray(w_m, float) - A @ (C @ a)
                                                   - np.asarray(b_g, float)))
    return w, a
