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
"""Shared metrics over a ``cam_imu.json`` calibration-result dict.

One definition for the extractions that were hand-rolled in three places (`recalib/roundtrip.py`,
the repeatability study, `regression/cam_imu_adapter.py`): the stereo baseline, a lever
magnitude, and the per-IMU intrinsic/bias rows. Also the across-run spread of those values: chordal
mean rotation and RMS deviations, and for a batch of results (`discover_batch_results`) the
whole table (`repeatability_stats`). gtsam is imported where it is used, so the extractions stay
light.
"""
from __future__ import annotations

import numpy as np


def lever_mm(vec3) -> float:
    """Euclidean norm of a 3-vector lever (mm in, mm out). Takes a raw vector — callers pass a
    cam/IMU ``p_*inI_mm`` list or a mean vector, not a dict — so it serves every call site."""
    return float(np.linalg.norm(np.asarray(vec3, float)))


def stereo_baseline(cams) -> float | None:
    """Stereo baseline ``|p_cam1 - p_cam0|`` (mm) from a result's ``cams`` list, or ``None`` for a
    mono rig (<2 cameras)."""
    if len(cams) < 2:
        return None
    p0 = np.asarray(cams[0]["p_CinI_mm"], float)
    p1 = np.asarray(cams[1]["p_CinI_mm"], float)
    return float(np.linalg.norm(p1 - p0))


def imu_rows(result: dict) -> list[dict]:
    """Per-IMU intrinsic/bias rows from a ``cam_imu.json`` dict: the base IMU (top-level
    ``Dw_diag``/``Da_diag``/``b_g``/``b_a``) first, then each aux IMU from ``imus[]``. Absent
    fields are ``None`` (the json carries no biases unless injected, and aux ``Dw/Da`` only when
    non-trivial). Keyed by ``name`` so callers can align two results by IMU rather than by order."""
    def row(name, d):
        return {"name": name, "Dw": d.get("Dw_diag"), "Da": d.get("Da_diag"),
                "b_g": d.get("b_g"), "b_a": d.get("b_a")}

    rows = [row("imu0", result)]
    for m in result.get("imus", []):
        rows.append(row(m.get("name", "aux"), m))
    return rows


def mean_rot(rotvecs):
    """Chordal mean rotation: average the rotation matrices, project back to SO(3) via SVD, with
    a reflection guard (det<0) so the result is a proper rotation."""
    import gtsam
    Rs = [gtsam.Rot3.Expmap(np.asarray(rv, float)).matrix() for rv in rotvecs]
    U, _, Vt = np.linalg.svd(sum(Rs) / len(Rs))
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return gtsam.Rot3(Rm)


def rot_rms_deg(rotvecs):
    """RMS angular deviation (deg) of the rotations about their chordal mean."""
    import gtsam
    Rm = mean_rot(rotvecs)
    d = [np.rad2deg(np.linalg.norm(gtsam.Rot3.Logmap(
        Rm.inverse().compose(gtsam.Rot3.Expmap(np.asarray(rv, float)))))) for rv in rotvecs]
    return float(np.sqrt(np.mean(np.square(d))))


def lever_rms_mm(levers):
    """RMS deviation (mm) of lever vectors about their mean."""
    L = np.asarray(levers, float)
    return float(np.sqrt(np.mean(np.sum((L - L.mean(0)) ** 2, axis=1))))


def discover_batch_results(d) -> list[tuple[str, dict]]:
    """``(number, cam_imu.json dict)`` of every ``bag<NN>/`` or ``mic<NN>/`` result dir under `d`,
    sorted by path. Dirs with a suffix after the number (e.g. ``bag09_t265_2_only``, a diagnostic
    or sweep run) are skipped."""
    import glob
    import json
    import os
    import re

    bags = []
    for p in sorted(glob.glob(os.path.join(d, "bag*", "cam_imu.json"))
                    + glob.glob(os.path.join(d, "mic*", "cam_imu.json"))):
        m = re.fullmatch(r"(?:bag|mic)(\d+)", os.path.basename(os.path.dirname(p)))
        if not m:
            continue
        with open(p) as f:
            bags.append((m.group(1), json.load(f)))
    return bags


def repeatability_stats(results) -> dict:
    """Across-run 1-sigma spread of every estimated cam-IMU parameter, from ``(name, cam_imu.json
    dict)`` pairs of one rig (the layout of the first run is used for all):

    - ``reproj``: per-run reprojection RMS (px); ``baseline``: per-run stereo baseline (mm), or
      ``None`` for a mono rig;
    - ``cams``: per camera ``name``, ``lever_rms`` (mm), ``lever_mean`` (mm), ``rot_rms`` (deg),
      ``td_std`` (ms);
    - ``imus``: base then aux IMUs, ``name`` and the per-run ``W``/``A`` Dw/Da diagonals;
    - ``aux``: per aux IMU the T_IstoI ``lever_rms``, ``lever_mean``, ``rot_rms``, ``td_std``.
    """
    runs = [b for b, _ in results]
    res = {b: r for b, r in results}
    first = res[runs[0]]
    out = dict(runs=runs, reproj=np.array([res[b]["reproj_rms_px"] for b in runs]), baseline=None)
    ncam = len(first["cams"])
    if ncam >= 2:
        out["baseline"] = np.array([stereo_baseline(res[b]["cams"]) for b in runs])
    out["cams"] = []
    for c in range(ncam):
        lev = [res[b]["cams"][c]["p_CinI_mm"] for b in runs]
        rv = [res[b]["cams"][c]["rotvec"] for b in runs]
        td = np.array([res[b]["cams"][c]["td_ms"] for b in runs])
        out["cams"].append(dict(name=first["cams"][c]["name"], lever_rms=lever_rms_mm(lev),
                                lever_mean=np.asarray(lev, float).mean(0), rot_rms=rot_rms_deg(rv),
                                td_std=td.std()))
    names = ["imu0(base)"] + [m["name"] for m in first["imus"]]
    out["imus"] = [dict(name=nm, W=np.array([imu_rows(res[b])[i]["Dw"] for b in runs]),
                        A=np.array([imu_rows(res[b])[i]["Da"] for b in runs]))
                   for i, nm in enumerate(names)]
    out["aux"] = []
    for j, m0 in enumerate(first["imus"]):
        lev = [res[b]["imus"][j]["p_IsinI_mm"] for b in runs]
        rv = [res[b]["imus"][j]["rotvec"] for b in runs]
        td = np.array([res[b]["imus"][j]["td_ms"] for b in runs])
        out["aux"].append(dict(name=m0["name"], lever_rms=lever_rms_mm(lev),
                               lever_mean=np.asarray(lev, float).mean(0), rot_rms=rot_rms_deg(rv),
                               td_std=td.std()))
    return out
