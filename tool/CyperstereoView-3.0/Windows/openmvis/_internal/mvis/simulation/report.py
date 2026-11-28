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
"""Recovery report / metrics — the single place that scores a cam-IMU recovery
against ground truth and renders the true-vs-estimate table.

Replaces the report logic that was copy-pasted across the study scripts (and every
scratchpad one-off): `recovery_errors(gt, res)` returns a structured error dict;
`format_recovery_table(gt, res)` renders the human-readable table. IMU intrinsics
are reconstructed the canonical way (`ImuModel.from_tangent`, NOT `retract`).
"""
from __future__ import annotations

import numpy as np
import gtsam

from mvis.core.camera_models import DIST_NAMES, canonical_cal_vec
from mvis.imu import ImuModel, parse_imu_model

_CAM_NAMES = ["fx", "fy", "cx", "cy"]


# models whose distortion rows carry their registry names (kb4: k1..k4, ucm:
# alpha); the others (omni included) keep the historical generic d0, d1, …
# labels, so their existing recovery output is unchanged
_NAMED_DIST = ("kb4", "ucm")


def _intr_names(n: int, model: str | None = None):
    """Labels for the first `n` intrinsics: fx/fy/cx/cy then the distortion
    coefficients — named for `_NAMED_DIST` models, else d0, d1, … for however
    many the model carries."""
    if model in _NAMED_DIST:
        names = list(DIST_NAMES[model])
        names += [f"d{k}" for k in range(len(names), max(0, n - 4))]
        return (_CAM_NAMES + names)[:n]
    return _CAM_NAMES[:n] + [f"d{k}" for k in range(max(0, n - 4))]


def _rot_err_deg(R_gt: gtsam.Rot3, R_est: gtsam.Rot3) -> float:
    """Geodesic angle (degrees) between two rotations — the norm of the residual
    rotation's log, so it is the true SO(3) distance, not a per-axis difference."""
    return float(np.rad2deg(np.linalg.norm(
        gtsam.Rot3.Logmap(R_gt.inverse().compose(R_est)))))


def _gt_intr(gt: dict, cal) -> np.ndarray:
    """GT intrinsics, a legacy short vector (8-long omni: no p2) zero-padded to
    the estimate's model length so the score never silently drops a coefficient."""
    gi = np.asarray(gt['cam_intrinsics'], float)
    if len(gi) < len(cal.intr):
        gi = np.asarray(canonical_cal_vec(cal.model, gi), float)
    return gi


def recovery_errors(gt: dict, res) -> dict:
    """Structured |error| of a recovery vs GT. Keys: ok, reproj_px, cam_intr
    (per-coeff abs px), R_CtoI_deg, lever_mm (x/y/z + norm), td_ms, Dw/Da
    (per-diag abs), and optionally readout_ms and board_poses (rel-to-anchor)."""
    cal = res.calib
    gi = _gt_intr(gt, cal)
    ei = np.asarray(cal.intr, float)
    n = min(len(gi), len(ei))
    cam_intr = {nm: float(abs(ei[k] - gi[k]))
                for nm, k in zip(_intr_names(n, cal.model), range(n))}

    gl = np.asarray(gt['T_CtoI'].translation(), float) * 1e3
    el = np.asarray(cal.T.translation(), float) * 1e3
    lever = {a: float(el[i] - gl[i]) for i, a in enumerate("xyz")}
    lever["norm"] = float(np.linalg.norm(el - gl))

    model = parse_imu_model(res.diagnostics.get('imu_model', 'calyx'))
    ii = ImuModel.from_tangent(np.asarray(res.imu_intr_xi, float), model)
    g = gt['imu_intr']
    Dw = {a: float(abs(np.diag(ii.Dw())[i] - np.diag(g.Dw())[i]))
          for i, a in enumerate("xyz")}
    Da = {a: float(abs(np.diag(ii.Da())[i] - np.diag(g.Da())[i]))
          for i, a in enumerate("xyz")}

    out = dict(ok=bool(res.ok), reproj_px=float(res.diagnostics.get('reproj_rms_px', np.nan)),
               cam_intr=cam_intr,
               R_CtoI_deg=_rot_err_deg(gt['T_CtoI'].rotation(), cal.T.rotation()),
               lever_mm=lever, td_ms=float(cal.td * 1e3), Dw=Dw, Da=Da)

    if gt.get('readout'):
        out['readout_ms'] = dict(gt=float(gt['readout'] * 1e3), est=float(cal.tr * 1e3),
                                 err=float(abs(cal.tr - gt['readout']) * 1e3))

    # board poses, relative to the anchor (board 0) — gauge-invariant
    gbp = gt.get('board_poses')
    ebp = res.diagnostics.get('board_poses')
    if gbp is not None and ebp and len(gbp) > 1:
        ids = sorted(int(b) for b in ebp)
        a = ids[0]
        gA, eA = gbp[a], gtsam.Pose3(np.asarray(ebp[a], float))
        boards = {}
        for b in ids[1:]:
            if b >= len(gbp):
                continue
            rel_gt = gA.between(gbp[b])
            rel_est = eA.between(gtsam.Pose3(np.asarray(ebp[b], float)))
            d = rel_gt.between(rel_est)
            boards[b] = dict(dR_deg=float(np.rad2deg(np.linalg.norm(
                gtsam.Rot3.Logmap(d.rotation())))),
                dt_mm=float(np.linalg.norm(np.asarray(d.translation()) * 1e3)))
        out['board_poses'] = boards
    return out


def format_recovery_table(gt: dict, res) -> str:
    """Human-readable true / est / |err| table for one recovery."""
    cal = res.calib
    gi = _gt_intr(gt, cal)
    ei = np.asarray(cal.intr, float)
    lines = [f"ok={res.ok}  reproj={res.diagnostics.get('reproj_rms_px', float('nan')):.3f} px",
             f"  {'param':<13}{'true':>13}{'est':>13}{'|err|':>13}"]

    def row(nm, t, e, u=""):
        """Append one `param / true / est / |err|` line, with an optional unit tag."""
        lines.append(f"  {nm:<13}{t:13.5f}{e:13.5f}{abs(e - t):13.5f} {u}")

    for nm, t, e in zip(_intr_names(min(len(gi), len(ei)), cal.model), gi, ei):
        row("cam " + nm, t, e)
    rg = gtsam.Rot3.Logmap(gt['T_CtoI'].rotation()) * 180 / np.pi
    re = gtsam.Rot3.Logmap(cal.T.rotation()) * 180 / np.pi
    for a, t, e in zip("xyz", rg, re):
        row("R_CtoI " + a, t, e, "deg")
    for a, t, e in zip("xyz", np.asarray(gt['T_CtoI'].translation()) * 1e3,
                       np.asarray(cal.T.translation()) * 1e3):
        row("t_CtoI " + a, t, e, "mm")
    row("td", 0.0, cal.td * 1e3, "ms")
    if gt.get('readout'):
        row("readout tr", gt['readout'] * 1e3, cal.tr * 1e3, "ms")
    model = parse_imu_model(res.diagnostics.get('imu_model', 'calyx'))
    ii = ImuModel.from_tangent(np.asarray(res.imu_intr_xi, float), model)
    g = gt['imu_intr']
    for a, t, e in zip("xyz", np.diag(g.Dw()), np.diag(ii.Dw())):
        row("gyro Dw " + a, t, e)
    for a, t, e in zip("xyz", np.diag(g.Da()), np.diag(ii.Da())):
        row("accel Da " + a, t, e)
    return "\n".join(lines)


def format_absolute(gt, res, *, rig: str | None = None, cam_labels=None) -> str:
    """Every calibration result of a `synth.run_subset` recovery as absolute TRUE vs RECOVERED:
    per camera the intrinsics, the T_CtoI lever and rotation angle, td; per IMU the gyro Dw
    diagonal and, for an aux IMU, T_IstoI and td. `res` is the run_subset output (`rows` from
    `zscore_table`, `reproj_rms_px`, `n_fail`); rows with |z| >= 3 carry a `*`. `rig` names the
    setup in the header (default "<n> cam + <m> imu"); `cam_labels[c]` tags camera c."""
    from mvis.core.rotation import rotation_angle
    from mvis.simulation.from_result import gt_cam_model
    from mvis.simulation.synth import intr_row_names

    def angle_deg(R):
        return float(np.degrees(rotation_angle(R)))

    def star(z):
        return "" if abs(z) < 3 else " *"

    R = {r["param"]: r for r in res["rows"]}
    rig = rig or "%d cam + %d imu" % (len(gt.cams), len(gt.imus))
    L = ["\n================  ALL CALIBRATION RESULTS: TRUE vs RECOVERED  "
         "(%s, reproj=%.3f px)  ================" % (rig, res["reproj_rms_px"])]

    def lever_lines(Tg, d, name):
        Tr = Tg.compose(gtsam.Pose3.Expmap(d))
        tg, tr = Tg.translation() * 1e3, Tr.translation() * 1e3
        L.append("  extrinsic %s translation (mm):" % name)
        L.append("    TRUE  [%7.1f %7.1f %7.1f]  rot %.2f deg"
                 % (tg[0], tg[1], tg[2], angle_deg(Tg.rotation().matrix())))
        L.append("    RECOV [%7.1f %7.1f %7.1f]  rot %.2f deg   lever err %.2f mm"
                 % (tr[0], tr[1], tr[2], angle_deg(Tr.rotation().matrix()), np.linalg.norm(tr - tg)))

    def td_line(td):
        L.append("  td: TRUE %+.3f ms  RECOV %+.3f ms  err %+.3f ms  z %.2f%s"
                 % (td["gt"] * 1e3, td["est"] * 1e3, td["err"] * 1e3, td["z"], star(td["z"])))

    for c, cam in enumerate(gt.cams):
        label = cam_labels[c] if cam_labels and c < len(cam_labels) else None
        L.append("\n--- CAMERA %d%s ---" % (c, " (%s)" % label if label else ""))
        L.append("  %-6s %12s %12s %12s %8s" % ("intr", "TRUE", "RECOV", "err", "z"))
        for nm in intr_row_names(gt_cam_model(cam)):
            r = R.get("cam%d/%s" % (c, nm))
            if r:
                L.append("  %-6s %12.4f %12.4f %+12.4f %8.2f%s"
                         % (nm, r["gt"], r["est"], r["err"], r["z"], star(r["z"])))
        d = np.array([R["cam%d/rot%d" % (c, a)]["err"] for a in range(3)]
                     + [R["cam%d/pos%d" % (c, a)]["err"] for a in range(3)])
        lever_lines(gtsam.Pose3(np.array(cam["T_CtoI"])), d, "T_CtoI")
        td_line(R["cam%d/td" % c])

    for m in gt.imus:
        nm = m["name"]
        L.append("\n--- IMU %s%s ---" % (nm, "  (reference frame)" if m["is_base"] else ""))
        L.append("  gyro scale Dw diagonal (TRUE -> RECOV):")
        for a in range(3):
            r = R.get("%s/Dw%d" % (nm, a))
            if r:
                L.append("    Dw[%d] %8.4f -> %8.4f  (%+.1f%%)  z %.2f%s"
                         % (a, r["gt"], r["est"], 100 * r["err"] / r["gt"], r["z"], star(r["z"])))
        if not m["is_base"]:
            d = np.array([R["%s/pose%d" % (nm, a)]["err"] for a in range(6)])
            lever_lines(gtsam.Pose3(np.array(m["T_IstoI"])), d, "T_IstoI")
            td_line(R["%s/td" % nm])
    L.append("\n  reproj = %.3f px (noise floor),  total |z|>=3: %d/%d  (* = |z|>=3)"
             % (res["reproj_rms_px"], res["n_fail"], len(res["rows"])))
    return "\n".join(L)
