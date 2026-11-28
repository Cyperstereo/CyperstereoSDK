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
"""Calibration verdict — grade a finished MVIS result GREEN / YELLOW / RED.

Single source of the quality thresholds used by `mvis verdict` and the cam-IMU
HTML report. A verdict is a list of checks, each with a stable ``id`` (so an
agent or a human can look it up), the measured value, the threshold and a
status ``pass | warn | fail``. The overall grade is RED if any check fails,
YELLOW if any warns, GREEN otherwise.

Inputs:
  * camera-only result YAML (``mvis run`` mode ``cam``: cameras /
    reprojection_errors sections)
  * cam-IMU summary JSON (``mvis run`` mode ``cam-imu``: ``output.path``)

``ok=true`` from the solver is never treated as evidence of a good fit; it only
means the solve finished and passed its own gates.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import yaml

# Reprojection grading thresholds (px) in corner-norm (Kalibr) units: the
# original per-axis thresholds (median 1.0/2.0, p95 2.0/4.0) scaled by sqrt(2).
MED_GRADE = (math.sqrt(2), 2 * math.sqrt(2))        # ~1.41 / 2.83 px
P95_GRADE = (2 * math.sqrt(2), 4 * math.sqrt(2))    # ~2.83 / 5.66 px
RMS_GRADE = MED_GRADE
CHI2_GRADE = (1.5, 3.0)                             # chi2 / dof
# Physical plausibility
FOCAL_ASPECT_WARN = 0.02        # |fx/fy - 1|
PP_OFFSET_WARN = 0.10           # |c - size/2| / size
IMU_SCALE_WARN = 0.05           # |Dw/Da diagonal - 1|
TD_WARN_MS = 50.0               # |camera-IMU td|; can be real (USB latency) -> warn only
MIN_OBS_WARN = 500              # corner observations per camera

PASS, WARN, FAIL = "pass", "warn", "fail"
_VERDICT_GRADE = {PASS: "GREEN", WARN: "YELLOW", FAIL: "RED"}   # worst check status -> grade
#: How the HTML reports name a status (badge text and the ``badge-<name>`` class).
GRADE_DISPLAY = {PASS: "good", WARN: "marginal", FAIL: "poor"}


@dataclass
class Check:
    id: str
    subject: str
    value: object
    threshold: str
    status: str
    note: str = ""


@dataclass
class Verdict:
    kind: str
    grade: str = "GREEN"
    checks: list = field(default_factory=list)
    independent_check: str = "none"     # none | roundtrip | holdout | repeatability | cross-tool

    def add(self, *a, **kw):
        self.checks.append(Check(*a, **kw))

    def finish(self):
        from mvis.doctor.findings import worst_status
        self.grade = _VERDICT_GRADE[worst_status(c.status for c in self.checks)]
        return self

    def to_dict(self):
        d = asdict(self)
        d["verified"] = self.independent_check != "none"
        return d


def grade(value, good, marginal, *, strict=False):
    """Lower is better: pass <= good, warn <= marginal, fail above; with ``strict``, pass < good
    and warn < marginal. None for a missing or NaN value."""
    if value is None or value != value:
        return None
    if strict:
        return PASS if value < good else WARN if value < marginal else FAIL
    return PASS if value <= good else WARN if value <= marginal else FAIL


_grade = grade


def _residual_checks(v, subject, median=None, p95=None, rms=None):
    if median is not None and p95 is not None:
        v.add("reproj.median", subject, round(median, 4),
              "<= %.2f / %.2f px" % MED_GRADE, _grade(median, *MED_GRADE))
        v.add("reproj.p95", subject, round(p95, 4),
              "<= %.2f / %.2f px" % P95_GRADE, _grade(p95, *P95_GRADE))
    elif rms is not None:
        v.add("reproj.rms", subject, round(rms, 4),
              "<= %.2f / %.2f px" % RMS_GRADE, _grade(rms, *RMS_GRADE))


def _intrinsics_checks(v, subject, fx, fy, cx, cy, size, dist):
    if not (fx > 0 and fy > 0):
        v.add("intr.focal_positive", subject, [fx, fy], "> 0", FAIL)
        return
    asp = abs(fx / fy - 1.0)
    v.add("intr.focal_aspect", subject, round(asp, 4),
          f"|fx/fy-1| <= {FOCAL_ASPECT_WARN}", PASS if asp <= FOCAL_ASPECT_WARN else WARN,
          "non-square pixels are rare; check board size / coverage")
    if size:
        off = max(abs(cx - size[0] / 2) / size[0], abs(cy - size[1] / 2) / size[1])
        v.add("intr.principal_point", subject, round(off, 4),
              f"offset/size <= {PP_OFFSET_WARN}", PASS if off <= PP_OFFSET_WARN else WARN,
              "principal point far from centre: weak edge coverage?")
    finite = all(isinstance(c, (int, float)) and math.isfinite(c) for c in dist)
    v.add("intr.distortion_finite", subject, finite, "finite", PASS if finite else FAIL)


def assess_cam(path: str) -> Verdict:
    """Grade a camera-only result YAML."""
    with open(path) as f:
        d = yaml.safe_load(f)
    v = Verdict(kind="cam")
    rescue = d.get("lm_rescue")
    if rescue and rescue.get("failed"):
        v.add("solve.lm_rescue", "solve", rescue.get("ratio_after"), "< 0.5 of the cost still to go", FAIL,
              "LM still stuck after the rescue: the result is not at a minimum")
    errs = {e["camera"]: e for e in (d.get("reprojection_errors") or {}).get("per_camera", [])}
    for cam in d.get("cameras", []):
        cid = cam["id"]
        s = f"cam{cid}"
        it = cam.get("intrinsics", {})
        _intrinsics_checks(v, s, it.get("fx", 0), it.get("fy", 0), it.get("cx", 0),
                           it.get("cy", 0), cam.get("image_size"),
                           (cam.get("distortion") or {}).get("coeffs", []))
        e = errs.get(cid)
        if e is None:
            v.add("reproj.missing", s, None, "present", FAIL)
            continue
        _residual_checks(v, s, e.get("median"), e.get("p95"), e.get("rms"))
        n = e.get("num_observations", 0)
        v.add("data.observations", s, n, f">= {MIN_OBS_WARN}",
              PASS if n >= MIN_OBS_WARN else WARN, "few corners: low coverage")
    if not d.get("cameras"):
        v.add("result.cameras", "result", 0, ">= 1 camera", FAIL)
    return v.finish()


def _diag_checks(v, imu, subject="Dw/Da"):
    for key in ("Dw_diag", "Da_diag"):
        vals = imu.get(key)
        if not vals:
            continue
        dev = max(abs(x - 1.0) for x in vals)
        v.add(f"imu.{key[:2].lower()}_scale", subject if subject != "Dw/Da" else key,
              round(dev, 4), f"|diag-1| <= {IMU_SCALE_WARN}",
              PASS if dev <= IMU_SCALE_WARN else WARN,
              "large scale error: plausible for a raw consumer IMU; confirm across recordings")


def assess_cam_imu(path: str, reference: dict | None = None) -> Verdict:
    """Grade a cam-IMU summary JSON; optional `reference` (CamImuRunConfig.reference)."""
    with open(path) as f:
        d = json.load(f)
    v = Verdict(kind="cam-imu")
    v.add("solve.ok", "solve", d.get("ok"), "true", PASS if d.get("ok") else FAIL,
          "solver gates failed (init or rms gate)" if not d.get("ok") else "")
    _residual_checks(v, "all cameras", d.get("reproj_median_px"),
                     d.get("reproj_p95_px"), d.get("reproj_rms_px"))
    diag = d.get("diagnostics") or {}
    if "iterations" in diag and "max_iterations" in diag:
        capped = diag["iterations"] >= diag["max_iterations"]
        v.add("solve.converged", "solve", diag["iterations"],
              f"< max_iterations ({diag['max_iterations']})", WARN if capped else PASS,
              "hit the iteration cap: may not be converged" if capped else "")
    no_step = diag.get("lm_no_step")
    if no_step:
        v.add("solve.lm_step", "solve", 0, "> 0 iterations", FAIL if no_step.get("failed") else WARN,
              no_step.get("reason", ""))
    for k, label in (("cam_chi2_dof", "camera"), ("imu_chi2_dof", "IMU")):
        if diag.get(k) is not None:
            v.add(f"fit.{k}", label, diag[k], "<= %.1f / %.1f" % CHI2_GRADE,
                  _grade(diag[k], *CHI2_GRADE))
    for c in d.get("cams", []):
        s = c["name"]
        _residual_checks(v, s, c.get("reproj_median"), c.get("reproj_p95"), c.get("reproj"))
        td = c.get("td_ms")
        if td is not None:
            v.add("time.td", s, td, f"|td| <= {TD_WARN_MS} ms",
                  PASS if abs(td) <= TD_WARN_MS else WARN,
                  "large offset: real acquisition latency or a bad stamp")
    _diag_checks(v, d)
    for m in d.get("imus", []):
        if m.get("chi2_dof") is not None:
            v.add("fit.aux_imu_chi2_dof", m["name"], m["chi2_dof"],
                  "<= %.1f / %.1f" % CHI2_GRADE, _grade(m["chi2_dof"], *CHI2_GRADE))
        _diag_checks(v, m, m["name"])
    if reference:
        _reference_checks(v, d, reference)
    return v.finish()


def _reference_checks(v, d, ref, lever_tol_mm=10.0, baseline_tol_mm=5.0):
    """Compare against known values (CAD / spec) — consistency, not truth.
    `ref` keys per camera may be `camN`, `N` or `'N'`; `baseline_mm` a scalar or
    a vector (mvis.analysis.reference)."""
    from mvis.analysis.reference import reference_deltas
    ref = reference_deltas(ref)
    cams = {c["name"]: c for c in d.get("cams", [])}
    for i, lever in ref["lever_mm"].items():
        name = f"cam{i}"
        c = cams.get(name)
        if c is None or not c.get("p_CinI_mm"):
            continue
        err = math.dist(c["p_CinI_mm"], lever)
        v.add("ref.lever", name, round(err, 2), f"|Δ| <= {lever_tol_mm} mm",
              PASS if err <= lever_tol_mm else WARN, ref["label"])
    base = ref["baseline_mm"]
    c1 = cams.get("cam1")
    if base is not None and c1 and c1.get("p_CtoC0_mm"):
        b = math.hypot(*c1["p_CtoC0_mm"])
        err = abs(b - base)
        v.add("ref.baseline", "cam1", round(err, 2), f"|Δ| <= {baseline_tol_mm} mm",
              PASS if err <= baseline_tol_mm else WARN, ref["label"])


def assess(path: str, reference: dict | None = None) -> Verdict:
    """Dispatch on file type: .json -> cam-IMU summary, .yaml/.yml -> camera-only."""
    if path.endswith(".json"):
        return assess_cam_imu(path, reference)
    return assess_cam(path)


def format_text(v: Verdict) -> str:
    sym = {PASS: "ok  ", WARN: "WARN", FAIL: "FAIL", None: "?   "}
    lines = [f"verdict: {v.grade}  ({v.kind}; independent check: {v.independent_check})"]
    for c in v.checks:
        note = f"  — {c.note}" if c.note and c.status != PASS else ""
        lines.append(f"  [{sym.get(c.status)}] {c.id:<24} {c.subject:<12} "
                     f"{c.value!s:<12} {c.threshold}{note}")
    return "\n".join(lines)
