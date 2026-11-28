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
"""Round-trip report + driver: compare a FORWARD cam-IMU calibration to its RECALIBRATION on
the apply-calib-corrected bag. A correct correction drives the IMU intrinsics/biases to
identity/zero while leaving the geometry (extrinsics, stereo baseline) unchanged; the time
offset is observability-dependent and reported as a warning, not a failure.

``roundtrip_report`` takes two ``cam_imu.json`` dicts. ``run_roundtrip`` orchestrates the full
calibrate -> apply-calib -> recalibrate -> report pipeline for a ROS1-bag config and backs
the ``mvis roundtrip`` CLI subcommand.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import yaml

from mvis.recalib.calib_spec import load_rig_calib
from mvis.recalib.result_metrics import imu_rows, lever_mm, stereo_baseline


def roundtrip_report(forward: dict, recal: dict, *, scale_tol: float = 0.01,
                     bias_tol: float = 0.02, baseline_tol_mm: float = 2.0,
                     td_warn_ms: float = 2.0) -> dict:
    """Compare two ``cam_imu.json`` dicts. Returns ``{ok, checks}`` where ``checks`` is a list
    of ``{name, before, after, expect, pass, kind, warn}``. ``ok`` is the AND of all
    non-warn, non-info checks: intrinsics→identity, biases→0, extrinsics + baseline preserved.
    ``kind='td'`` checks are warnings (td residual is dataset-observability-dependent).

    Tolerances (chosen from observed real-data round-trips, e.g. MIC/3cams):
      scale_tol=0.01       Dw/Da within 1% of unit scale is a clean collapse (forward scales run 2-7%).
      bias_tol=0.02        residual gyro(rad/s)/accel(m/s^2) bias below this is effectively removed.
      baseline_tol_mm=2.0  stereo baseline / lever drift above 2 mm is a real geometry change, not noise.
      td_warn_ms=2.0       td residual above this only WARNs — time offset can be weakly observable."""
    checks = []

    def add(name, before, after, expect, ok, kind="collapse", warn=False):
        checks.append({"name": name, "before": before, "after": after, "expect": expect,
                       "pass": bool(ok), "kind": kind, "warn": warn})

    def _r(v):                          # round a vector for display, pass None through
        return np.round(v, 4).tolist() if v is not None else None

    # --- IMU intrinsics -> identity, biases -> 0 (base + aux), aligned by NAME ---
    recal_imus = {r["name"]: r for r in imu_rows(recal)}
    for fr in imu_rows(forward):
        rr = recal_imus.get(fr["name"])
        if rr is None:
            continue
        n = fr["name"]
        if rr["Dw"] is not None:
            add(f"{n} Dw->I", _r(fr["Dw"]), _r(rr["Dw"]),
                f"|.-1|<={scale_tol}", all(abs(x - 1) <= scale_tol for x in rr["Dw"]))
        if rr["Da"] is not None:
            add(f"{n} Da->I", _r(fr["Da"]), _r(rr["Da"]),
                f"|.-1|<={scale_tol}", all(abs(x - 1) <= scale_tol for x in rr["Da"]))
        if rr["b_g"] is not None:
            add(f"{n} b_g->0", _r(fr["b_g"]), _r(rr["b_g"]),
                f"|.|<={bias_tol}", all(abs(x) <= bias_tol for x in rr["b_g"]))
        if rr["b_a"] is not None:
            add(f"{n} b_a->0", _r(fr["b_a"]), _r(rr["b_a"]),
                f"|.|<={bias_tol}", all(abs(x) <= bias_tol for x in rr["b_a"]))

    # --- each IMU's gyro<->accel time offset -> 0 after the accel time shift ---
    tga = [("imu0", forward.get("t_ga_ms"), recal.get("t_ga_ms"))]
    recal_aux = {m.get("name"): m for m in recal.get("imus", [])}
    tga += [(m.get("name"), m.get("t_ga_ms"), recal_aux.get(m.get("name"), {}).get("t_ga_ms"))
            for m in forward.get("imus", [])]
    for n, ft, rt in tga:
        if ft is not None and rt is not None:
            add(f"{n} t_ga(gyro->accel)->0 (ms)", round(ft, 3), round(rt, 3), "->0",
                abs(rt) <= td_warn_ms, kind="td", warn=True)

    # --- cameras: lever preserved + td residual (warn) ---
    oc, rc = forward.get("cams", []), recal.get("cams", [])
    for o, r in zip(oc, rc):
        dl = lever_mm(np.subtract(o["p_CinI_mm"], r["p_CinI_mm"]))
        add(f"{o['name']} lever preserved", round(lever_mm(o["p_CinI_mm"]), 1),
            round(lever_mm(r["p_CinI_mm"]), 1), f"drift<={baseline_tol_mm}mm",
            dl <= baseline_tol_mm, kind="preserve")
        add(f"{o['name']} td residual", round(o["td_ms"], 2), round(r["td_ms"], 2), "->0",
            abs(r["td_ms"]) <= td_warn_ms, kind="td", warn=True)

    # --- stereo baseline preserved ---
    ob, rb = stereo_baseline(oc), stereo_baseline(rc)
    if ob is not None and rb is not None:
        add("stereo baseline preserved", round(ob, 2), round(rb, 2),
            f"drift<={baseline_tol_mm}mm", abs(ob - rb) <= baseline_tol_mm, kind="preserve")

    add("reproj RMS", round(forward.get("reproj_rms_px", float("nan")), 3),
        round(recal.get("reproj_rms_px", float("nan")), 3), "~same", True, kind="info")

    ok = all(c["pass"] for c in checks if not c["warn"] and c["kind"] != "info")
    return {"ok": ok, "checks": checks}


def format_roundtrip_report(rep: dict) -> str:
    """Human-readable before->after table with a top-line PASS/FAIL."""
    lines = ["ROUND-TRIP: %s" % ("PASS ✅" if rep["ok"] else "FAIL ❌"),
             "  %-26s %-22s %-22s %-14s %s" % ("check", "before", "after", "expect", "")]
    for c in rep["checks"]:
        mark = "warn" if c["warn"] else ("ok" if c["pass"] else "FAIL")
        lines.append("  %-26s %-22s %-22s %-14s %s"
                     % (c["name"], str(c["before"]), str(c["after"]), c["expect"], mark))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver: full calibrate -> apply-calib -> recalibrate -> report pipeline
# ---------------------------------------------------------------------------

def _calib_yaml(result_json: str) -> str:
    hits = glob.glob(os.path.join(os.path.dirname(result_json), "*calib*.yaml"))
    return hits[0] if hits else ""


def _with_biases(result_json: str) -> dict:
    """Load a cam_imu.json + inject base AND aux IMU b_g/b_a, sourced from its calib YAML via
    ``load_rig_calib`` (the json itself carries no biases). Aux biases are matched to ``imus[]``
    by name. A calib YAML ``load_rig_calib`` can't parse (e.g. a model-less ``xi``) is skipped
    rather than crashing the report."""
    with open(result_json) as f:
        d = json.load(f)
    cy = _calib_yaml(result_json)
    if cy and os.path.exists(cy):
        try:
            rig = load_rig_calib(cy)
        except ValueError:
            return d
        d["b_g"] = rig.base_imu.b_g.tolist()
        d["b_a"] = rig.base_imu.b_a.tolist()
        if rig.base_imu.t_ga:                    # summary jsons predate t_ga_ms
            d.setdefault("t_ga_ms", rig.base_imu.t_ga * 1e3)
        aux = {a.name: a for a in rig.aux_imus}
        for m in d.get("imus", []):
            a = aux.get(m.get("name"))
            if a is not None:
                m["b_g"] = a.b_g.tolist()
                m["b_a"] = a.b_a.tolist()
    return d


def apply_calib_command(rcpath: str) -> list[str]:
    """The `mvis apply-calib` step as a subprocess of this interpreter (as the regression adapters run MVIS)."""
    return [sys.executable, "-m", "mvis.cli.main", "apply-calib", "-c", rcpath]


def run_roundtrip(config_path: str, *, out: str | None = None, keep_corrected: bool = False,
                  keep_bias: bool = False, bag_override: str | None = None) -> dict:
    """Run the full round-trip for a ROS1-bag cam-IMU config and return the report dict
    (``{ok, checks}``). Forward-calibrates, applies the calib to correct the bag, recalibrates
    the corrected bag, and compares. Writes ``<out>/report.{txt,json}``. Deletes the corrected
    bag unless ``keep_corrected``. Raises ``ValueError`` on a non-ros1 config or low disk."""
    from mvis.cli.main import _override_source, calibrate_cam_imu  # lazy: avoid import cycle
    from mvis.core.config import CamImuRunConfig

    cfg = CamImuRunConfig.from_yaml(config_path)
    if bag_override:
        _override_source(cfg, bag_override)     # let a euroc forward config round-trip from a bag
    if cfg.input.format != "ros1":
        raise ValueError("roundtrip needs a ROS1-bag config (apply-calib is ros1-only). "
                         "Convert a EuRoC folder to a bag first (a catalog dataset: `mvis "
                         "datasets convert --dataset <key>`; any mav0 folder: "
                         "mvis.datasets.converters.euroc_folder_to_rosbag."
                         "convert_euroc_folder_to_rosbag(src, out.bag)) and pass "
                         "--bag <converted.bag>.")
    bag = cfg.input.path
    name = os.path.splitext(os.path.basename(config_path))[0]
    out = out or f"results/roundtrip/{name}"
    os.makedirs(f"{out}/forward", exist_ok=True)
    os.makedirs(f"{out}/recal", exist_ok=True)
    corrected = os.path.join(os.path.dirname(bag),
                             os.path.splitext(os.path.basename(bag))[0] + ".roundtrip_corrected.bag")

    print("== [1/3] forward calibrate ==", flush=True)
    cfg.output.path = f"{out}/forward/cam_imu.json"
    calibrate_cam_imu(cfg)
    fcalib = _calib_yaml(cfg.output.path)

    free = shutil.disk_usage(os.path.dirname(os.path.abspath(bag))).free
    need = os.path.getsize(bag)
    if free < need:
        raise ValueError(f"not enough disk for the corrected bag (~{need/1e9:.0f} GB needed, "
                         f"{free/1e9:.0f} GB free) — free space or pass out= on a bigger disk.")

    print("== [2/3] apply-calib (correct the bag) ==", flush=True)
    rcpath = f"{out}/recalib.yaml"
    with open(rcpath, "w") as f:
        yaml.safe_dump({"calib": fcalib, "input": bag, "output": corrected,
                        "remove_bias": not keep_bias}, f)
    # this interpreter's MVIS, not whatever `mvis` is on PATH (unset when mvis is run by its full path)
    subprocess.run(apply_calib_command(rcpath), check=True)

    print("== [3/3] recalibrate the corrected bag ==", flush=True)
    cfg2 = CamImuRunConfig.from_yaml(config_path)
    _override_source(cfg2, corrected)     # corrected bag is ros1; distinct name -> fresh cache
    cfg2.output.path = f"{out}/recal/cam_imu.json"
    calibrate_cam_imu(cfg2)

    rep = roundtrip_report(_with_biases(f"{out}/forward/cam_imu.json"),
                           _with_biases(f"{out}/recal/cam_imu.json"))
    txt = format_roundtrip_report(rep)
    print("\n" + txt)
    with open(f"{out}/report.txt", "w") as f:
        f.write(txt)
    with open(f"{out}/report.json", "w") as f:
        json.dump(rep, f, indent=1)

    if not keep_corrected and os.path.exists(corrected):
        os.remove(corrected)
        print(f"\nremoved {corrected}")
    print(f"report -> {out}/report.txt")
    return rep
