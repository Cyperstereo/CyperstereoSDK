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
"""Read the calibration process out of an older cam-IMU report (the Plotly
"Trajectory & scene" figure, before `analysis.views`) as a v2 scene spec.

For runs solved before the canvas views: the sensor pages can show the new
player without a re-solve. Camera frames are never copied: the result is what
`spec.without_frames` would give, plus the drawn board edge per camera.

Run as ``python -m mvis.analysis.views.legacy <run_dir> [...] [--force]`` to write
`cam_imu_views.json` beside each older run from its `cam_imu_report.html`.
"""
from __future__ import annotations

import base64
import json

import numpy as np

from mvis.analysis.views.geometry import project
from mvis.analysis.views.spec import VERSION, _r, without_frames
from mvis.calibration.factors.camera_projection import k_cal_vec


def _arr(a) -> np.ndarray:
    """A Plotly array: a list, or packed {"bdata", "dtype"}."""
    if isinstance(a, dict):
        return np.frombuffer(base64.b64decode(a["bdata"]), dtype=a["dtype"]).astype(float)
    return np.array([np.nan if x is None else x for x in a], float)


def _figure(html: str):
    """(progress spec, Plotly traces) of the report's last progress figure."""
    i = html.rindex('mvisProgress("')
    j = html.index(",", i)
    pid = html[i + 14:j - 1]
    dec = json.JSONDecoder()
    spec, _ = dec.raw_decode(html[j + 1:])
    k = html.index(f'"{pid}",', html.index("Plotly.newPlot("))
    data, _ = dec.raw_decode(html[html.index("[", k):])
    return spec, data


def _trace(data, prefix):
    tr = next((t for t in data if str(t.get("name", "")).startswith(prefix)), None)
    if tr is None:
        return None
    p = np.c_[_arr(tr["x"]), _arr(tr["y"]), _arr(tr["z"])]
    return p[~np.isnan(p).any(axis=1)]


def scene_from_report(html: str, cams: list[dict]) -> dict | None:
    """v2 scene spec from an older report's HTML. `cams` gives per camera
    `model`, `intr` (full vector) and `size`, from the run's results. None
    when the report has no process figure."""
    if 'mvisProgress("' not in html:
        return None
    old, data = _figure(html)
    xyz = np.asarray(old["xyz"], float)
    n = len(xyz)
    R = np.asarray(old["R"], float).reshape(n, 3, 3) if old.get("R") else np.tile(np.eye(3), (n, 1, 1))
    t = old.get("t")
    spec = {"v": VERSION, "t": t, "unit": "s" if t is not None else "step", "xyz": _r(xyz), "R": _r(R.reshape(n, 9)), "cams": []}
    pts, outline = _trace(data, "Board corners"), _trace(data, "Board Outline")
    if pts is not None and outline is not None and len(outline) >= 4:
        ring = outline[:4]
        nrm = np.cross(ring[1] - ring[0], ring[3] - ring[0])
        nrm = nrm / (np.linalg.norm(nrm) or 1.0)
        spec["board"] = {"pts": _r(pts), "outline": _r(ring), "normal": _r(nrm), "center": _r(ring.mean(axis=0))}
    else:
        lm = _trace(data, "landmarks")
        spec["landmarks"] = _r(lm if lm is not None else np.zeros((0, 3)), 4)
    ring_W = np.asarray(spec["board"]["outline"]) if "board" in spec else None
    for c, v in enumerate(old.get("views", [])):
        info = cams[c] if c < len(cams) else {}
        intr = np.asarray(info.get("intr", [1, 1, 0, 0]), float)
        model = info.get("model", "radtan")
        cam = {"name": v.get("name", f"cam{c}"), "color": v.get("color", "#2563EB"), "k": _r(intr[:4], 4),
               "size": [int(x) for x in info.get("size", (640, 480))]}
        if v.get("cam") is not None:
            cam["R"], cam["p"] = _r(v["cam"][:9], 6), _r(v["cam"][9:12], 6)
        else:
            cam["R"], cam["p"] = _r(np.eye(3).reshape(9)), [0.0, 0.0, 0.0]
        p = v.get("panel") or {}
        for k in ("uvb", "uvo", "cov", "n"):
            if k in p:
                cam[k] = p[k]
        if v.get("ids") is not None:
            cam["ids"] = v["ids"]
        if ring_W is not None:
            try:
                cv = k_cal_vec(model, intr[:4], intr[4:])
            except Exception:
                cv = None
            if cv is not None:
                Rc, pc = np.asarray(cam["R"], float).reshape(3, 3), np.asarray(cam["p"], float)
                out = []
                for k in range(n):
                    Rw, tw = R[k] @ Rc, R[k] @ pc + xyz[k]
                    Xc = (Rw.T @ (ring_W - tw).T).T
                    q = [project(model, cv, x) if x[2] > 1e-3 else None for x in Xc]
                    out.append(None if any(u is None for u in q) else _r(np.array(q), 1))
                cam["outline"] = out
        spec["cams"].append(cam)
    return without_frames(spec)


def backfill(run_dir, force: bool = False) -> str:
    """Write `cam_imu_views.json` beside an older run (scene only; no frames) from
    its report, unless one is there. Returns what it did."""
    from pathlib import Path

    import yaml
    d = Path(run_dir)
    out = d / "cam_imu_views.json"
    if out.exists() and not force:
        return f"kept {out}"
    report = d / "cam_imu_report.html"
    if not report.exists():
        return f"no report in {d}"
    run = json.loads((d / "cam_imu.json").read_text())
    cal = yaml.safe_load((d / "cam_imu_calib.yaml").read_text())["cameras"]
    cams = [{"model": (c.get("distortion") or {}).get("model") or c.get("model"), "intr": r["intr"], "size": c["image_size"]}
            for c, r in zip(cal, run["cams"])]
    scene = scene_from_report(report.read_text(), cams)
    if scene is None:
        return f"no process figure in {report}"
    out.write_text(json.dumps({"scene": scene, "rig": None, "from": "legacy report"}, separators=(",", ":"), allow_nan=False))
    return f"wrote {out} ({out.stat().st_size // 1024} kB)"


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a != "--force"]
    for a in args:
        print(backfill(a, force="--force" in sys.argv))

