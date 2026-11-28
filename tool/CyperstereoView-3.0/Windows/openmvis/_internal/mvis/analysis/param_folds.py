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
"""The Summary's "Calibration parameters by sensor": one closed fold per sensor, its key numbers on
the closed line, every calibrated parameter inside (value ± 1σ where the solve has σ), and a Copy
button for that sensor's block of the result YAML. Shared by the camera-IMU and camera-only reports;
each builds the groups from the per-sensor rows its other sections already show, so the fold can't
disagree with them.

Number formats, one per unit (also used by the reports' tables): px 3 decimals (focal length and
principal point 2), mm 2, ms 3, rad 6, scale factors 6, dimensionless coefficients 6.
"""
from __future__ import annotations

import html as _html
import math

FMT = {"px": "{:.3f}", "px2": "{:.2f}", "mm": "{:.2f}", "ms": "{:+.3f}", "rad": "{:+.6f}",
       "rads": "{:+.6f}", "mps2": "{:+.5f}", "scale": "{:.6f}", "coef": "{:+.6f}", "deg": "{:.0f}"}
UNIT = {"px": "px", "px2": "px", "mm": "mm", "ms": "ms", "rad": "rad", "rads": "rad/s", "mps2": "m/s²",
        "scale": "", "coef": "", "deg": "°"}


def _ok(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def num(v, kind: str) -> str:
    """One number in its unit's format, or an em dash when absent."""
    return FMT[kind].format(float(v)) if _ok(v) else "—"


def val(v, kind: str, sigma=None) -> str:
    """A value with its unit, and ± 1σ when the solve has a non-zero σ."""
    if not _ok(v):
        return "—"
    s = num(v, kind)
    if _shows(sigma, kind):
        s += f" <span class='pm'>± {_sig(sigma, kind)}</span>"
    u = UNIT[kind]
    return s + (f" {u}" if u and kind != "deg" else u)


def _sig(sigma, kind: str) -> str:
    return FMT[kind].replace("+", "").format(abs(float(sigma)))


def _shows(sigma, kind: str) -> bool:
    """A σ worth printing: present and non-zero at the value's own precision (a pinned parameter's
    marginal comes back as ~1e-9, which would print as '± 0.00')."""
    return _ok(sigma) and any(ch in "123456789" for ch in _sig(sigma, kind))


def vec(vs, kind: str, sigmas=None) -> str:
    """A 3-vector (x, y, z) in one unit as three aligned cells, each with its σ when present."""
    if vs is None:
        return "—"
    sig = list(sigmas) if sigmas is not None else [None] * len(vs)
    cells = []
    for v, s in zip(vs, sig):
        sg = f"<small class='pm'>± {_sig(s, kind)}</small>" if _shows(s, kind) else ""
        cells.append(f"<span>{num(v, kind)}{sg}</span>")
    return f"<span class='v3'>{''.join(cells)}</span>"      # unit in the row label (see unit_label)


def unit_label(label: str, kind: str) -> str:
    """A row label with its unit, for rows whose value is a vec()."""
    u = UNIT[kind]
    return f"{label} ({u})" if u else label


def group(title: str, rows) -> tuple:
    """A titled block of (label, value-html) rows; rows whose value is None are left out."""
    return title, [(k, v) for k, v in rows if v is not None]


def sensor_fold(head: str, key: str, groups, yaml_text: str | None = None) -> str:
    """One sensor: `head` (e.g. 'cam0') and `key` (its key numbers) on the closed line; inside, each
    group as a two-column table, then the sensor's YAML with a Copy button."""
    body = []
    for title, rows in groups:
        rows = [(k, v) for k, v in rows if v is not None]
        if not rows:
            continue
        body.append(f"<div class='param-group'><h5>{_html.escape(title)}</h5><table class='kv'>"
                    + "".join(f"<tr><th>{_html.escape(k)}</th><td class='num'>{v}</td></tr>" for k, v in rows)
                    + "</table></div>")
    if yaml_text:
        body.append("<div class='param-group wide'><h5>YAML</h5><div class='copy-wrap'>"
                    "<button type='button' class='copy-btn'>Copy</button>"
                    f"<pre class='copy-pre'>{_html.escape(yaml_text)}</pre></div></div>")
    return (f"<details class='appendix param-fold'><summary><b>{_html.escape(head)}</b>"
            f"<span class='param-key'>{key}</span></summary>"
            f"<div class='appendix-content param-grid'>{''.join(body)}</div></details>")


def folds_block(folds) -> str:
    """The Summary block: heading, an Expand all toggle (falls back to plain folds without JS) and the folds."""
    if not folds:
        return ""
    return ("<div class='param-folds'><h3>Calibration parameters by sensor "
            "<button type='button' class='fold-all'>Expand all</button></h3>"
            + "".join(folds)
            + "<p class='note'>Value ± 1σ where the solve has σ. Rotations as log(R) in rad.</p></div>")


def yaml_block(data) -> str | None:
    """One sensor's block of the result YAML as text, for the Copy button."""
    if data is None:
        return None
    import yaml
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=100)
