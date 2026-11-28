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
"""The cam-IMU `reference:` block (factory, CAD or vendor values), read one way.

A config's `reference:` names per-camera values by `camN`, by the index `N` or by
its string `'N'`; `baseline_mm` is a scalar length or a 3-vector. The verdict, the
report's vs-reference section and the rig ghost all read it through this module,
so a key form one of them accepts is accepted by all (before 2026-11 the report
dropped `cam0` keys and the verdict dropped `0` keys and vector baselines).
Standard library only: `mvis.analysis.verdict` imports it.
"""
from __future__ import annotations

import math
import re
import warnings

_CAM_KEY = re.compile(r"(?:cam)?(\d+)\Z")


def cam_index(key) -> int | None:
    """Camera index of a reference key: `cam1`, `1` or `'1'` -> 1; else None."""
    if isinstance(key, bool):
        return None
    if isinstance(key, int):
        return key if key >= 0 else None
    m = _CAM_KEY.match(str(key).strip())
    return int(m.group(1)) if m else None


def by_camera(values: dict | None) -> dict[int, object]:
    """{camera index: value} of a per-camera reference map, in the map's order.
    Keys that name no camera are dropped; when two keys name one camera (`cam0`
    and `0`) the first wins, with a warning if their values differ. (Before 2026-11
    the rig ghost preferred `N`, then `'N'`, then `camN`, whatever the order.)"""
    out: dict[int, object] = {}
    for k, v in (values or {}).items():
        i = cam_index(k)
        if i is None:
            continue
        if i not in out:
            out[i] = v
        elif not _same(out[i], v):
            warnings.warn(f"reference: two keys name camera {i} with different values; "
                          f"the first in the map ({out[i]!r}) is used, {k!r}: {v!r} is ignored",
                          stacklevel=2)
    return out


def _same(a, b) -> bool:
    try:
        return bool(a == b)
    except ValueError:                    # numpy arrays: elementwise ==
        return bool((a == b).all()) if getattr(a, "shape", None) == getattr(b, "shape", None) else False


def camera_value(values: dict | None, i: int):
    """The reference value for camera `i` (any key form), or None."""
    return by_camera(values).get(i)


def baseline_norm_mm(value) -> float | None:
    """The reference stereo baseline length in mm: a scalar as is, a vector by
    its norm; None when unset."""
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    try:                                  # list, tuple or numpy array
        seq = [float(x) for x in value]
    except TypeError:                     # numpy scalar
        return float(value)
    return math.hypot(*seq)


def reference_deltas(reference: dict | None) -> dict | None:
    """The comparable parts of a `reference:` block, normalised: `label`,
    `lever_mm` as {camera index: value}, and `baseline_mm` as a length in mm
    (or None). None when there is no reference. Reference rotations are read per
    camera by the report (cam_imu_report._reference_angle_rad)."""
    if not reference:
        return None
    return {"label": reference.get("label", "reference"),
            "lever_mm": by_camera(reference.get("lever_mm")),
            "baseline_mm": baseline_norm_mm(reference.get("baseline_mm"))}
