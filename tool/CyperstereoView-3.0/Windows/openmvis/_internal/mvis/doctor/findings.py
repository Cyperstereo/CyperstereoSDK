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
"""The finding record of `mvis inspect` / `mvis check` and the two things done with a list of
them: fold the statuses to the worst one and print one finding per line."""
from __future__ import annotations

from dataclasses import dataclass

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Finding:
    id: str
    subject: str
    status: str
    message: str


def worst_status(statuses) -> str:
    """``fail`` if any status is ``fail``, else ``warn`` if any is ``warn``, else ``pass``
    (anything else, such as None for an ungraded check, counts as a pass)."""
    st = set(statuses)
    return FAIL if FAIL in st else WARN if WARN in st else PASS


def format_finding(f) -> str:
    """``[FAIL] <id> <subject>: <message>``, the id padded to 22 characters."""
    return f"[{f.status.upper():<4}] {f.id:<22} {f.subject}: {f.message}"
