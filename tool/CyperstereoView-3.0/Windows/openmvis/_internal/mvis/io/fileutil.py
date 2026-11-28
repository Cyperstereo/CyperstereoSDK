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
"""Small file helpers shared across packages: a file's sha256 and a JSON writer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHUNK_BYTES = 1 << 20


def compute_sha256(path) -> str:
    """Hex sha256 of a file's bytes, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data, *, indent=1, sort_keys=False) -> Path:
    """``json.dump(data, indent=, sort_keys=)`` to ``path`` (no trailing newline), creating
    the parent folders. Returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)
    return out
