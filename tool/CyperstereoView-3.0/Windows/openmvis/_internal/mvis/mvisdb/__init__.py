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
"""mvisdb — the MVIS database feature, one folder, two public names.

``MVISDatabase`` — the structure (pure Python, zero sqlite).
``DatabaseManager`` (``mvis.mvisdb.manager``) — all operations, the only sqlite path.
"""
from mvis.mvisdb.mvisdatabase import MVISDatabase
from mvis.mvisdb.spec import CaptureSpec

__all__ = ["MVISDatabase", "CaptureSpec"]
# DatabaseManager is imported from mvis.mvisdb.manager on purpose: importing
# mvisdb must never load sqlite3, not even transitively through this module.
