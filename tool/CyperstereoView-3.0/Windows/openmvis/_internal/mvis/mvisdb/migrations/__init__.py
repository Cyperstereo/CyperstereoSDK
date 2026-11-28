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
"""Numbered, forward-only schema migrations.

Each migration is a module ``m<NNNN>_<slug>.py`` exposing ``VERSION: int`` and
``up(conn) -> None``. ``apply_migrations`` runs every migration with
``VERSION > schema_meta.schema_version`` in order, inside ONE transaction, then
bumps the version. A file whose version is HIGHER than the code's raises
``SchemaTooNewError`` (handled by the manager before this is called).
"""
from __future__ import annotations

import sqlite3

from mvis.mvisdb.migrations import (
    m0001_initial, m0002_cam_only_base_state, m0003_reproj_stats,
)

#: registration order == application order; append only.
MIGRATIONS: list = [m0001_initial, m0002_cam_only_base_state,
                    m0003_reproj_stats]


def current_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT schema_version FROM schema_meta").fetchone()
    return int(row[0]) if row else 0


def apply_migrations(conn: sqlite3.Connection, migrations: list | None = None) -> int:
    """Apply pending migrations; returns the resulting schema version.
    ``migrations`` is injectable for tests; defaults to the registry."""
    migrations = MIGRATIONS if migrations is None else migrations
    have = current_version(conn)
    pending = [m for m in migrations if m.VERSION > have]
    if not pending:
        return have
    try:
        conn.execute("BEGIN")
        for m in sorted(pending, key=lambda m: m.VERSION):
            m.up(conn)
            conn.execute("UPDATE schema_meta SET schema_version = ?", (m.VERSION,))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return current_version(conn)
