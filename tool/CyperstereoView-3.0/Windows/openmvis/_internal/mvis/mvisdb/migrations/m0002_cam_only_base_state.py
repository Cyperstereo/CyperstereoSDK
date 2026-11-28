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
"""v2 — widen state_var.var_type CHECK with 'T_cam_to_board' so a cam-only
run's base state (reference camera w.r.t. the board) can persist.
SQLite cannot alter a CHECK, so the table is rebuilt in place; the runner
wraps this in one transaction."""

VERSION = 2

_DDL = """
CREATE TABLE state_var_v2 (
    run_id      INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    var_type    TEXT NOT NULL CHECK (var_type IN
                  ('T_imu_to_world','v','b_g','b_a','T_board_to_world','landmark_xyz',
                   'T_cam_to_board')),
    sensor_type TEXT NOT NULL CHECK (sensor_type IN ('cam','imu','board','world')),
    sensor_id   INTEGER NOT NULL,
    dtype       TEXT NOT NULL,
    shape       TEXT NOT NULL,
    data        BLOB NOT NULL,
    PRIMARY KEY (run_id, var_type, sensor_type, sensor_id)
) WITHOUT ROWID
"""


def up(conn) -> None:
    conn.execute(_DDL)
    conn.execute("INSERT INTO state_var_v2 SELECT * FROM state_var")
    conn.execute("DROP TABLE state_var")
    conn.execute("ALTER TABLE state_var_v2 RENAME TO state_var")
