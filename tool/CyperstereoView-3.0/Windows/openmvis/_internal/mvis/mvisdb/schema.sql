-- OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
-- Copyright (C) 2026 Yulin Yang
--
-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, version 3 of the License.
--
-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU General Public License for more details.
--
-- You should have received a copy of the GNU General Public License
-- along with this program.  If not, see <https://www.gnu.org/licenses/>.
--
-- SPDX-FileCopyrightText: 2026 Yulin Yang <contact@openmvis.com>
-- SPDX-License-Identifier: GPL-3.0-only
-- MVIS observation store — schema v1 (design decisions D1-D70).
-- One file per bag: sealed measurement core + per-run overlay + trajectory.
-- Conventions: StrEnums land as TEXT + CHECK; small facts are rows; large
-- written-once arrays are BLOBs (raw little-endian, dtype/shape recorded);
-- floats are REAL (exact float64 round-trip); nothing derived is stored.

PRAGMA foreign_keys = ON;

-- ── meta ────────────────────────────────────────────────────────────────────
CREATE TABLE schema_meta (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL,
    mvis_commit    TEXT,
    created_at     REAL NOT NULL
);

-- ── measurement core (written once by ingest, sealed under content_hash) ────
CREATE TABLE capture (
    id            INTEGER PRIMARY KEY CHECK (id = 1),
    name          TEXT NOT NULL,
    source_type   TEXT CHECK (source_type IN
                    ('images','timestamped_images','euroc','ros1bag','ros2bag')),
    source_path   TEXT,
    clock         TEXT CHECK (clock IN ('header','bag','filename_ns','frame_index')),
    capture_mode  TEXT NOT NULL DEFAULT 'board',      -- comma-set of board|tag|free
    identity_hash TEXT,                               -- CaptureSpec.identity_hash()
    content_hash  TEXT,                               -- set at seal(); NULL = building
    spec_json     TEXT,
    created_at    REAL NOT NULL
);

CREATE TABLE camera (
    camera_id        INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    topic            TEXT,
    shutter          TEXT NOT NULL DEFAULT 'unknown'
                     CHECK (shutter IN ('global','rolling','unknown')),
    readout_s        REAL,
    rolling_mode     INTEGER CHECK (rolling_mode IN (0,1,2) OR rolling_mode IS NULL),
    intrinsics       BLOB,                            -- (4,) f64 fx fy cx cy
    distortion       BLOB,                            -- (N,) f64
    camera_model     TEXT CHECK (camera_model IN
                       ('radtan','equidistant','fov','double_sphere','eucm','ucm',
                        'omni_radtan','kb4')
                       OR camera_model IS NULL),
    res_w            INTEGER,
    res_h            INTEGER,
    calib_provenance TEXT NOT NULL DEFAULT 'unknown'
                     CHECK (calib_provenance IN
                       ('unknown','prior_fixed','prior_initialized','estimated','ground_truth'))
);

CREATE TABLE imu (
    imu_id           INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    topic            TEXT,
    sigma_g          REAL, sigma_bg REAL, sigma_a REAL, sigma_ba REAL,
    frequency_hz     REAL,
    xi24             BLOB,                            -- (24,) f64
    intrinsics_model TEXT CHECK (intrinsics_model IN ('kalibr','rpng','calyx')
                       OR intrinsics_model IS NULL),
    calib_provenance TEXT NOT NULL DEFAULT 'unknown'
                     CHECK (calib_provenance IN
                       ('unknown','prior_fixed','prior_initialized','estimated','ground_truth'))
);

CREATE TABLE board (
    board_id   INTEGER PRIMARY KEY,
    board_type TEXT NOT NULL CHECK (board_type IN ('aprilgrid','aruco_grid','chessboard')),
    tag_family TEXT,                                  -- NULL on chessboard
    rows       INTEGER NOT NULL,
    cols       INTEGER NOT NULL,
    spacing    REAL NOT NULL,                         -- metres, RESOLVED (D15)
    size       REAL NOT NULL,
    start_id   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE board_pt (
    board_id    INTEGER NOT NULL REFERENCES board(board_id),
    board_pt_id INTEGER NOT NULL,
    tag_id      INTEGER NOT NULL,
    corner_id   INTEGER CHECK (corner_id IN (0,1,2,3) OR corner_id IS NULL),
    x REAL NOT NULL, y REAL NOT NULL, z REAL NOT NULL,
    PRIMARY KEY (board_id, board_pt_id)
) WITHOUT ROWID;

CREATE TABLE rig (
    id               INTEGER PRIMARY KEY CHECK (id = 1),
    base_camera_id   INTEGER NOT NULL,
    base_imu_id      INTEGER,                          -- NULL = no IMU designated
    sync_kind        TEXT,
    sync_tolerance_s REAL
);

CREATE TABLE stereo_pair (
    cam_a   INTEGER NOT NULL REFERENCES camera(camera_id),
    cam_b   INTEGER NOT NULL REFERENCES camera(camera_id),
    implied INTEGER NOT NULL DEFAULT 0,               -- cam_plan 2-camera implicit pair (D-stereo)
    PRIMARY KEY (cam_a, cam_b)
) WITHOUT ROWID;

CREATE TABLE frame (
    frame_id  INTEGER PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES camera(camera_id),
    timestamp REAL NOT NULL,
    UNIQUE (camera_id, timestamp)
);

CREATE TABLE board_detection (
    frame_id INTEGER NOT NULL REFERENCES frame(frame_id),
    board_id INTEGER NOT NULL REFERENCES board(board_id),
    PRIMARY KEY (frame_id, board_id)
) WITHOUT ROWID;
-- counts / detect_status are DERIVED (D24) — never columns here.

-- One blob per (frame, board): the written-once measurement arrays.
CREATE TABLE detection (
    frame_id  INTEGER NOT NULL,
    board_id  INTEGER NOT NULL,
    n         INTEGER NOT NULL,
    point_id  BLOB NOT NULL,                          -- (n,)  i64
    uv        BLOB NOT NULL,                          -- (n,2) f64, exact
    emission  BLOB NOT NULL,                          -- (n,)  i64 detector_emission_order
    PRIMARY KEY (frame_id, board_id),
    FOREIGN KEY (frame_id, board_id) REFERENCES board_detection(frame_id, board_id)
) WITHOUT ROWID;

CREATE TABLE imu_meas (
    imu_id INTEGER PRIMARY KEY REFERENCES imu(imu_id),
    n      INTEGER NOT NULL,
    t      BLOB NOT NULL,                             -- (n,)  f64
    gyro   BLOB NOT NULL,                             -- (n,3) f64
    accel  BLOB NOT NULL                              -- (n,3) f64
);

-- ── run overlay (everything estimated; keyed by run_id; disposable) ─────────
CREATE TABLE run (
    run_id              INTEGER PRIMARY KEY,
    calib_choice        TEXT NOT NULL CHECK (calib_choice IN ('cam_only','cam_imu')),
    origin              TEXT NOT NULL DEFAULT 'solved'
                        CHECK (origin IN ('solved','ground_truth','imported')),   -- D70
    description         TEXT NOT NULL DEFAULT '',                                 -- D54
    solve_mode          TEXT,
    variant             TEXT,
    collection          TEXT,
    state_hz            REAL,
    window_lo           REAL, window_hi REAL,
    min_obs             INTEGER,
    config_json         TEXT,
    config_sha256       TEXT,
    mvis_commit         TEXT,
    machine_id          INTEGER,
    status              TEXT NOT NULL DEFAULT 'running'
                        CHECK (status IN ('running','ok','failed')),
    started_at          REAL NOT NULL DEFAULT 0,
    wall_s              REAL,
    residual_promote_px REAL,
    sensor_policy_json  TEXT,                         -- D63 block, schema v1 as JSON
    factor_hz_json      TEXT,
    overall_rms         REAL                          -- I5 (v3): headline stat
);

-- Sparse: only non-default rows are written (save diffs the inline flags, D53).
CREATE TABLE frame_status (
    run_id          INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    frame_id        INTEGER NOT NULL REFERENCES frame(frame_id),
    is_motion_still INTEGER,                          -- NULL = unknown
    is_selected     INTEGER NOT NULL DEFAULT 1,
    exclude_reason  TEXT,
    PRIMARY KEY (run_id, frame_id)
) WITHOUT ROWID;

CREATE TABLE board_detection_status (
    run_id      INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    frame_id    INTEGER NOT NULL,
    board_id    INTEGER NOT NULL,
    pose        BLOB,                                 -- (4,4) f64 T_cam_to_board
    pose_source TEXT CHECK (pose_source IN
                  ('pnp','refined_per_camera','refined_cam_only','refined_cam_imu')
                  OR pose_source IS NULL),            -- D61 ladder enforced by the writer
    is_valid    INTEGER NOT NULL DEFAULT 1,
    is_selected INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (run_id, frame_id, board_id)
) WITHOUT ROWID;

CREATE TABLE detection_status (
    run_id          INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    frame_id        INTEGER NOT NULL,
    board_id        INTEGER NOT NULL,
    point_id        INTEGER NOT NULL,
    is_valid        INTEGER NOT NULL DEFAULT 1,
    is_selected     INTEGER NOT NULL DEFAULT 1,
    exclude_reason  TEXT,
    residual_px     REAL,
    residual_source TEXT CHECK (residual_source IN
                      ('pnp','refined_per_camera','refined_cam_only','refined_cam_imu')
                      OR residual_source IS NULL),
    PRIMARY KEY (run_id, frame_id, board_id, point_id)
) WITHOUT ROWID;

CREATE TABLE result_camera (
    run_id       INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    camera_id    INTEGER NOT NULL REFERENCES camera(camera_id),
    intrinsics   BLOB NOT NULL,                       -- (4,) f64
    distortion   BLOB NOT NULL,
    camera_model TEXT NOT NULL,
    readout_s    REAL,
    provenance   TEXT NOT NULL CHECK (provenance IN
                   ('unknown','prior_fixed','prior_initialized','estimated','ground_truth')),
    sigmas       BLOB,
    -- I5: per-camera reprojection stats (nullable; solver output)
    rms          REAL, mean_err REAL, median_err REAL,
    p90          REAL, p95 REAL, n_obs INTEGER,
    PRIMARY KEY (run_id, camera_id)
) WITHOUT ROWID;

CREATE TABLE result_imu (
    run_id     INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    imu_id     INTEGER NOT NULL REFERENCES imu(imu_id),
    xi24       BLOB NOT NULL,
    model      TEXT NOT NULL CHECK (model IN ('kalibr','rpng','calyx')),
    provenance TEXT NOT NULL CHECK (provenance IN
                 ('unknown','prior_fixed','prior_initialized','estimated','ground_truth')),
    sigmas     BLOB,
    PRIMARY KEY (run_id, imu_id)
) WITHOUT ROWID;

CREATE TABLE extrinsic (
    run_id     INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    from_type  TEXT NOT NULL CHECK (from_type IN ('cam','imu')),
    from_id    INTEGER NOT NULL,
    to_type    TEXT NOT NULL CHECK (to_type IN ('cam','imu')),
    to_id      INTEGER NOT NULL,
    T          BLOB NOT NULL,                         -- (4,4) f64 T_from_to_to
    td_s       REAL,                                  -- LIVE for cam-only too (D61 review)
    sigma6     BLOB,
    provenance TEXT NOT NULL CHECK (provenance IN
                 ('unknown','prior_fixed','prior_initialized','estimated','ground_truth')),
    PRIMARY KEY (run_id, from_type, from_id, to_type, to_id)
) WITHOUT ROWID;

-- estimate integrity record: which reference stream + state rate this run used
CREATE TABLE estimate_meta (
    run_id         INTEGER PRIMARY KEY REFERENCES run(run_id) ON DELETE CASCADE,
    reference_type TEXT NOT NULL CHECK (reference_type IN ('cam','imu')),
    reference_id   INTEGER NOT NULL,
    state_hz       REAL
);

-- ── trajectory (L4) — populated for cam_imu; blobs like detection ───────────
CREATE TABLE state_knot (
    run_id    INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    k         INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    PRIMARY KEY (run_id, k)
) WITHOUT ROWID;

CREATE TABLE state_var (
    run_id      INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    var_type    TEXT NOT NULL CHECK (var_type IN
                  ('T_imu_to_world','v','b_g','b_a','T_board_to_world','landmark_xyz',
                   'T_cam_to_board')),
    sensor_type TEXT NOT NULL CHECK (sensor_type IN ('cam','imu','board','world')),
    sensor_id   INTEGER NOT NULL,
    dtype       TEXT NOT NULL,
    shape       TEXT NOT NULL,                        -- json list
    data        BLOB NOT NULL,
    PRIMARY KEY (run_id, var_type, sensor_type, sensor_id)
) WITHOUT ROWID;

CREATE TABLE frame_bracket (
    run_id      INTEGER NOT NULL REFERENCES run(run_id) ON DELETE CASCADE,
    frame_id    INTEGER NOT NULL,
    knot_lo     INTEGER, knot_hi INTEGER,
    lam         REAL,
    ok          INTEGER NOT NULL DEFAULT 1,
    drop_reason TEXT,
    PRIMARY KEY (run_id, frame_id)
) WITHOUT ROWID;

CREATE TABLE world_frame (
    run_id      INTEGER PRIMARY KEY REFERENCES run(run_id) ON DELETE CASCADE,
    kind        TEXT NOT NULL DEFAULT 'gravity_aligned',
    g_magnitude REAL NOT NULL DEFAULT 9.80665,
    yaw_gauge   REAL
);
