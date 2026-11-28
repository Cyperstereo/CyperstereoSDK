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
"""DatabaseManager — ALL operations on ``.mvisdb`` files; the ONLY sqlite path
Surface: save (= write core once + ATTACH a run), load (= fold one run's
overlay back into inline flags), state (direct cheap estimate read),
frame_arrays, runs. Cross-bag indexing and ingest orchestration are not here.

Write discipline: a bag is built as ``<path>.building`` under an flock and
atomically renamed on seal, so a crashed or concurrent build is never readable
as valid. The measurement core is written ONCE, keyed by
``observation_fingerprint``; every later save against the same file must match
it bit for bit or it is a DIFFERENT capture. A ground_truth run is
write-once. The pose-source ladder is enforced again at the row level here.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # Windows has no fcntl; msvcrt locks the same .lock file
    fcntl = None

import numpy as np

from mvis.mvisdb.mvisdatabase import (
    Board, BoardPt, BoardType, CalibChoice, CalibProvenance, CameraInfo,
    CameraIntrinsicsState, CameraModel, CameraStream, CamImuBaseState,
    ReprojStats,
    CamOnlyBaseState, Corner,
    EstimateStates, ImuInfo, ImuIntrinsicsModel, ImuIntrinsicsState,
    ImuMeasurements, ImuStream, MVISDatabase, PoseCamToBoard, PoseSource,
    RelativeCalib, Residual, Rig, Run, RunOrigin, SensorState, SensorType,
    ShutterType, TagDetection, TagFamily, VarType, WorldFrame,
    observation_fingerprint,
)

from mvis.mvisdb.migrations import apply_migrations

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3


@contextmanager
def _exclusive_lock(lock_path: Path):
    """Hold an exclusive lock until the block ends. flock on POSIX; one locked
    byte via msvcrt on Windows (the file must be at least that long)."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+b") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            return
        import msvcrt
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        while True:
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                time.sleep(0.05)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def _schema_sql() -> str:
    """Lazy: schema.sql ships as package data; reading it at import time
    broke wheel installs without the file (CI wheel smoke)."""
    return (Path(__file__).parent / "schema.sql").read_text()


class MvisDbError(Exception):
    pass


class NoRunError(MvisDbError):
    pass


class RunNotSolvedError(MvisDbError):
    pass


class SealedError(MvisDbError):
    pass


class SchemaTooNewError(MvisDbError):
    pass


def _b(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(a).tobytes()


def _f64(raw: bytes | None, shape) -> np.ndarray | None:
    if raw is None:
        return None
    return np.frombuffer(raw, np.float64).reshape(shape).copy()


def _dist_f64(raw: bytes | None, model) -> np.ndarray | None:
    """A stored distortion blob, zero-padded up to the model's coefficient
    count: runs written before omni_radtan carried p2 stored 4 omni coeffs
    ([xi,k1,k2,p1] -> p2 = 0). Never trims (cam-only rows keep their padding)."""
    d = _f64(raw, -1)
    if d is None or model is None:
        return d
    from mvis.core.camera_models import n_dist
    n = n_dist(model.value if isinstance(model, CameraModel) else str(model))
    return np.pad(d, (0, n - len(d))) if len(d) < n else d


def _without_removed_model(row: tuple, db_path) -> tuple:
    """A `camera` row with a removed model (an older capture, kb8) as model-less:
    the capture's detections do not depend on the model, so the file stays usable
    for a new solve. A stored RESULT with that model still raises when read."""
    from mvis.core.camera_models import is_removed_model, removed_model_message
    if not is_removed_model(row[8]):
        return row
    logger.warning("%s, camera %s: %s; ignoring the stored model",
                   db_path, row[0], removed_model_message(row[8]))
    return row[:8] + (None,) + row[9:]


def _i64(raw: bytes, shape=-1) -> np.ndarray:
    return np.frombuffer(raw, np.int64).reshape(shape).copy()


class DatabaseManager:
    """Held ONLY at the edges (cli, regression, scripts, studies, tests)."""

    # ── connections ────────────────────────────────────────────────────────
    def _connect(self, path: str | Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path))
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _open_checked(self, path: str | Path) -> sqlite3.Connection:
        if not Path(path).exists():
            raise FileNotFoundError(path)
        conn = self._connect(path)
        row = conn.execute("SELECT schema_version FROM schema_meta").fetchone()
        if row and row[0] > SCHEMA_VERSION:
            conn.close()
            raise SchemaTooNewError(f"file schema {row[0]} > code {SCHEMA_VERSION}")
        if row and row[0] < SCHEMA_VERSION:
            apply_migrations(conn)               # forward, in one transaction
        return conn

    def _create(self, path: Path, mvis_commit: str | None) -> sqlite3.Connection:
        conn = self._connect(path)
        conn.executescript(_schema_sql())
        conn.execute(
            "INSERT INTO schema_meta (id, schema_version, mvis_commit, created_at)"
            " VALUES (1, ?, ?, ?)", (SCHEMA_VERSION, mvis_commit, time.time()))
        return conn

    # ── save: core-once + attach-a-run ─────────────────────────────────────
    def save(self, db: MVISDatabase, path: str | Path, *,
             identity_hash: str | None = None,
             spec_json: str | None = None,
             attach_run: bool = True,
             replace: bool = False) -> int:
        """Persist the ONE in-memory run: create+seal the core if this is the
        bag's first save (flock + .building + atomic rename), verify the
        fingerprint otherwise, then attach the run overlay. Returns run_id
        (also written onto db.run.run_id).

        attach_run=False saves the CAPTURE only (the detection cache's case:
        a capture is not an experiment, so no placeholder run row) and
        returns 0.

        replace=True drops an existing file under the lock first, so concurrent
        savers of one detection cache cannot race on the delete."""
        path = Path(path)
        fp = observation_fingerprint(db)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with _exclusive_lock(lock_path):
            if replace:
                path.unlink(missing_ok=True)
            if not path.exists():
                building = path.with_suffix(path.suffix + ".building")
                if building.exists():
                    building.unlink()                 # a crashed build is never readable
                conn = self._create(building, db.run.mvis_commit)
                try:
                    self._write_core(conn, db, fp, identity_hash, spec_json)
                    conn.commit()
                    conn.close()
                    os.replace(building, path)        # seal = the rename
                except BaseException:
                    conn.close()
                    if building.exists():
                        building.unlink()
                    raise
            conn = self._open_checked(path)
            try:
                stored = conn.execute("SELECT content_hash FROM capture").fetchone()[0]
                if stored != fp:
                    raise SealedError(
                        "measurement fingerprint mismatch — this file holds a "
                        "DIFFERENT capture (sealed core is never rewritten)")
                run_id = self._attach_run(conn, db) if attach_run else 0
                conn.commit()
            finally:
                conn.close()
        if attach_run:
            db.run.run_id = run_id
        return run_id

    def _write_core(self, conn, db: MVISDatabase, fp: str,
                    identity_hash: str | None = None,
                    spec_json: str | None = None) -> None:
        conn.execute(
            "INSERT INTO capture (id, name, capture_mode, content_hash,"
            " identity_hash, spec_json, created_at)"
            " VALUES (1, ?, 'board', ?, ?, ?, ?)",
            (db.name, fp, identity_hash, spec_json, time.time()))
        for s in db.streams:
            i = s.info
            res = i.resolution or (None, None)
            conn.execute(
                "INSERT INTO camera VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (s.camera_id, i.name, i.topic, i.shutter.value, i.readout_s,
                 i.rolling_mode,
                 _b(i.intrinsics) if i.intrinsics is not None else None,
                 _b(i.distortion) if i.distortion is not None else None,
                 i.camera_model.value if i.camera_model else None,
                 res[0], res[1], i.calib_provenance.value))
        for s in db.imus:
            i = s.info
            conn.execute(
                "INSERT INTO imu VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (s.imu_id, i.name, i.topic, i.sigma_g, i.sigma_bg, i.sigma_a,
                 i.sigma_ba, i.frequency_hz,
                 _b(i.xi24) if i.xi24 is not None else None,
                 i.intrinsics_model.value if i.intrinsics_model else None,
                 i.calib_provenance.value))
            if s.measurements is not None:
                m = s.measurements
                conn.execute("INSERT INTO imu_meas VALUES (?,?,?,?,?)",
                             (s.imu_id, len(m.t), _b(m.t), _b(m.gyro), _b(m.accel)))
        for b in db.boards.values():
            conn.execute(
                "INSERT INTO board VALUES (?,?,?,?,?,?,?,?)",
                (b.board_id, b.board_type.value,
                 b.tag_family.value if b.tag_family else None,
                 b.rows, b.cols, b.spacing, b.size, b.start_id))
            conn.executemany(
                "INSERT INTO board_pt VALUES (?,?,?,?,?,?,?)",
                [(b.board_id, p.board_pt_id, p.tag_id,
                  int(p.corner_id) if p.corner_id is not None else None,
                  float(p.board_pt_3d[0]), float(p.board_pt_3d[1]),
                  float(p.board_pt_3d[2])) for p in b.pts.values()])
        r = db.rig
        conn.execute("INSERT INTO rig VALUES (1,?,?,?,?)",
                     (r.base_camera_id, r.base_imu_id, r.sync_kind,
                      r.sync_tolerance_s))
        conn.executemany("INSERT INTO stereo_pair (cam_a, cam_b) VALUES (?,?)",
                         r.stereo_pairs)
        for s in db.streams:
            for f in s.frames:
                conn.execute("INSERT INTO frame VALUES (?,?,?)",
                             (f.frame_id, f.camera_id, f.timestamp))
                for g in f.board_detections:
                    conn.execute("INSERT INTO board_detection VALUES (?,?)",
                                 (f.frame_id, g.board_id))
                    dets = g.detections
                    conn.execute(
                        "INSERT INTO detection VALUES (?,?,?,?,?,?)",
                        (f.frame_id, g.board_id, len(dets),
                         _b(np.array([d.point_id for d in dets], np.int64)),
                         _b(np.array([d.uv for d in dets], np.float64).reshape(-1, 2)),
                         _b(np.array([d.detector_emission_order for d in dets],
                                     np.int64))))

    def _attach_run(self, conn, db: MVISDatabase) -> int:
        run = db.run
        if run.origin is RunOrigin.GROUND_TRUTH:
            n = conn.execute("SELECT COUNT(*) FROM run WHERE origin='ground_truth'"
                             ).fetchone()[0]
            if n:
                raise SealedError("a ground_truth run already exists (write-once)")
        win = run.window or (None, None)
        policy = {f"{k[0].value}:{k[1]}":
                  {a: [getattr(p, a).state.value, getattr(p, a).sigma]
                   for a in ("intrinsics", "readout", "extrinsics", "td")}
                  for k, p in run.sensor_policy.items()}
        cur = conn.execute(
            "INSERT INTO run (calib_choice, origin, description, solve_mode,"
            " variant, collection, state_hz, window_lo, window_hi, min_obs,"
            " config_json, config_sha256, mvis_commit, machine_id, status,"
            " started_at, wall_s, residual_promote_px, sensor_policy_json,"
            " factor_hz_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run.calib_choice.value, run.origin.value, run.description,
             run.solve_mode, run.variant, run.collection, run.state_hz,
             win[0], win[1], run.min_obs, json.dumps(run.config),
             run.config_sha256, run.mvis_commit, run.machine_id, run.status,
             run.started_at, run.wall_s, run.residual_promote_px,
             json.dumps(policy),
             json.dumps({f"{k[0].value}:{k[1]}": v
                         for k, v in run.factor_hz.items()})))
        run_id = cur.lastrowid
        # sparse flag diffs (overlay = the serialization of the inline flags)
        for s in db.streams:
            for f in s.frames:
                if (f.is_motion_still is not None or not f.is_selected_to_use
                        or f.exclude_reason):
                    conn.execute(
                        "INSERT INTO frame_status VALUES (?,?,?,?,?)",
                        (run_id, f.frame_id,
                         None if f.is_motion_still is None else int(f.is_motion_still),
                         int(f.is_selected_to_use), f.exclude_reason))
                for g in f.board_detections:
                    if (g.pose_cam_to_board is not None or not g.is_valid_to_use
                            or not g.is_selected):
                        p = g.pose_cam_to_board
                        conn.execute(
                            "INSERT INTO board_detection_status VALUES (?,?,?,?,?,?,?)",
                            (run_id, f.frame_id, g.board_id,
                             _b(p.T) if p else None,
                             p.source.value if p else None,
                             int(g.is_valid_to_use), int(g.is_selected)))
                    for d in g.detections:
                        if (not d.is_valid or not d.is_selected
                                or d.exclude_reason or d.residual is not None):
                            conn.execute(
                                "INSERT INTO detection_status VALUES (?,?,?,?,?,?,?,?,?)",
                                (run_id, f.frame_id, g.board_id, d.point_id,
                                 int(d.is_valid), int(d.is_selected),
                                 d.exclude_reason,
                                 d.residual.residual_px if d.residual else None,
                                 d.residual.residual_source.value if d.residual else None))
        es = db.estimate_states
        if es is not None:
            ref = es.reference_stream
            conn.execute("INSERT INTO estimate_meta VALUES (?,?,?,?)",
                         (run_id, ref[0].value, ref[1], es.state_hz))
            if es.overall_rms is not None:
                conn.execute("UPDATE run SET overall_rms=? WHERE run_id=?",
                             (es.overall_rms, run_id))
            for ss in es.sensor_states:
                kind, sid = ss.sensor
                intr = ss.intrinsics
                if isinstance(intr, CameraIntrinsicsState):
                    rp = ss.reproj
                    conn.execute(
                        "INSERT INTO result_camera VALUES"
                        " (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (run_id, sid, _b(intr.intrinsics), _b(intr.distortion),
                         intr.camera_model.value, intr.readout_s,
                         intr.provenance.value,
                         _b(intr.sigmas) if intr.sigmas is not None else None,
                         rp.rms if rp else None, rp.mean if rp else None,
                         rp.median if rp else None, rp.p90 if rp else None,
                         rp.p95 if rp else None, rp.n_obs if rp else None))
                else:
                    conn.execute(
                        "INSERT INTO result_imu VALUES (?,?,?,?,?,?)",
                        (run_id, sid, _b(intr.xi24), intr.model.value,
                         intr.provenance.value,
                         _b(intr.sigmas) if intr.sigmas is not None else None))
                if ss.relative is not None:
                    rel = ss.relative
                    conn.execute(
                        "INSERT INTO extrinsic VALUES (?,?,?,?,?,?,?,?,?)",
                        (run_id, kind.value, sid, ref[0].value, ref[1],
                         _b(rel.T_x_to_ref), rel.td_x_to_ref_s,
                         _b(rel.sigma6) if rel.sigma6 is not None else None,
                         rel.provenance.value))
            if es.world_frame is not None:
                w = es.world_frame
                conn.execute("INSERT INTO world_frame VALUES (?,?,?,?)",
                             (run_id, w.kind, w.g_magnitude, w.yaw_gauge))
            if es.boards:
                for bid, T7 in es.boards.items():
                    conn.execute(
                        "INSERT INTO state_var VALUES (?,?,?,?,?,?,?)",
                        (run_id, VarType.T_BOARD_TO_WORLD.value, "board", bid,
                         "float64", json.dumps(list(np.shape(T7))), _b(np.asarray(T7))))
            bs = es.base_state
            if isinstance(bs, CamOnlyBaseState):
                conn.executemany(
                    "INSERT INTO state_knot VALUES (?,?,?)",
                    [(run_id, k, float(t)) for k, t in enumerate(bs.t)])
                conn.execute(
                    "INSERT INTO state_var VALUES (?,?,?,?,?,?,?)",
                    (run_id, VarType.T_CAM_TO_BOARD.value, "cam",
                     es.reference_stream[1], "float64",
                     json.dumps(list(np.shape(bs.T_cam_to_board))),
                     _b(np.asarray(bs.T_cam_to_board, np.float64))))
            elif isinstance(bs, CamImuBaseState):
                conn.executemany(
                    "INSERT INTO state_knot VALUES (?,?,?)",
                    [(run_id, k, float(t)) for k, t in enumerate(bs.t)])
                for vt, arr in ((VarType.T_IMU_TO_WORLD, bs.T_imu_to_world),
                                (VarType.V, bs.v), (VarType.B_G, bs.b_g),
                                (VarType.B_A, bs.b_a)):
                    conn.execute(
                        "INSERT INTO state_var VALUES (?,?,?,?,?,?,?)",
                        (run_id, vt.value, "imu", db.rig.base_imu_id or 0,
                         "float64", json.dumps(list(np.shape(arr))),
                         _b(np.asarray(arr, np.float64))))
        return run_id

    # ── load: fold one run's overlay back into inline flags ────────────────
    def load(self, path: str | Path, run_id: int | None = None) -> MVISDatabase:
        conn = self._open_checked(path)
        try:
            name = conn.execute("SELECT name FROM capture").fetchone()[0]
            rig_row = conn.execute("SELECT base_camera_id, base_imu_id, sync_kind,"
                                   " sync_tolerance_s FROM rig").fetchone()
            rig = Rig(base_camera_id=rig_row[0], base_imu_id=rig_row[1],
                      stereo_pairs=[tuple(r) for r in conn.execute(
                          "SELECT cam_a, cam_b FROM stereo_pair")],
                      sync_kind=rig_row[2], sync_tolerance_s=rig_row[3])
            run = self._read_run(conn, run_id) if run_id is not None else Run(
                CalibChoice.CAM_ONLY)
            db = MVISDatabase(name=name, rig=rig, run=run)
            for r in conn.execute("SELECT * FROM board"):
                pts = {row[1]: BoardPt(row[1], row[2],
                                       Corner(row[3]) if row[3] is not None else None,
                                       np.array(row[4:7]))
                       for row in conn.execute(
                           "SELECT * FROM board_pt WHERE board_id=?", (r[0],))}
                db.boards[r[0]] = Board(r[0], BoardType(r[1]),
                                        rows=r[3], cols=r[4], spacing=r[5],
                                        size=r[6], start_id=r[7],
                                        tag_family=TagFamily(r[2]) if r[2] else None,
                                        pts=pts)
            for r in conn.execute("SELECT * FROM camera ORDER BY camera_id"):
                r = _without_removed_model(r, path)
                info = CameraInfo(
                    name=r[1], topic=r[2], shutter=ShutterType(r[3]),
                    readout_s=r[4], rolling_mode=r[5],
                    intrinsics=_f64(r[6], -1), distortion=_dist_f64(r[7], r[8] or None),
                    camera_model=CameraModel(r[8]) if r[8] else None,
                    resolution=(r[9], r[10]) if r[9] is not None else None,
                    calib_provenance=CalibProvenance(r[11]))
                db.streams.append(CameraStream(r[0], info))
            for r in conn.execute("SELECT * FROM imu ORDER BY imu_id"):
                info = ImuInfo(name=r[1], topic=r[2], sigma_g=r[3], sigma_bg=r[4],
                               sigma_a=r[5], sigma_ba=r[6], frequency_hz=r[7],
                               xi24=_f64(r[8], -1),
                               intrinsics_model=ImuIntrinsicsModel(r[9]) if r[9] else None,
                               calib_provenance=CalibProvenance(r[10]))
                m = conn.execute("SELECT n,t,gyro,accel FROM imu_meas WHERE imu_id=?",
                                 (r[0],)).fetchone()
                meas = (ImuMeasurements(_f64(m[1], -1), _f64(m[2], (-1, 3)),
                                        _f64(m[3], (-1, 3))) if m else None)
                db.imus.append(ImuStream(r[0], info, meas))
            self._read_frames(conn, db)
            if run_id is not None:
                self._fold_overlay(conn, db, run_id)
                db.estimate_states = self._read_estimate(conn, db, run_id)
            db._next_frame_id = 1 + max(
                (f.frame_id for s in db.streams for f in s.frames), default=-1)
            return db
        finally:
            conn.close()

    def _read_run(self, conn, run_id: int) -> Run:
        r = conn.execute("SELECT * FROM run WHERE run_id=?", (run_id,)).fetchone()
        if r is None:
            raise NoRunError(f"no run {run_id}")
        run = Run(CalibChoice(r[1]), run_id=r[0], origin=RunOrigin(r[2]),
                  description=r[3], solve_mode=r[4], variant=r[5], collection=r[6],
                  state_hz=r[7],
                  window=(r[8], r[9]) if r[8] is not None else None,
                  min_obs=r[10], config=json.loads(r[11] or "{}"),
                  config_sha256=r[12], mvis_commit=r[13], machine_id=r[14],
                  status=r[15], started_at=r[16], wall_s=r[17],
                  residual_promote_px=r[18])
        for k, v in json.loads(r[20] or "{}").items():
            kind, sid = k.split(":")
            run.factor_hz[(SensorType(kind), int(sid))] = v
        return run

    def _read_frames(self, conn, db: MVISDatabase) -> None:
        streams = {s.camera_id: s for s in db.streams}
        frames_by_id = {}
        for fid, cid, ts in conn.execute(
                "SELECT frame_id, camera_id, timestamp FROM frame"
                " ORDER BY camera_id, timestamp"):
            from mvis.mvisdb.mvisdatabase import FrameData
            f = FrameData(timestamp=ts, frame_id=fid, camera_id=cid)
            streams[cid].frames.append(f)
            frames_by_id[fid] = f
        for fid, bid, n, pid_raw, uv_raw, em_raw in conn.execute(
                "SELECT frame_id, board_id, n, point_id, uv, emission FROM detection"):
            f = frames_by_id[fid]
            pids, uv, em = _i64(pid_raw), _f64(uv_raw, (-1, 2)), _i64(em_raw)
            dets = [TagDetection(uv=uv[i], point_id=int(pids[i]),
                                 detector_emission_order=int(em[i]))
                    for i in range(n)]
            db.add_board_detection(f, bid, dets)
        self._frames_by_id = frames_by_id      # transient, used by _fold_overlay

    def _fold_overlay(self, conn, db: MVISDatabase, run_id: int) -> None:
        fb = self._frames_by_id
        for fid, still, sel, reason in conn.execute(
                "SELECT frame_id, is_motion_still, is_selected, exclude_reason"
                " FROM frame_status WHERE run_id=?", (run_id,)):
            f = fb[fid]
            f.is_motion_still = None if still is None else bool(still)
            f.is_selected_to_use = bool(sel)
            f.exclude_reason = reason
        for fid, bid, pose, src, valid, sel in conn.execute(
                "SELECT frame_id, board_id, pose, pose_source, is_valid,"
                " is_selected FROM board_detection_status WHERE run_id=?", (run_id,)):
            for g in fb[fid].board_detections:
                if g.board_id == bid:
                    if pose is not None:
                        g.pose_cam_to_board = PoseCamToBoard(
                            _f64(pose, (4, 4)), PoseSource(src))
                    g.is_valid_to_use = bool(valid)
                    g.is_selected = bool(sel)
        for fid, bid, pid, valid, sel, reason, rpx, rsrc in conn.execute(
                "SELECT frame_id, board_id, point_id, is_valid, is_selected,"
                " exclude_reason, residual_px, residual_source"
                " FROM detection_status WHERE run_id=?", (run_id,)):
            for g in fb[fid].board_detections:
                if g.board_id == bid:
                    for d in g.detections:
                        if d.point_id == pid:
                            d.is_valid, d.is_selected = bool(valid), bool(sel)
                            d.exclude_reason = reason
                            if rpx is not None:
                                d.residual = Residual(rpx, PoseSource(rsrc))

    def _read_estimate(self, conn, db, run_id: int) -> EstimateStates | None:
        meta = conn.execute("SELECT reference_type, reference_id, state_hz FROM"
                            " estimate_meta WHERE run_id=?", (run_id,)).fetchone()
        if meta is None:
            return None
        es = EstimateStates(reference_stream=(SensorType(meta[0]), meta[1]),
                            state_hz=meta[2])
        orow = conn.execute("SELECT overall_rms FROM run WHERE run_id=?",
                            (run_id,)).fetchone()
        if orow and orow[0] is not None:
            es.overall_rms = float(orow[0])
        rel_by_sensor = {}
        for ft, fi, T, td, s6, prov in conn.execute(
                "SELECT from_type, from_id, T, td_s, sigma6, provenance FROM"
                " extrinsic WHERE run_id=?", (run_id,)):
            rel_by_sensor[(SensorType(ft), fi)] = RelativeCalib(
                _f64(T, (4, 4)), td, _f64(s6, -1), CalibProvenance(prov))
        run_fhz = db.run.factor_hz
        for (cid, intr, dist, model, tr, prov, sig,
             rms, mean, med, p90, p95, n_obs) in conn.execute(
                "SELECT camera_id, intrinsics, distortion, camera_model,"
                " readout_s, provenance, sigmas, rms, mean_err, median_err,"
                " p90, p95, n_obs FROM result_camera WHERE run_id=?",
                (run_id,)):
            key = (SensorType.CAM, cid)
            rp = (ReprojStats(rms, mean, med, p90, p95, n_obs)
                  if any(v is not None for v in (rms, mean, med, p90, p95))
                  else None)
            es.sensor_states.append(SensorState(
                sensor=key,
                intrinsics=CameraIntrinsicsState(
                    _f64(intr, -1), _dist_f64(dist, model), CameraModel(model), tr,
                    CalibProvenance(prov), _f64(sig, -1)),
                factor_hz=run_fhz.get(key), relative=rel_by_sensor.get(key),
                reproj=rp))
        for iid, xi, model, prov, sig in conn.execute(
                "SELECT imu_id, xi24, model, provenance, sigmas FROM result_imu"
                " WHERE run_id=?", (run_id,)):
            key = (SensorType.IMU, iid)
            es.sensor_states.append(SensorState(
                sensor=key,
                intrinsics=ImuIntrinsicsState(_f64(xi, -1),
                                              ImuIntrinsicsModel(model),
                                              CalibProvenance(prov), _f64(sig, -1)),
                factor_hz=run_fhz.get(key), relative=rel_by_sensor.get(key)))
        w = conn.execute("SELECT kind, g_magnitude, yaw_gauge FROM world_frame"
                         " WHERE run_id=?", (run_id,)).fetchone()
        if w:
            es.world_frame = WorldFrame(*w)
        boards = {}
        vars_ = {}
        for vt, st, sid, dtype, shape, data in conn.execute(
                "SELECT var_type, sensor_type, sensor_id, dtype, shape, data"
                " FROM state_var WHERE run_id=?", (run_id,)):
            arr = np.frombuffer(data, dtype).reshape(json.loads(shape)).copy()
            if vt == VarType.T_BOARD_TO_WORLD.value:
                boards[sid] = arr
            else:
                vars_[vt] = arr
        if boards:
            es.boards = boards
        knots = [t for (t,) in conn.execute(
            "SELECT timestamp FROM state_knot WHERE run_id=? ORDER BY k", (run_id,))]
        if knots and VarType.T_CAM_TO_BOARD.value in vars_:
            es.base_state = CamOnlyBaseState(
                np.asarray(knots), vars_[VarType.T_CAM_TO_BOARD.value])
        elif knots and VarType.T_IMU_TO_WORLD.value in vars_:
            es.base_state = CamImuBaseState(
                np.asarray(knots), vars_[VarType.T_IMU_TO_WORLD.value],
                vars_[VarType.V.value], vars_[VarType.B_G.value],
                vars_[VarType.B_A.value])
        return es

    def capture_info(self, path: str | Path) -> dict:
        conn = self._open_checked(path)
        try:
            r = conn.execute("SELECT name, identity_hash, content_hash, spec_json"
                             " FROM capture").fetchone()
            return dict(name=r[0], identity_hash=r[1], content_hash=r[2],
                        spec_json=r[3])
        finally:
            conn.close()

    def ensure_capture(self, path: str | Path, spec, build) -> MVISDatabase:
        """The lifecycle door: HIT -> load the core, the
        bag is never opened; MISS -> ``build(spec) -> MVISDatabase`` (the
        manager DRIVES io; io never imports the manager), then save+seal;
        STALE (identity mismatch) -> refuse — the rebuild-beside/confirm flow
        is a later tier's operator concern, silence is not."""
        path = Path(path)
        ih = spec.identity_hash()
        if path.exists():
            info = self.capture_info(path)
            if info["identity_hash"] == ih:
                return self.load(path)                     # cache HIT
            raise SealedError(
                "capture identity mismatch (stale detections for this spec) — "
                "refusing to reuse or overwrite; rebuild-beside lands in T2+")
        db = build(spec)                                   # cache MISS
        self.save(db, path, identity_hash=ih,
                  spec_json=json.dumps(dataclasses.asdict(spec), default=str,
                                       sort_keys=True))
        return db

    # ── the cheap direct door ──────────────────────────────────────────────
    def state(self, path: str | Path, run_id: int) -> EstimateStates:
        """Direct overlay read: kB, ms, never touches detection. Raises
        NoRunError / RunNotSolvedError distinctly."""
        conn = self._open_checked(path)
        try:
            if conn.execute("SELECT 1 FROM run WHERE run_id=?",
                            (run_id,)).fetchone() is None:
                raise NoRunError(f"no run {run_id}")
            db = MVISDatabase(name="", run=self._read_run(conn, run_id))
            es = self._read_estimate(conn, db, run_id)
            if es is None or not es.sensor_states:
                raise RunNotSolvedError(f"run {run_id} has no results")
            return es
        finally:
            conn.close()

    # ── listings + columnar fast path ──────────────────────────────────────
    def runs(self, path: str | Path) -> list[dict]:
        conn = self._open_checked(path)
        try:
            return [dict(run_id=r[0], calib_choice=r[1], origin=r[2],
                         description=r[3], status=r[4])
                    for r in conn.execute(
                        "SELECT run_id, calib_choice, origin, description,"
                        " status FROM run ORDER BY run_id")]
        finally:
            conn.close()

    def latest_solved_run_id(self, path: str | Path) -> int | None:
        """The id of the newest run with status ``ok``, or None when no run solved."""
        solved = [r["run_id"] for r in self.runs(path) if r.get("status") == "ok"]
        return max(solved) if solved else None

    def latest_solved_state(self, path: str | Path) -> EstimateStates | None:
        """The estimate of the newest run with status ``ok``, or None when no run solved."""
        rid = self.latest_solved_run_id(path)
        return None if rid is None else self.state(path, rid)

    def stream_resolutions(self, path: str | Path) -> dict[int, tuple[int, int]]:
        """camera_id -> (width, height) from the capture's camera table, for
        streams that carry a resolution (res_w/res_h NULL ones are omitted).
        A light read for the export views, which must rebuild
        CameraParams.image_size without loading the whole capture."""
        conn = self._open_checked(path)
        try:
            return {int(r[0]): (int(r[1]), int(r[2]))
                    for r in conn.execute(
                        "SELECT camera_id, res_w, res_h FROM camera"
                        " ORDER BY camera_id")
                    if r[1] is not None and r[2] is not None}
        finally:
            conn.close()

    def frame_arrays(self, path: str | Path, camera_id: int) -> dict[str, np.ndarray]:
        """Columnar read straight from the blobs — no dataclass tree."""
        conn = self._open_checked(path)
        try:
            fi, ts, pid, bid, uv, row = [], [], [], [], [], []
            frames = conn.execute(
                "SELECT frame_id, timestamp FROM frame WHERE camera_id=?"
                " ORDER BY timestamp", (camera_id,)).fetchall()
            index_of = {f: i for i, (f, _) in enumerate(frames)}
            t_of = dict(frames)
            for fid, b, n, p_raw, uv_raw, _ in conn.execute(
                    "SELECT frame_id, board_id, n, point_id, uv, emission FROM"
                    " detection WHERE frame_id IN (SELECT frame_id FROM frame"
                    " WHERE camera_id=?)", (camera_id,)):
                p = _i64(p_raw)
                u = _f64(uv_raw, (-1, 2))
                fi.extend([index_of[fid]] * n)
                ts.extend([t_of[fid]] * n)
                bid.extend([b] * n)
                pid.extend(p.tolist())
                uv.append(u)
            return {
                "frame_index": np.asarray(fi, np.int64),
                "timestamp": np.asarray(ts, float),
                "point_id": np.asarray(pid, np.int64),
                "board_id": np.asarray(bid, np.int64),
                "uv": (np.concatenate(uv) if uv else np.empty((0, 2))),
                "row": (np.concatenate(uv)[:, 1].astype(np.int64) if uv
                        else np.empty(0, np.int64)),
            }
        finally:
            conn.close()
