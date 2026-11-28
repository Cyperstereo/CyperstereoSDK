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
"""Board detection pipeline with streaming thumbnails and lazy loading.

Moves the detection logic from ``scripts/run_calibration.py`` into the
library so that any caller (CLI, standalone scripts, test harnesses)
can share the same load → detect → filter → cache workflow.

Usage::

    from mvis.io.data_loader import DataLoader
    from mvis.io.detection_pipeline import DetectionPipeline

    loader = DataLoader(config)
    det = DetectionPipeline(board, thumb_dir="results/thumbnails",
                            cache_path="results/detections.json")
    result = det.run_streaming(loader, cache_key=loader.cache_key,
                               target_hz=5.0)
    # result.all_frames, result.image_sizes, result.detection_images, ...
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from mvis.detection.board import Board
from mvis.core.types import FrameObservations, Observation
from mvis.io.data_source import _ROS1_TYPES
from mvis.io.frame_detector import detect_frame   # shared front-end


def write_thumbnail(thumb_dir, board, cid, idx, img, obs, ts=None, reset=False):
    """Draw the detection overlay on `img` and write `cam<cid>/<idx>.jpg`.

    With `ts`, also appends `idx,timestamp` to `cam<cid>/index.csv` (truncated
    first when `reset`), so reports can match a pose time to its thumbnail.
    Shared by DetectionPipeline and the cam-IMU loaders.
    """
    cam_dir = Path(thumb_dir) / f"cam{cid}"
    cam_dir.mkdir(parents=True, exist_ok=True)
    if ts is not None:
        with open(cam_dir / THUMB_INDEX_NAME, "w" if reset else "a") as f:
            f.write(f"{idx},{ts:.9f}\n")

    vis = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
           if len(img.shape) == 2 else img.copy())

    is_chess = getattr(board, '_num_tags', -1) == 0
    r = max(4, int(0.006 * max(vis.shape[:2])))

    if is_chess and obs:
        bcols = getattr(board, '_cols', 1)
        rows_map: dict[int, list[Observation]] = {}
        for o in obs:
            rows_map.setdefault(o.point_id // bcols, []).append(o)
        for row_id in sorted(rows_map):
            pts = sorted(rows_map[row_id],
                         key=lambda o: o.point_id)
            for k in range(len(pts) - 1):
                p1 = tuple(pts[k].pixel.astype(int))
                p2 = tuple(pts[k + 1].pixel.astype(int))
                cv2.line(vis, p1, p2, (0, 200, 0), max(1, r // 3))

    for o in obs:
        pt = tuple(o.pixel.astype(int))
        cv2.circle(vis, pt, r, (0, 255, 0), -1)

    h0, w0 = vis.shape[:2]
    scale = _THUMB_MAX_DIM / float(max(h0, w0))
    if scale < 1.0:
        vis = cv2.resize(vis, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(cam_dir / f"{idx:05d}.jpg"), vis,
                [cv2.IMWRITE_JPEG_QUALITY, _THUMB_JPEG_QUALITY])


def _make_cache(cache_path):
    """The detection cache is the sqlite SqliteDetectionCache (checked against the
    JSON cache on a real bag: bit-identical detections, identical solves, true cross-process
    hit, 33->12.7 MB). Nothing writes the legacy JSON cache any more; old detections.json
    files are read through `mvis.io.detection_cache.legacy_json_frames`."""
    from mvis.mvisdb.shim import SqliteDetectionCache
    return SqliteDetectionCache(cache_path)

logger = logging.getLogger("mvis.io.detection_pipeline")

_THUMB_MAX_DIM = 320
_THUMB_JPEG_QUALITY = 70
#: Per-camera thumbnail index (`idx,timestamp` lines) beside the JPEGs.
THUMB_INDEX_NAME = "index.csv"

#: Seconds between detection progress lines; ``MVIS_DETECT_PROGRESS_S`` overrides,
#: 0 turns them off.
DETECT_PROGRESS_INTERVAL_S = 30.0


def _progress_interval_s() -> float:
    raw = os.environ.get("MVIS_DETECT_PROGRESS_S")
    if raw is None:
        return DETECT_PROGRESS_INTERVAL_S
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(f"MVIS_DETECT_PROGRESS_S={raw!r} is not a number; "
                       f"using {DETECT_PROGRESS_INTERVAL_S:g} s")
        return DETECT_PROGRESS_INTERVAL_S


class _DetectProgress:
    """Throttled progress log for a running board detection.

    Every ``interval_s`` it logs, per camera, the frames read so far and (when
    ``counts_detections``) how many had a usable board, plus the overall frame rate.
    A long detection (large bags, many cameras) would otherwise be silent until
    it ends. Thread-safe: the parallel path's workers report detections while
    the producer reports reads.
    """

    def __init__(self, counts_detections: bool = True,
                 interval_s: float | None = None, clock=time.monotonic):
        self._counts_detections = counts_detections
        self._interval = _progress_interval_s() if interval_s is None else interval_s
        self._clock = clock
        self._t0 = clock()
        self._next = self._t0 + self._interval
        self._read: dict[int, int] = {}
        self._detected: dict[int, int] = {}
        self._lock = threading.Lock()

    def read(self, cid: int) -> None:
        with self._lock:
            self._read[cid] = self._read.get(cid, 0) + 1
        self.maybe_log()

    def detected(self, cid: int) -> None:
        with self._lock:
            self._detected[cid] = self._detected.get(cid, 0) + 1
        self.maybe_log()     # parallel workers keep logging while the queues drain

    def maybe_log(self) -> None:
        if self._interval <= 0:
            return
        now = self._clock()
        with self._lock:     # format and log under the lock: lines come out in count order
            if now < self._next:
                return
            self._next = now + self._interval
            logger.info(self._format(now))

    def finish(self) -> None:
        """One last line with the final counts, once detection is done (when progress is on)."""
        if self._interval <= 0:
            return
        with self._lock:
            logger.info(self._format(self._clock()))

    def _format(self, now: float) -> str:
        elapsed = now - self._t0
        total = sum(self._read.values())
        rate = total / elapsed if elapsed > 0 else 0.0
        cams = []
        for cid in sorted(self._read):
            n = self._read[cid]
            if self._counts_detections:
                cams.append(f"cam{cid} {n} read/{self._detected.get(cid, 0)} detected")
            else:
                cams.append(f"cam{cid} {n} read")
        m, s = divmod(int(elapsed), 60)
        return (f"  detecting: {m}m{s:02d}s, {total} frames ({rate:.1f}/s) — "
                + ", ".join(cams))


@dataclass
class DetectionResult:
    """Output of :meth:`DetectionPipeline.run_streaming`."""

    #: Diversity-filtered frames per camera (for optimization).
    all_frames: dict[int, list[FrameObservations]] = field(default_factory=dict)

    #: ALL detected frames per camera before diversity filtering
    #: (for trajectory alignment in Stage 3).
    all_frames_full_rate: dict[int, list[FrameObservations]] = field(
        default_factory=dict)

    #: Image sizes per camera {cam_id: (width, height)}.
    image_sizes: dict[int, tuple[int, int]] = field(default_factory=dict)

    #: Report-needed thumbnails per camera.
    #: Only ≤35 images loaded (first 5 + 30 lowest-detection);
    #: remaining entries are None.
    detection_images: dict[int, list[np.ndarray | None]] = field(
        default_factory=dict)

    #: Per-frame detection metadata per camera.
    detection_histories: dict[int, list] = field(default_factory=dict)

    #: Directory where ALL thumbnails are saved on disk (or None).
    thumb_dir: Path | None = None

    #: The in-memory MVISDatabase the sweep path built.
    #: When present, its frame selection flags are the AUTHORITY for
    #: all_frames / all_frames_full_rate above. None on cache-hit and
    #: legacy-fallback paths (they grow the carrier later).
    db: object | None = None


class DetectionPipeline:
    """Detect calibration board in images with streaming thumbnail I/O.

    Features:

    * **Streaming** — frames come one at a time from
      :meth:`DataLoader.stream` and are freed right after detection.
    * **Streaming thumbnails** — annotated detection images are written
      to disk frame-by-frame, never accumulated in memory.
    * **Diversity filtering** — selects frames with sufficient board
      motion to ensure calibration coverage.
    * **Detection caching** — saves/loads detections from a JSON cache
      to skip re-detection on repeated runs.
    * **Report thumbnails** — after filtering, loads only the ≤35
      thumbnails needed for the HTML report.
    """

    def __init__(
        self,
        board: Board,
        thumb_dir: str | Path | None = None,
        cache_path: str | None = None,
        stream_store=None,
    ):
        """``stream_store``: optional ``io.stream_store.StreamDetectionStore``
        shared with other runs (cam-IMU included); looked up before detecting,
        filled after a fresh detection."""
        self._board = board
        self._stream_store = stream_store
        self._raw: dict[int, list[FrameObservations]] | None = None
        self._last_db = None            # sweep-path carrier stash (2c-2)
        self._thumb_dir = Path(thumb_dir) if thumb_dir else None
        self._thumb_index_started: set[int] = set()   # cams whose index.csv was reset
        self._cache_path = cache_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_streaming(
        self,
        loader,
        cam_names: dict[int, str] | None = None,
        cache_key: str = "",
        ref_cam: int = 0,
        enable_td: bool = False,
        target_hz: float | dict = 5.0,
    ) -> DetectionResult:
        """Stream-detect all frames, cache everything, subsample after.

        Memory-efficient: only 1 image in memory at a time during
        detection.  On cache hit, no images are loaded at all.

        This method:

        * Detects **every** frame from the source (no pre-load subsampling).
        * Saves **all** detections to cache (not just diversity-filtered).
        * Applies rate + diversity subsampling **after** detection on
          the cached ``FrameObservations``.

        Args:
            loader: :class:`DataLoader` instance.
            cam_names: ``{cam_id: display_name}``.  Inferred from
                *loader* if not given.
            cache_key: Key for detection cache validation.
            ref_cam: Reference camera ID.
            enable_td: Skip diversity filter for ref camera.
            target_hz: Target rate for post-detection subsampling.

        Returns:
            :class:`DetectionResult`.
        """
        self._last_db = None            # set by the sweep paths (2c-2)
        cam_names = cam_names or loader.cam_names()
        board_config = self._board_config()

        # --- Check cache ---
        if self._cache_path:
            cache = _make_cache(self._cache_path)
            if cache.is_valid(cache_key, board_config):
                logger.info(
                    "Using cached detections — skipping data loading")
                all_detected, det_hist, img_sizes = cache.load(
                    cache_key, board_config)
                self._last_db = getattr(cache, "last_db", None)  # 2c-3
                return self._subsample_and_return(
                    all_detected, img_sizes, det_hist, cam_names,
                    ref_cam, enable_td, target_hz)

        # --- Shared stream store (one detection per recording) ---
        store_streams = self._store_streams(loader)
        if store_streams is not None:
            hit = self._stream_store_lookup(store_streams)
            if hit is not None:
                all_detected, image_sizes, frame_counts = hit
                if self._thumb_dir is not None:
                    logger.warning("thumbnails not available: detections from "
                                   "shared store")
                if self._cache_path:
                    _make_cache(self._cache_path).save(
                        all_detected, cache_key, board_config, {}, image_sizes)
                return self._subsample_and_return(
                    all_detected, image_sizes, {}, cam_names,
                    ref_cam, enable_td, target_hz)
            self._raw = {cid: [] for cid in store_streams}

        # --- Stream detection ---
        camera_ids = sorted(
            c.camera_id for c in loader._config.cameras)
        n_cams = len(camera_ids)
        use_parallel = n_cams > 1

        if use_parallel:
            logger.info(
                f"Streaming detection ({n_cams} cameras in parallel)...")
            all_detected, image_sizes, frame_counts = \
                self._stream_detect_parallel_via_sweep(loader, camera_ids)
        else:
            logger.info("Streaming detection (1 image at a time)...")
            all_detected, image_sizes, frame_counts = \
                self._stream_detect_via_sweep(loader)

        # --- Fill the shared stream store (every frame with >= 1 corner) ---
        if store_streams is not None and self._raw is not None:
            for cid, (src, topic) in store_streams.items():
                self._stream_store.put(src, topic, self._board,
                                       self._raw.get(cid, []),
                                       frames_read=frame_counts.get(cid, 0))
            self._raw = None

        # --- Save ALL detections to cache ---
        if self._cache_path:
            cache = _make_cache(self._cache_path)
            cache.save(all_detected, cache_key, board_config,
                       {}, image_sizes)

        # --- Subsample and return ---
        return self._subsample_and_return(
            all_detected, image_sizes, {}, cam_names,
            ref_cam, enable_td, target_hz)

    def _store_streams(self, loader) -> dict[int, tuple[str, str]] | None:
        """``{camera_id: (bag, topic)}`` when the shared store applies (a store is
        configured and every camera reads a ROS1 bag topic), else ``None``."""
        if self._stream_store is None:
            return None
        cams = getattr(getattr(loader, "_config", None), "cameras", None) or []
        if not cams or any(c.source_type not in _ROS1_TYPES
                           or not c.topic for c in cams):
            return None
        return {c.camera_id: (c.source_path, c.topic)
                for c in sorted(cams, key=lambda c: c.camera_id)}

    def _stream_store_lookup(self, streams):
        """All cameras from the store (frames with a usable board), bridged
        through detection.sweep exactly like a fresh multi-camera detection;
        ``None`` if any camera misses."""
        min_pts = self._board.min_points_for_pose()
        cam_results: dict[int, list[FrameObservations]] = {}
        for cid, (src, topic) in streams.items():
            frames = self._stream_store.get(src, topic, self._board,
                                            camera_id=cid, min_obs=min_pts)
            if frames is None:
                return None
            cam_results[cid] = frames
        logger.info("Detections from the shared stream store — skipping detection")
        sizes = {cid: tuple(fr[-1].image_size) for cid, fr in cam_results.items() if fr}
        counts = {cid: len(fr) for cid, fr in cam_results.items()}
        return self._bridge_via_sweep(cam_results, sizes, counts)

    def _keep_raw(self, cid: int, frame: FrameObservations) -> None:
        """Record a frame for the shared store (any frame with a corner)."""
        raw = getattr(self, "_raw", None)          # absent on __new__-built pipelines (tests)
        if raw is not None and frame.observations:
            raw.setdefault(cid, []).append(frame)

    def _stream_detect_via_sweep(self, loader):
        """The sequential detection core routes through the detection
        component — sweep() fills an in-memory MVISDatabase and the bridge
        converts back to the legacy shapes while call sites migrate.
        Thumbnails ride a source tee (they move to analysis/ later).
        Geometry-free boards (feature mode) fall back to the legacy loop
        until their schema home lands."""
        from mvis.detection import BoardFeatureDetector, sweep
        from mvis.mvisdb.bridge import board_from_geometry, frame_observations_from_db
        from mvis.mvisdb.mvisdatabase import (
            CalibChoice, CameraInfo, CameraStream, MVISDatabase, Rig, Run)
        try:
            board_entity = board_from_geometry(self._board)
        except (ValueError, Exception) as e:                    # unknown/geometry-free
            logger.warning(f"sweep path unavailable ({e}); legacy sequential loop")
            return self._stream_detect_sequential(loader)

        at_logger = logging.getLogger("mvis.detection.apriltag_board")
        at_level = at_logger.level
        at_logger.setLevel(logging.WARNING)
        if self._thumb_dir:
            self._thumb_dir.mkdir(parents=True, exist_ok=True)

        db = MVISDatabase(name="stream", rig=Rig(), run=Run(CalibChoice.CAM_ONLY))
        db.boards[0] = board_entity
        frame_counts: dict[int, int] = {}
        sizes: dict[int, tuple[int, int]] = {}
        t0 = time.monotonic()
        progress = _DetectProgress()
        min_pts = self._board.min_points_for_pose()
        cur = [None]       # camera of the frame sweep is detecting (set at yield)

        class _Predetected:
            """Carries already-run detections so the thumbnail tee never
            detects twice."""
            def __init__(self, obs):
                self.obs = obs
            shape = (0, 0)

        def tee():
            for cid, img, ts in loader.stream():
                if cid not in frame_counts:
                    frame_counts[cid] = 0
                    db.streams.append(CameraStream(cid, CameraInfo(f"cam{cid}")))
                    if self._thumb_dir:
                        (self._thumb_dir / f"cam{cid}").mkdir(exist_ok=True)
                idx = frame_counts[cid]
                frame_counts[cid] += 1
                progress.read(cid)
                h, w = img.shape[:2]
                sizes[cid] = (int(w), int(h))
                if self._thumb_dir is not None or getattr(self, "_raw", None) is not None:
                    frame = detect_frame(img, self._board, cid, ts)  # legacy thumbnail / store need obs
                    if self._thumb_dir is not None:
                        self._write_thumbnail(cid, idx, img, frame.observations, ts)
                    self._keep_raw(cid, frame)
                    cur[0] = cid
                    yield _Predetected(frame.observations), ts, cid
                else:
                    cur[0] = cid
                    yield img, ts, cid

        class _TeeDetector(BoardFeatureDetector):
            def detect(self, image, timestamp=0.0):
                if isinstance(image, _Predetected):
                    from mvis.detection.detectors import _to_tag_detections
                    dets = _to_tag_detections(image.obs)
                else:
                    dets = super().detect(image, timestamp)
                if len(dets) >= min_pts:          # sweep's min_obs rule
                    progress.detected(cur[0])
                return dets

        sweep(tee(), [_TeeDetector(self._board, 0)], db, min_obs=min_pts)
        progress.finish()
        for st in db.streams:
            if st.frames:
                st.info.resolution = sizes.get(st.camera_id)
        all_detected = {st.camera_id: frame_observations_from_db(db, st.camera_id)
                        for st in db.streams if st.frames}
        image_sizes = {cid: sizes[cid] for cid in all_detected}
        self._last_db = db
        at_logger.setLevel(at_level)
        elapsed = time.monotonic() - t0
        for cid in sorted(all_detected):
            logger.info(f"  cam{cid}: {len(all_detected[cid])}/"
                        f"{frame_counts.get(cid, 0)} detected, {elapsed:.1f}s "
                        "(via detection.sweep)")
        return all_detected, image_sizes, frame_counts

    def _stream_detect_sequential(self, loader):
        """Stream-detect all frames sequentially (single camera or fallback)."""
        at_logger = logging.getLogger("mvis.detection.apriltag_board")
        at_level = at_logger.level
        at_logger.setLevel(logging.WARNING)

        if self._thumb_dir:
            self._thumb_dir.mkdir(parents=True, exist_ok=True)

        all_detected: dict[int, list[FrameObservations]] = {}
        image_sizes: dict[int, tuple[int, int]] = {}
        frame_counts: dict[int, int] = {}
        t0 = time.monotonic()
        progress = _DetectProgress()

        for cid, img, ts in loader.stream():
            if cid not in all_detected:
                all_detected[cid] = []
                frame_counts[cid] = 0
                if self._thumb_dir:
                    (self._thumb_dir / f"cam{cid}").mkdir(
                        exist_ok=True)

            idx = frame_counts[cid]
            frame_counts[cid] += 1

            frame = detect_frame(img, self._board, cid, ts)  # shared front-end
            obs = frame.observations
            self._write_thumbnail(cid, idx, img, obs, ts)
            self._keep_raw(cid, frame)

            if len(obs) >= self._board.min_points_for_pose():
                image_sizes[cid] = frame.image_size
                all_detected[cid].append(frame)
                progress.detected(cid)
            progress.read(cid)
        progress.finish()

        at_logger.setLevel(at_level)
        elapsed = time.monotonic() - t0

        for cid in sorted(all_detected):
            n = len(all_detected[cid])
            total = frame_counts.get(cid, 0)
            logger.info(
                f"  cam{cid}: {n}/{total} detected, {elapsed:.1f}s")

        return all_detected, image_sizes, frame_counts

    def _stream_detect_parallel_via_sweep(self, loader, camera_ids):
        """The parallel path's results flow through the detection
        component. The thread-per-camera detection mechanics stay legacy (they
        are proven and per-camera FIFO order is preserved); the collected
        frames are then swept into an MVISDatabase and bridged back, so the
        component's invariants (time order per camera, frozen uv, derived
        rows) hold for this path too and violations surface loudly. Sweep-
        native parallelism lands with the board/ dismantling."""
        cam_results, cam_image_sizes, cam_frame_counts = \
            self._stream_detect_parallel(loader, camera_ids)
        return self._bridge_via_sweep(cam_results, cam_image_sizes, cam_frame_counts)

    def _bridge_via_sweep(self, cam_results, cam_image_sizes, cam_frame_counts):
        """Sweep collected frames into an MVISDatabase and bridge back (the
        component's invariants enforced); sets ``self._last_db``."""
        from mvis.detection import sweep
        from mvis.detection.detectors import _to_tag_detections, BoardFeatureDetector
        from mvis.mvisdb.bridge import board_from_geometry, frame_observations_from_db
        from mvis.mvisdb.mvisdatabase import (
            CalibChoice, CameraInfo, CameraStream, MVISDatabase, Rig, Run)
        try:
            board_entity = board_from_geometry(self._board)
        except Exception as e:                     # geometry-free: legacy shapes as-is
            logger.warning(f"component round-trip skipped ({e})")
            return cam_results, cam_image_sizes, cam_frame_counts

        class _Pre:
            def __init__(self, obs): self.obs = obs
            shape = (0, 0)

        class _PreDetector(BoardFeatureDetector):
            def detect(self, image, timestamp=0.0):
                return _to_tag_detections(image.obs)

        db = MVISDatabase(name="stream", rig=Rig(), run=Run(CalibChoice.CAM_ONLY))
        db.boards[0] = board_entity
        for cid in sorted(cam_results):
            db.streams.append(CameraStream(cid, CameraInfo(
                f"cam{cid}", resolution=cam_image_sizes.get(cid))))
        src = ((_Pre(f.observations), f.timestamp, cid)
               for cid in sorted(cam_results) for f in cam_results[cid])
        sweep(src, [_PreDetector(self._board, 0)], db)
        db.validate()                              # the invariants, enforced
        logger.info("  parallel results validated via detection.sweep "
                    f"({sum(len(v) for v in cam_results.values())} frames)")
        out = {cid: frame_observations_from_db(db, cid) for cid in cam_results}
        self._last_db = db
        return out, cam_image_sizes, cam_frame_counts

    def _stream_detect_parallel(self, loader, camera_ids):
        """Stream-detect with one worker thread per camera."""
        import copy
        import queue

        at_logger = logging.getLogger("mvis.detection.apriltag_board")
        at_level = at_logger.level
        at_logger.setLevel(logging.WARNING)

        if self._thumb_dir:
            self._thumb_dir.mkdir(parents=True, exist_ok=True)
            for cid in camera_ids:
                (self._thumb_dir / f"cam{cid}").mkdir(exist_ok=True)

        # Per-camera state (no sharing between threads)
        cam_queues = {
            cid: queue.Queue(maxsize=4) for cid in camera_ids
        }
        cam_results: dict[int, list[FrameObservations]] = {
            cid: [] for cid in camera_ids
        }
        cam_image_sizes: dict[int, tuple[int, int]] = {}
        cam_frame_counts = {cid: 0 for cid in camera_ids}
        cam_boards = {
            cid: copy.deepcopy(self._board) for cid in camera_ids
        }

        min_pts = self._board.min_points_for_pose()
        progress = _DetectProgress()

        def _worker(cid):
            board = cam_boards[cid]
            while True:
                item = cam_queues[cid].get()
                if item is None:
                    break
                img, ts, idx = item
                frame = detect_frame(img, board, cid, ts)  # shared front-end
                obs = frame.observations
                self._write_thumbnail(cid, idx, img, obs, ts)
                self._keep_raw(cid, frame)
                if len(obs) >= min_pts:
                    cam_image_sizes[cid] = frame.image_size
                    cam_results[cid].append(frame)
                    progress.detected(cid)
                del img

        # Start workers
        threads = {}
        for cid in camera_ids:
            t = threading.Thread(
                target=_worker, args=(cid,), daemon=True)
            t.start()
            threads[cid] = t

        # Producer: dispatch stream to per-camera queues
        t0 = time.monotonic()
        for cid, img, ts in loader.stream():
            if cid not in cam_frame_counts:
                cam_frame_counts[cid] = 0
            idx = cam_frame_counts[cid]
            cam_frame_counts[cid] += 1
            cam_queues[cid].put((img, ts, idx))
            progress.read(cid)

        # Sentinel + join
        for cid in camera_ids:
            cam_queues[cid].put(None)
        for t in threads.values():
            t.join()
        progress.finish()

        at_logger.setLevel(at_level)
        elapsed = time.monotonic() - t0

        for cid in sorted(cam_results):
            n = len(cam_results[cid])
            total = cam_frame_counts.get(cid, 0)
            logger.info(
                f"  cam{cid}: {n}/{total} detected, "
                f"{elapsed:.1f}s (parallel)")

        # The workers fill cam_image_sizes in completion order; key it by camera
        # id so what is stored from it (capture.spec_json) does not depend on
        # which thread finished first.
        cam_image_sizes = dict(sorted(cam_image_sizes.items()))
        return cam_results, cam_image_sizes, cam_frame_counts

    def _subsample_and_return(
        self,
        all_detected: dict[int, list[FrameObservations]],
        image_sizes: dict[int, tuple[int, int]],
        det_hist: dict,
        cam_names: dict[int, str],
        ref_cam: int,
        enable_td: bool,
        target_hz: float | dict,
    ) -> DetectionResult:
        """Apply rate + diversity subsampling on detected frames."""
        from mvis.io.frame_selector import (
            co_observed_selection,
            subsample_frames_by_hz,
            subsample_by_detection_diversity,
        )

        db = getattr(self, "_last_db", None)

        # Step 1: Rate subsample — the selector computes the keep-set; when
        # the sweep path built a db, the FLAGS become the authority (2c-2):
        # dropped frames are marked detection:hz_subsample and the output
        # lists are derived from the flags (same objects, same order —
        # timestamps are unique per camera, validate() enforces it).
        rate_filtered = subsample_frames_by_hz(
            all_detected, target_hz=target_hz)

        # Step 2: Diversity filter
        all_frames: dict[int, list[FrameObservations]] = {}
        all_frames_full_rate: dict[int, list[FrameObservations]] = {}
        diversity_kept: dict[int, set[float]] = {}

        independent: dict[int, list[int]] = {}
        for cid in sorted(rate_filtered):
            frames = rate_filtered[cid]
            all_frames_full_rate[cid] = frames

            if enable_td and cid == ref_cam:
                all_frames[cid] = list(frames)
                diversity_kept[cid] = {f.timestamp for f in frames}
                logger.info(
                    f"  {cam_names.get(cid, f'cam{cid}')}: "
                    f"{len(frames)} frames kept (ref camera, "
                    f"diversity filter skipped)")
            else:
                dummy = [(None, f.timestamp) for f in frames]
                obs_list = [f.observations for f in frames]
                selected_idx, _ = subsample_by_detection_diversity(
                    dummy, obs_list,
                    min_time_gap=0.05,
                    image_size=image_sizes.get(cid))
                independent[cid] = selected_idx

        # No time offset estimated => the cameras are treated as synchronized
        # and the joint solve pairs non-ref frames with ref pose nodes at the
        # SAME instant. Select jointly so every kept non-ref frame has one
        # (td-estimated runs keep the per-camera selection above).
        if not enable_td and len(independent) > 1 and ref_cam in independent:
            period = {}
            for cid, fr in all_detected.items():
                ts = np.sort([f.timestamp for f in fr])
                period[cid] = float(np.median(np.diff(ts))) if len(ts) > 1 \
                    else 0.0
            independent = co_observed_selection(
                rate_filtered, independent, ref_cam, period)

        for cid, selected_idx in independent.items():
            frames = rate_filtered[cid]
            cam_frames = [frames[i] for i in selected_idx]
            all_frames[cid] = cam_frames
            diversity_kept[cid] = {f.timestamp for f in cam_frames}
            logger.info(
                f"  {cam_names.get(cid, f'cam{cid}')}: "
                f"{len(cam_frames)} diverse frames "
                f"(from {len(frames)} rate-filtered)")
        all_frames = {cid: all_frames[cid] for cid in sorted(all_frames)}

        if db is not None:
            rate_kept = {cid: {f.timestamp for f in fr}
                         for cid, fr in rate_filtered.items()}
            for stream in db.streams:
                rk = rate_kept.get(stream.camera_id, set())
                dk = diversity_kept.get(stream.camera_id, set())
                for fr in stream.frames:
                    if fr.timestamp not in rk:
                        fr.is_selected_to_use = False
                        fr.exclude_reason = "detection:hz_subsample"
                    elif fr.timestamp not in dk:
                        fr.is_selected_to_use = False
                        fr.exclude_reason = "detection:diversity"
            # flags are now the authority: the lists above must be exactly
            # the flag query (checked cheaply here, hard in the unit test)
            for cid, fr in all_frames.items():
                n_flag = sum(1 for f in db.camera(cid).frames
                             if f.is_selected_to_use)
                assert n_flag == len(fr), (cid, n_flag, len(fr))

        # Load report-needed thumbnails from disk (first 5 + 30 lowest)
        det_images = self._load_cached_thumbnails(
            all_frames, sorted(all_frames.keys()))

        return DetectionResult(
            all_frames=all_frames,
            all_frames_full_rate=all_frames_full_rate,
            image_sizes=image_sizes,
            detection_images=det_images,
            detection_histories=det_hist,
            thumb_dir=self._thumb_dir,
            db=db,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _board_config(self) -> dict:
        """The FULL scalar board/detector config: the old
        5-field dict missed spacing, start_id, marker/detector knobs — the
        exact holes that let a detector-parameter change reuse stale corners.
        It is ``io.capture_key.board_fingerprint``."""
        from mvis.io.capture_key import board_fingerprint
        return board_fingerprint(self._board)

    def _write_thumbnail(
        self,
        cid: int,
        idx: int,
        img: np.ndarray,
        obs: list[Observation],
        ts: float | None = None,
    ) -> None:
        """Draw detection overlay and write thumbnail to disk (+ index line;
        the index restarts on this pipeline's first write per camera)."""
        if self._thumb_dir is None:
            return
        reset = cid not in self._thumb_index_started
        self._thumb_index_started.add(cid)
        write_thumbnail(self._thumb_dir, self._board, cid, idx, img, obs, ts,
                        reset=reset)

    def _load_cached_thumbnails(
        self,
        all_frames: dict[int, list[FrameObservations]],
        camera_ids: list[int],
    ) -> dict[int, list[np.ndarray | None]]:
        """Load report-needed thumbnails from disk.

        Loads only the ~35 images needed for the HTML report (first 5 for
        sample detections + 30 lowest-detection frames for Appendix A).
        All other entries are None to save memory.

        Thumbnails on disk are the full raw set (indexed sequentially).
        Since subsampled FrameObservations don't store their raw index,
        we load the first 5 and 30 lowest by matching sequential position
        in the on-disk thumbnail list.
        """
        if not self._thumb_dir or not self._thumb_dir.exists():
            return {cid: [] for cid in camera_ids}

        result: dict[int, list[np.ndarray | None]] = {}
        for cid in camera_ids:
            frames = all_frames.get(cid, [])
            needed = set()
            # First 5 for sample detections
            for i in range(min(5, len(frames))):
                needed.add(i)
            # 30 lowest by obs count for appendix
            pairs = sorted(
                enumerate(frames), key=lambda x: len(x[1].observations))
            for i, _ in pairs[:30]:
                needed.add(i)

            cam_dir = self._thumb_dir / f"cam{cid}"
            imgs: list[np.ndarray | None] = [None] * len(frames)

            if cam_dir.exists():
                all_jpgs = sorted(cam_dir.glob("*.jpg"))
                # Only load the specific indices we need
                for i in needed:
                    if i < len(all_jpgs):
                        img = cv2.imread(str(all_jpgs[i]))
                        if img is not None:
                            imgs[i] = img
            result[cid] = imgs
        return result
