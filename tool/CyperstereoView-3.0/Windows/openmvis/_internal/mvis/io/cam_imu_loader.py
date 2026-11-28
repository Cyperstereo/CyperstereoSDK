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
"""Unified cam-IMU data loader.

One library entry point that loads, for a calibration sequence, the per-camera
board/feature detections + per-IMU (gyro, accel) arrays — for BOTH the camera and
cam-IMU pipelines, across ROS1 bags and EuRoC `mav0` folders. It handles:
  * mono16 images (decoded by io.image_decode, like every reader),
  * header-vs-bag clock mismatch (cam AND imu windowed/stamped on the HEADER clock,
    so they always agree — TUM-VI raw needed this), and
  * EuRoC `mav0` (cam dirs + imuN/data.csv) without converting to a bag.

Detection flows through the shared `io.frame_detector` front-end (the SAME
`board.detect` -> FrameObservations path the camera-calib pipeline uses) and reuses
`io.detection_cache.DetectionCache`, so a `cache_dir` skips re-detection on re-runs.

Returns per cam ``(t_cam, ids_list, uv_list)`` (header
seconds) and per imu ``(t, w, a)``. The cam dicts ALSO carry
``frames`` (the real FrameObservations — the ragged arrays are a derived view
of them, byte-equal by construction) and ``cache_path`` (the bag's capture
db, one file per bag+window, holding frames + IMU streams).
"""
from __future__ import annotations
import logging
import os
import numpy as np

from mvis.io.image_decode import image_file_to_gray, ros_image_to_gray
from mvis.io.imu_reader import euroc_imu_rows, ros1_imu_sample
from mvis.io.rosbag_reader import open_bag
from mvis.io.ros_stamp import header_stamp_s
from mvis.io.capture_key import board_signature, capture_key
from mvis.io.frame_detector import detect_frame, frames_to_arrays

logger = logging.getLogger("mvis.io.cam_imu_loader")


def _cache(cache_dir, basename, sig):
    """DetectionCache for this source+signature, or None when disabled.

    The signature is hashed into the filename so different topics / windows on the
    same bag get distinct cache files (no cross-topic collisions in sweeps). The
    dots of the basename become underscores: the cache's ``with_suffix`` used to
    cut ``.bag_<hash>`` off, so every signature of one bag shared one file (the
    ELP and ELP+T265 rows overwrote each other's capture).
    """
    if not cache_dir:
        return None
    import hashlib
    from mvis.io.detection_pipeline import _make_cache
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.md5(sig.encode()).hexdigest()[:8]
    stem = str(basename).replace(".", "_")
    return _make_cache(os.path.join(cache_dir, f"cam_imu_{stem}_{h}"))


def _stream_store(cache_dir):
    """The shared per-stream store (io.stream_store) in ``cache_dir``, or None."""
    if not cache_dir:
        return None
    from mvis.io.stream_store import StreamDetectionStore
    return StreamDetectionStore(cache_dir)


def _thumb_dir(cache):
    """Annotated per-frame thumbnails live beside the detection cache file
    (`<cache>.thumbs/cam<N>/` + index.csv); None without a cache."""
    path = getattr(cache, "_path", None) if cache is not None else None
    return f"{path}.thumbs" if path else None


def _thumbs_for(td, cid):
    """The thumbnail dir for the report if camera `cid` has an index there."""
    from mvis.io.detection_pipeline import THUMB_INDEX_NAME
    if td and os.path.exists(os.path.join(td, f"cam{cid}", THUMB_INDEX_NAME)):
        return td
    return None


class _ThumbWriter:
    """Writes the cam-IMU loaders' thumbnails on a fresh detection pass."""

    def __init__(self, cache, board):
        self.dir, self.board, self.n = _thumb_dir(cache), board, {}

    def __call__(self, cid, img, f):
        if self.dir is None:
            return
        from mvis.io.detection_pipeline import write_thumbnail
        idx = self.n.get(cid, 0)
        self.n[cid] = idx + 1
        write_thumbnail(self.dir, self.board, cid, idx, img, f.observations,
                        f.timestamp, reset=(idx == 0))


def _sig(source, names, board, t0, window, min_obs) -> str:
    """Cache signature: the capture key (source, camera topics or dirs, window
    ``(t0, length)``, ``min_obs``) and the board fingerprint."""
    return (capture_key(os.path.abspath(source), names, window=(t0, window), min_obs=min_obs)
            + f"|{board_signature(board)}")


def load_ros1(bag_path, cam_topics, imu_topics, board, t0=0.0, window=1e9,
              min_obs=1, cache_dir=None, abs_window=None):
    """Load a ROS bag (ROS1 ``.bag``, or ROS2 folder / ``.db3`` / ``.mcap``, see
    `io.rosbag_reader`) on the HEADER clock (consistent cam+imu windowing).

    cam_topics: list[str] image topics; imu_topics: list[str] (first = base).
    Returns (cams, imus): cams = [{topic,t_cam,ids_list,uv_list}], imus =
    [{topic,name,t,w,a}] (header seconds). Window is [first_cam_header+t0, +window].
    With cache_dir, detections are cached/reused (DetectionCache); IMU is
    always re-read (cheap, no image decode).
    abs_window: optional (lo, hi) absolute header-second override; when set,
    bypasses t0/window computation and uses the provided window directly.
    """
    cam_topics = list(cam_topics); imu_topics = list(imu_topics)

    # pass 1: earliest camera HEADER stamp -> window origin (header clock)
    with open_bag(bag_path)[0] as r:
        cons = [c for c in r.connections if c.topic in set(cam_topics)]
        first = None
        for conn, _bt, raw in r.messages(connections=cons):
            t = header_stamp_s(raw, _bt)
            first = t if first is None else min(first, t)
            break  # messages are time-ordered by bag-time; header≈monotonic enough
    lo, hi = first + t0, first + t0 + window
    if abs_window is not None:
        lo, hi = float(abs_window[0]), float(abs_window[1])

    # --- camera detections (shared front-end, cached) ---
    # Manual path keeps today's key (t0, window); motion path folds the resolved
    # window so motion windows get distinct cache keys.
    if abs_window is not None:
        sig = _sig(bag_path, cam_topics, board, t0=lo, window=(hi - lo), min_obs=min_obs)
    else:
        sig = _sig(bag_path, cam_topics, board, t0, window, min_obs)
    cache = _cache(cache_dir, os.path.basename(str(bag_path)), sig)
    frames_by_cam = None
    fresh_detect = False
    if cache is not None and cache.is_valid(sig, {}):
        frames_by_cam, _hist, _sz = cache.load(sig, {})
    # Shared per-stream store: detections another run (camera-only included) made
    # of these topics, filtered to this window and min_obs.
    store = _stream_store(cache_dir)
    if frames_by_cam is None and store is not None:
        frames_by_cam = store.get_all(bag_path, cam_topics, board,
                                      lo=lo, hi=hi, min_obs=min_obs)
        fresh_detect = cache is not None and frames_by_cam is not None
        if frames_by_cam is not None:
            logger.warning("thumbnails not available: detections from shared store")
    if frames_by_cam is None:
        fresh_detect = cache is not None
        thumb = _ThumbWriter(cache, board)
        frames_by_cam = {c: [] for c in range(len(cam_topics))}
        raw_by_cam = {c: [] for c in range(len(cam_topics))}     # >= 1 corner, for the store
        read_by_cam = {c: 0 for c in range(len(cam_topics))}
        topic_to_cid = {tp: c for c, tp in enumerate(cam_topics)}
        jpeg_warned = set()
        reader, deserialize = open_bag(bag_path)
        with reader as r:
            cons = [c for c in r.connections if c.topic in set(cam_topics)]
            for conn, _bt, raw in r.messages(connections=cons):
                t = header_stamp_s(raw, _bt)       # before decoding: skip out-of-window frames
                if not (lo <= t <= hi):
                    continue
                m = deserialize(raw, conn.msgtype)
                cid = topic_to_cid[conn.topic]
                img = ros_image_to_gray(m, conn.topic, jpeg_warned)
                f = detect_frame(img, board, cid, t)
                thumb(cid, img, f)
                read_by_cam[cid] += 1
                if f.observations:
                    raw_by_cam[cid].append(f)
                if len(f.observations) >= min_obs:
                    frames_by_cam[cid].append(f)
        if store is not None:
            for cid, tp in enumerate(cam_topics):
                store.put(bag_path, tp, board, raw_by_cam[cid],
                          frames_read=read_by_cam[cid], lo=lo, hi=hi)

    cams = []
    for c, tp in enumerate(cam_topics):
        t_cam, ids_list, uv_list = frames_to_arrays(frames_by_cam.get(c, []))
        cams.append(dict(topic=tp, t_cam=t_cam, ids_list=ids_list, uv_list=uv_list,
                         frames=frames_by_cam.get(c, []),
                         thumb_dir=_thumbs_for(_thumb_dir(cache), c),
                         cache_path=(str(getattr(cache, "_path", "")) or None
                                     if cache is not None else None)))

    # --- IMU (always read; image decode skipped) ---
    imus_final = _ros1_imu(bag_path, imu_topics, window=(lo, hi))
    # capture-completeness: the save waits for the IMU read so the capture
    # core declares + stores the bag's IMU streams (result_imu FK target).
    if fresh_detect:
        cache.save(frames_by_cam, sig, {}, imus=imus_final)
    return cams, imus_final


def load_euroc(mav0_dir, cam_subdirs, imu_subdirs, board, t0=0.0, window=1e9,
               min_obs=1, cache_dir=None, abs_window=None):
    """Load an EuRoC `mav0` sequence (cam dirs + imuN/data.csv). Same returns.

    abs_window: optional (lo, hi) absolute header-second override; when set,
    bypasses t0/window computation and uses the provided window directly.
    """
    import csv

    # window origin from first cam stamp
    first = None
    for cs in cam_subdirs:
        for row in csv.reader(open(os.path.join(mav0_dir, cs, "data.csv"))):
            if row and not row[0].startswith("#"):
                first = int(row[0]) * 1e-9; break
        if first is not None:
            break
    lo, hi = first + t0, first + t0 + window
    if abs_window is not None:
        lo, hi = float(abs_window[0]), float(abs_window[1])

    # --- camera detections (shared front-end, cached) ---
    # Manual path keeps today's key (t0, window); motion path folds the resolved
    # window so motion windows get distinct cache keys.
    if abs_window is not None:
        sig = _sig(mav0_dir, cam_subdirs, board, t0=lo, window=(hi - lo), min_obs=min_obs)
    else:
        sig = _sig(mav0_dir, cam_subdirs, board, t0, window, min_obs)
    cache = _cache(cache_dir, os.path.basename(os.path.abspath(mav0_dir)) or "mav0", sig)
    frames_by_cam = None
    fresh_detect = False
    if cache is not None and cache.is_valid(sig, {}):
        frames_by_cam, _hist, _sz = cache.load(sig, {})
    if frames_by_cam is None:
        fresh_detect = cache is not None
        thumb = _ThumbWriter(cache, board)
        frames_by_cam = {}
        for cid, cs in enumerate(cam_subdirs):
            frames_by_cam[cid] = []
            ddir = os.path.join(mav0_dir, cs, "data")
            for row in csv.reader(open(os.path.join(mav0_dir, cs, "data.csv"))):
                if not row or row[0].startswith("#"):
                    continue
                t = int(row[0]) * 1e-9
                if t < lo or t > hi:
                    continue
                fp = os.path.join(ddir, row[1].strip())
                img = image_file_to_gray(fp)
                if img is None:
                    raise ValueError(f"unreadable image: {fp}")
                f = detect_frame(img, board, cid, t)
                thumb(cid, img, f)
                if len(f.observations) >= min_obs:
                    frames_by_cam[cid].append(f)

    cams = []
    for cid, cs in enumerate(cam_subdirs):
        t_cam, ids_list, uv_list = frames_to_arrays(frames_by_cam.get(cid, []))
        cams.append(dict(topic=cs, t_cam=t_cam, ids_list=ids_list, uv_list=uv_list,
                         frames=frames_by_cam.get(cid, []),
                         thumb_dir=_thumbs_for(_thumb_dir(cache), cid),
                         cache_path=(str(getattr(cache, "_path", "")) or None
                                     if cache is not None else None)))

    imus = _euroc_imu(mav0_dir, imu_subdirs, window=(lo, hi))
    # capture-completeness: the save waits for the IMU read so the capture
    # core declares + stores the sequence's IMU streams (result_imu FK target).
    if fresh_detect:
        cache.save(frames_by_cam, sig, {}, imus=imus)
    return cams, imus


def _finalize_imu(d):
    t = np.asarray(d["t"], float); o = np.argsort(t)
    d["t"] = t[o]; d["w"] = np.asarray(d["w"], float).reshape(-1, 3)[o]
    d["a"] = np.asarray(d["a"], float).reshape(-1, 3)[o]
    return d


def _ros1_imu(bag_path, imu_topics, window=None):
    """IMU streams [{topic,name,t,w,a}] of a ROS1 or ROS2 bag on the header clock, sorted by
    stamp. ``window=(lo, hi)`` keeps the samples within 0.5 s of it; None keeps the whole bag."""
    reader, deserialize = open_bag(bag_path)
    imus = {tp: dict(topic=tp, name=tp.strip("/").split("/")[0], t=[], w=[], a=[])
            for tp in imu_topics}
    with reader as r:
        cons = [c for c in r.connections if c.topic in set(imu_topics)]
        for conn, _bt, raw in r.messages(connections=cons):
            t = header_stamp_s(raw, _bt)
            if window is not None and (t < window[0] - 0.5 or t > window[1] + 0.5):
                continue
            m = deserialize(raw, conn.msgtype)
            d = imus[conn.topic]
            d["t"].append(t)
            w, a = ros1_imu_sample(m)
            d["w"].append(w)
            d["a"].append(a)
    return [_finalize_imu(imus[tp]) for tp in imu_topics]


def _euroc_imu(mav0_dir, imu_subdirs, window=None):
    """IMU streams [{topic,name,t,w,a}] from ``<mav0_dir>/<subdir>/data.csv``, sorted by stamp
    (each subdir is also the stream's topic and name). ``window=(lo, hi)`` keeps the samples
    within 0.5 s of it; None keeps them all."""
    imus = []
    for sub in imu_subdirs:
        d = dict(topic=sub, name=sub, t=[], w=[], a=[])
        for t, w, a in euroc_imu_rows(os.path.join(mav0_dir, sub, "data.csv")):
            if window is None or window[0] - 0.5 <= t <= window[1] + 0.5:
                d["t"].append(t); d["w"].append(w); d["a"].append(a)
        imus.append(_finalize_imu(d))
    return imus


def _read_euroc_imu_only(mav0_dir, imu_subdirs):
    """Read all IMU data from an EuRoC mav0 dir (no windowing, no images). Accepts a bare
    subdir ("imu0") or a topic-style name ("/imu0"), mapped like load_euroc, so a leading "/"
    cannot make os.path.join treat the name as an absolute path."""
    return _euroc_imu(mav0_dir, [ims.strip("/").split("/")[0] for ims in imu_subdirs])


def read_imu_only(bag_path, imu_topics, fmt="ros1"):
    """Whole-bag IMU streams [{topic,name,t,w,a}] (header seconds), no image
    decode — feeds segment_select before windowing."""
    if fmt == "euroc":
        return _read_euroc_imu_only(bag_path, imu_topics)
    return _ros1_imu(bag_path, imu_topics)


def extract_bag(bag_path, cam_topic, imu_topic, t0_off, window, cache_dir,
                board):
    """Single-camera, single-IMU convenience over `load_ros1`.

    Returns the flat 6-tuple ``(t_cam, ids_list, uv_list, t_imu, w_m, a_m)`` (header
    seconds; cam and IMU windowed on the same header clock) for one camera topic and
    one IMU topic of a ROS1 bag. The detection cache is keyed on the board fingerprint.
    """
    cams, imus = load_ros1(bag_path, [cam_topic], [imu_topic], board,
                           t0=t0_off, window=window, min_obs=1,
                           cache_dir=cache_dir)
    c, im = cams[0], imus[0]
    return c["t_cam"], c["ids_list"], c["uv_list"], im["t"], im["w"], im["a"]
