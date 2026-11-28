# OpenMVIS: A Multi-Visual-Inertial Sensor Calibration Benchmark
# Copyright (C) 2026 Yulin Yang
# Copyright (C) 2026 Yuantai-Z
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
# SPDX-FileCopyrightText: 2026 Yuantai-Z
# SPDX-License-Identifier: GPL-3.0-only
"""Read a ROS1 bag, apply a calibration, write a corrected bag + provenance.

This is the one implementation of "apply a calibration to a ROS1 bag": `mvis apply-calib`,
`scripts/recalib_batch.py` and `tools/ros1_recalib` all call `apply_calibration`. The
per-operation switches, strict input checks, padded-row undistortion, integer-ns t_ga
interpolation and per-connection output come from Yuantai's tools/ros1_recalib exporter.
"""
from __future__ import annotations
from array import array
import contextlib
from dataclasses import dataclass
import heapq
import logging
import os
import numpy as np

try:
    from rosbags.rosbag1 import Reader, Writer
    from rosbags.typesys import Stores, get_typestore
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by the extra-less install
    # ROS1 bag I/O is the `ros1` extra; keep the module importable so
    # `import mvis.recalib.apply` (and anything that walks the package) works
    # on a base `pip install pymvis`, and fail with a legible message only if
    # someone actually calls into the bag readers/writers.
    _ROSBAGS_ERROR = exc

    def _missing_rosbags(*_args, **_kwargs):
        raise ModuleNotFoundError(
            "ROS1 bag support requires the 'rosbags' package: "
            "pip install 'pymvis[ros1]'"
        ) from _ROSBAGS_ERROR

    Reader = Writer = get_typestore = _missing_rosbags
    Stores = None

from mvis.recalib.sensor_map import build_topic_plan
from mvis.recalib.bagio import header_ns, raw_stamp_ns
from mvis.recalib.correct_imu import correct_imu_msg
from mvis.recalib.restamp import shifted_time_ns, set_header_stamp_ns
from mvis.recalib.provenance import write_provenance

log = logging.getLogger("mvis.recalib.apply")

_ROS1_TIME_END_NS = (2**32) * 1_000_000_000      # ROS1 time is uint32 seconds
_IMU_TYPE = "sensor_msgs/msg/Imu"
_IMAGE_TYPE = "sensor_msgs/msg/Image"
_CAMERA_TYPES = (_IMAGE_TYPE, "sensor_msgs/msg/CompressedImage")
_HEADER_FIRST = (_IMU_TYPE, *_CAMERA_TYPES)   # serialized stamp at bytes 4-12
# distortion coefficient counts each cv2 undistortion path accepts
_UNDISTORT_SIZES = {"radtan": (4, 5, 8, 12, 14), "plumb_bob": (4, 5, 8, 12, 14),
                    "equidistant": (4,), "fisheye": (4,)}


@dataclass(frozen=True)
class _AccelSeries:
    """Raw accel of one IMU topic, sorted by header stamp. Times are integer ns relative to
    `origin_ns`, then float: absolute ns near 1.7e18 have a float64 spacing of 256 ns."""
    origin_ns: int
    t_rel: np.ndarray        # (n,) float ns since origin_ns, nondecreasing
    accel: np.ndarray        # (n, 3)
    t_ga: float              # s


def shifted_accel(T_ns, A, t_query_ns):
    """Linear-interpolated accel (3,) at `t_query_ns` from a header-stamped raw
    series (edge-clamped) — undoes the gyro<->accel offset: corrected accel at
    stamp t is the measured accel at t + t_ga. Pass times relative to a common
    origin when they are absolute ROS stamps (float64 resolves 1.7e18 ns to 256 ns)."""
    return np.array([np.interp(t_query_ns, T_ns, A[:, i]) for i in range(3)])


def _prepass(reader, plan, tga_by_topic, td, ts, *, reject_duplicates):
    """Pass 1, over the mapped connections only (none when neither td nor t_ga is on):
    - {topic: _AccelSeries} for the IMU topics that need the t_ga shift. Stamps are sorted
      (np.interp needs increasing x); equal stamps raise when `reject_duplicates`.
    - with `td`, {connection id: suffix minimum of the output times in read order}: the
      earliest output time any not-yet-read message of that connection can still have.
    Image payloads are read here but not kept (the stamp comes from the raw header)."""
    rows = {t: [] for t in tga_by_topic}
    times = {}
    wanted = plan if td else rows
    conns = [c for c in reader.connections if c.topic in wanted]
    if conns:
        times = {c.id: array("q") for c in conns} if td else {}   # 8 B per message
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = None
            if conn.topic in rows:
                msg = ts.deserialize_ros1(raw, conn.msgtype)
                a = msg.linear_acceleration
                rows[conn.topic].append((header_ns(msg), (a.x, a.y, a.z)))
            if td:
                times[conn.id].append(shifted_time_ns(_base_ns(conn, t_ns, raw, msg, ts),
                                                      plan[conn.topic].td))
    series = {}
    for topic, rr in rows.items():
        if not rr:
            continue
        rr.sort(key=lambda r: r[0])
        stamps = [r[0] for r in rr]
        if reject_duplicates and any(a == b for a, b in zip(stamps, stamps[1:])):
            raise ValueError(f"duplicate header timestamps on t_ga topic: {topic}")
        origin = stamps[0]
        series[topic] = _AccelSeries(origin, np.asarray([s - origin for s in stamps], float),
                                     np.asarray([r[1] for r in rr], float), tga_by_topic[topic])
    suffix_min = {cid: np.minimum.accumulate(np.frombuffer(v, np.int64)[::-1])[::-1]
                  for cid, v in times.items() if v}
    return series, suffix_min


def _base_ns(conn, t_ns, raw, msg, ts):
    """The stamp td shifts: the header stamp, or the record time for header-less messages.
    Header-first types read it from the serialized bytes; others are deserialized."""
    if conn.msgtype in _HEADER_FIRST:
        return raw_stamp_ns(raw)
    if msg is None:
        msg = ts.deserialize_ros1(raw, conn.msgtype)
    return header_ns(msg) if getattr(msg, "header", None) is not None else t_ns


def _check_connections(connections, plan, undistort):
    """Strict mode: every mapped topic is present with a supported message type, and every
    camera to undistort has a model and coefficient count cv2 can undistort."""
    missing = set(plan) - {c.topic for c in connections}
    if missing:
        raise ValueError(f"mapped topics absent from input bag: {sorted(missing)}")
    for conn in connections:
        tp = plan.get(conn.topic)
        if tp is None:
            continue
        if tp.kind == "imu":
            allowed = (_IMU_TYPE,)
        elif undistort:
            allowed = (_IMAGE_TYPE,)
        else:
            allowed = _CAMERA_TYPES      # a restamp only needs the header
        if conn.msgtype not in allowed:
            raise ValueError(f"topic {conn.topic!r} has type {conn.msgtype!r}; "
                             f"expected {' or '.join(allowed)}")
        if tp.kind == "cam" and undistort:
            cam = tp.cam
            if cam.model not in _UNDISTORT_SIZES or cam.dist.size not in _UNDISTORT_SIZES[cam.model]:
                raise ValueError(f"unsupported undistortion model/coefficients for "
                                 f"{cam.name}: {cam.model}")
            if np.any(np.asarray(cam.K)[:2] <= 0):
                raise ValueError(f"camera {cam.name} focal lengths must be positive")


def _imu_values(msg):
    w, a = msg.angular_velocity, msg.linear_acceleration
    return [w.x, w.y, w.z, a.x, a.y, a.z]


def _add_connections(writer, connections):
    """One output connection per distinct source connection: each keeps its own msgdef,
    md5sum, callerid and latching (custom schemas and multiple publishers survive).
    Returns {source connection id: output connection}."""
    out, identical = {}, {}
    for conn in connections:
        ext = conn.ext
        definition = conn.msgdef.data
        signature = (conn.topic, conn.msgtype, definition, conn.digest,
                     getattr(ext, "callerid", None), getattr(ext, "latching", None))
        # rosbags refuses two connections with identical metadata
        if signature not in identical:
            identical[signature] = writer.add_connection(
                conn.topic, conn.msgtype, msgdef=definition, md5sum=conn.digest,
                callerid=signature[4], latching=signature[5])
        out[conn.id] = identical[signature]
    return out


class _OrderedWriter:
    """Writes records in the order a stable sort by output time would give, holding a record
    only while an unread message could still sort before it.

    Messages arrive in bag record-time order (rosbags merges time-sorted indexes), so every
    unread message that keeps its record time sorts at or after the current one. A td-shifted
    connection's unread messages sort at or after its suffix minimum from the pre-pass. A
    record at or below both bounds is final; ties keep read order, as the stable sort did.
    Memory is the records inside that window, not the bag."""

    def __init__(self, writer, out_conns, suffix_min):
        self._writer, self._out = writer, out_conns
        self._suffix_min = suffix_min
        self._cursor = dict.fromkeys(suffix_min, 0)
        self._pending = []          # heap of (t_new_ns, read index, connection id, raw)
        self._seq = 0
        self._last_t = None

    def push(self, conn_id, t_read_ns, t_new_ns, raw):
        if self._last_t is not None and t_read_ns < self._last_t:
            raise RuntimeError("bag reader yielded record times out of order")
        self._last_t = t_read_ns
        heapq.heappush(self._pending, (t_new_ns, self._seq, conn_id, raw))
        self._seq += 1
        bound = t_read_ns
        if conn_id in self._cursor:
            self._cursor[conn_id] += 1
        for cid, i in self._cursor.items():
            arr = self._suffix_min[cid]
            if i < len(arr) and arr[i] < bound:
                bound = int(arr[i])
        self._drain(bound)

    def _drain(self, bound=None):
        while self._pending and (bound is None or self._pending[0][0] <= bound):
            t_ns, _, conn_id, raw = heapq.heappop(self._pending)
            self._writer.write(self._out[conn_id], t_ns, raw)

    def finish(self):
        self._drain()


def apply_calibration(in_bag, out_bag, rig, mapping, base_imu_topic, *,
                      remove_bias=True, undistort=False, imu_intrinsics=True,
                      td=True, t_ga=True, strict=False, provenance_extra=None) -> dict:
    """Read a ROS1 bag, apply `rig` to the topics in `mapping` (topic -> sensor name) and
    write a corrected bag + `<out_bag>.recalib.yaml` provenance sidecar.

    Operations, each independent:
      imu_intrinsics  inverse IMU model (scale, skew, rotation, Tg);
      remove_bias     subtract the saved static gyro/accel biases;
      t_ga            accel re-interpolated at header + t_ga, before the inverse model;
      td              header stamp and record time := header stamp + sensor td (integer ns);
                      off keeps both original stamps;
      undistort       mono8/8UC1 images (row padding allowed) through cv2.
    A mapped message with no operation active is copied byte for byte.

    `strict` (tools/ros1_recalib) rejects instead of passing through: mapped topics missing
    or empty, unexpected message types, unsupported undistortion models or encodings,
    duplicate header stamps on a t_ga topic, non-finite corrected IMU values.
    `provenance_extra` adds or overrides sidecar keys.

    Returns a summary: message counts, per-operation counts, corrected and passthrough
    topics, t_ga metadata and the output paths. The bag is streamed: with td or t_ga on, a
    first pass reads the mapped topics (IMU series for t_ga, output stamps for td), then a
    second pass writes each message as soon as no unread message can sort before it, in the
    same global output-time order a full sort gives. A failed run removes the partial bag.
    """
    ts = get_typestore(Stores.ROS1_NOETIC)
    plan = build_topic_plan(rig, mapping, base_imu_topic)
    if undistort:
        from mvis.recalib.undistort import check_undistortable, undistort_image_msg
        check_undistortable(plan)
    # gyro<->accel time offset: topics whose calib carries t_ga get their accel
    # channel re-interpolated onto the gyro timeline (a'(t) = a_m(t + t_ga))
    # BEFORE the inverse intrinsic model (the Tg coupling wants aligned accel).
    tga_by_topic = {t: tp.imu.t_ga for t, tp in plan.items()
                    if t_ga and tp.kind == "imu" and tp.imu is not None
                    and abs(tp.imu.t_ga) > 1e-9}
    ops = {"imu_intrinsics": imu_intrinsics, "remove_bias": remove_bias, "td": td,
           "t_ga": t_ga, "undistort": undistort}
    counts = {"n_imu": 0, "n_cam": 0, "n_pass": 0}
    topic_counts = dict.fromkeys(plan, 0)
    op_counts = dict.fromkeys(ops, 0)
    corrected = set()
    with Reader(in_bag) as reader:
        connections = list(reader.connections)
        if strict:
            _check_connections(connections, plan, undistort)
            sizes = dict.fromkeys(plan, 0)
            for c in connections:
                if c.topic in sizes:
                    sizes[c.topic] += c.msgcount
            empty = sorted(t for t, n in sizes.items() if not n)
            if empty:
                raise ValueError(f"mapped topics contain no messages: {empty}")
        series, suffix_min = _prepass(reader, plan, tga_by_topic, td, ts,
                                      reject_duplicates=strict)
        if series:
            log.info("recalib: t_ga accel shift on %s",
                     {t: f"{s.t_ga * 1e3:+.3f} ms" for t, s in series.items()})
        tga_meta = {t: {"offset_seconds": s.t_ga, "samples": len(s.t_rel),
                        "edge_clamped_messages": 0} for t, s in series.items()}
        writer = Writer(out_bag)
        writer.open()                    # exclusive create: never replaces a file
        try:
            ordered = _OrderedWriter(writer, _add_connections(writer, connections), suffix_min)
            for conn, t_ns, raw in reader.messages():
                tp = plan.get(conn.topic)
                if tp is None:                                  # passthrough
                    ordered.push(conn.id, t_ns, t_ns, raw); counts["n_pass"] += 1
                    continue
                counts["n_imu" if tp.kind == "imu" else "n_cam"] += 1
                topic_counts[conn.topic] += 1
                if tp.kind == "imu":
                    active = [n for n in ("imu_intrinsics", "remove_bias") if ops[n]]
                else:
                    active = ["undistort"] if undistort else []
                if conn.topic in series:
                    active.append("t_ga")
                if td:
                    active.append("td")
                if not active:
                    ordered.push(conn.id, t_ns, t_ns, raw)
                    continue
                msg = ts.deserialize_ros1(raw, conn.msgtype)
                if tp.kind == "imu":
                    if conn.topic in series:            # undo the gyro<->accel offset
                        s = series[conn.topic]
                        query = (header_ns(msg) - s.origin_ns) + s.t_ga * 1e9
                        if query < s.t_rel[0] or query > s.t_rel[-1]:
                            tga_meta[conn.topic]["edge_clamped_messages"] += 1
                        a = shifted_accel(s.t_rel, s.accel, query)
                        (msg.linear_acceleration.x, msg.linear_acceleration.y,
                         msg.linear_acceleration.z) = map(float, a)
                    if imu_intrinsics or remove_bias:
                        correct_imu_msg(msg, tp.imu, remove_bias=remove_bias,
                                        imu_intrinsics=imu_intrinsics)
                        if strict and not np.all(np.isfinite(_imu_values(msg))):
                            raise ValueError(f"non-finite corrected IMU sample on {conn.topic}")
                elif undistort:
                    undistort_image_msg(msg, tp.cam, strict=strict)
                if td:
                    # Re-stamp from the HEADER clock, not the bag record time: the solver
                    # reads header stamps, and on live-recorded bags record time carries
                    # the per-topic receive latency (t265: cams +4 ms, IMU +9 ms) — using
                    # t_ns here bakes the DIFFERENTIAL latency into the corrected headers
                    # (recal td = L_imu - L_cam, ~+4.5 ms on the t265 rig). Header-less
                    # messages fall back to t_ns.
                    base_ns = header_ns(msg) if getattr(msg, "header", None) is not None else t_ns
                    t_new = shifted_time_ns(base_ns, tp.td)
                    if not 0 <= t_new < _ROS1_TIME_END_NS:
                        raise ValueError(f"corrected timestamp outside ROS1 range on {conn.topic}")
                    set_header_stamp_ns(msg, t_new)
                else:
                    t_new = t_ns
                ordered.push(conn.id, t_ns, t_new, ts.serialize_ros1(msg, conn.msgtype))
                corrected.add(conn.topic)
                for name in active:
                    op_counts[name] += 1
            ordered.finish()
            writer.close()
        except BaseException:
            with contextlib.suppress(Exception):
                writer.close()
            with contextlib.suppress(FileNotFoundError):
                os.unlink(out_bag)       # created above by this call
            raise

    observed = {c.topic for c in connections if c.msgcount}
    passthrough = sorted(observed - corrected)
    extra = {"operation_counts": op_counts, "message_counts": counts,
             "mapped_topic_counts": topic_counts,
             "t_ga": {"method": "linear interpolation on sorted raw header stamps",
                      "edges": "clamp",
                      "duplicate_stamps": "reject" if strict else "keep",
                      "topics": tga_meta}}
    extra.update(provenance_extra or {})
    prov = write_provenance(
        str(out_bag), source_bag=str(in_bag),
        calib_path=getattr(rig, "_src_path", "<unknown>"),
        mapping=mapping, remove_bias=remove_bias, undistort=undistort,
        imu_intrinsics=imu_intrinsics, td=td, t_ga=t_ga,
        base_imu_topic=base_imu_topic, corrected_topics=sorted(corrected),
        passthrough_topics=passthrough, extra=extra)
    log.info("recalib: %d imu, %d cam, %d passthrough → %s",
             counts["n_imu"], counts["n_cam"], counts["n_pass"], out_bag)
    return {**counts, "out_bag": out_bag, "provenance": prov,
            "operation_counts": op_counts, "mapped_topic_counts": topic_counts,
            "corrected_topics": sorted(corrected), "passthrough_topics": passthrough,
            "t_ga": tga_meta}
