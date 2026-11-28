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
"""Dataset inspection (`mvis inspect`) — Stage 1 data qualification.

Reads only metadata plus a short sample of each stream, never the whole
recording, and reports what a calibration needs to know: container format,
which MVIS modes can read it, camera streams (rate, resolution, encoding), IMU
streams (rate, units sanity, stamp order) and the camera/IMU time overlap.
Every finding carries a stable ``id`` and a status ``pass | warn | fail``.
"""
from __future__ import annotations

import csv
import glob
import os
from dataclasses import asdict, dataclass, field

import numpy as np

from mvis.doctor.findings import Finding, format_finding, worst_status  # noqa: F401  (Finding re-exported)
from mvis.io.imu_reader import euroc_imu_values, ros1_imu_sample
from mvis.io.ros_stamp import header_stamp_s

IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
IMU_TYPE = "sensor_msgs/msg/Imu"
IMU_SAMPLE = 2000               # IMU messages read per topic for unit/stamp checks
GRAVITY = 9.80665
MIN_DURATION_S = 30.0           # cam-IMU needs a long enough excitation window
FORMAT_MODES = {"ros1": ["cam", "cam-imu"], "euroc": ["cam", "cam-imu"],
                "ros2": ["cam", "cam-imu"], "images": ["cam"]}
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm")


@dataclass
class Stream:
    name: str
    kind: str                   # camera | imu | other
    count: int = 0
    rate_hz: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    info: dict = field(default_factory=dict)


@dataclass
class Report:
    path: str
    format: str
    supported_modes: list
    duration_s: float | None = None
    streams: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    status: str = "pass"

    def find(self, *a):
        self.findings.append(Finding(*a))

    def finish(self):
        self.status = worst_status(f.status for f in self.findings)
        return self

    def to_dict(self):
        return asdict(self)


def detect_format(path: str) -> str:
    if os.path.isfile(path):
        if path.endswith(".bag"):
            return "ros1"
        if path.endswith((".mcap", ".db3")):
            return "ros2"
        raise ValueError(f"unknown file type: {path}")
    if os.path.isdir(os.path.join(path, "mav0")) or glob.glob(os.path.join(path, "cam*/data.csv")):
        return "euroc"
    if os.path.isfile(os.path.join(path, "metadata.yaml")):
        return "ros2"
    if any(f.lower().endswith(IMG_EXT) for f in os.listdir(path)):
        return "images"
    raise ValueError(f"cannot recognise dataset layout: {path}")


def _imu_checks(rep, name, t, w, a):
    """Stamp order, rate and units from a sample of IMU messages."""
    t = np.asarray(t, float)
    dt = np.diff(t)
    if dt.size and np.any(dt <= 0):
        rep.find("imu.stamps_monotonic", name, "fail",
                 f"{int(np.sum(dt <= 0))} non-increasing stamps in the first {t.size} samples")
    else:
        rep.find("imu.stamps_monotonic", name, "pass", "stamps strictly increasing")
    if dt.size > 10:
        med = float(np.median(dt))
        gaps = int(np.sum(dt > 3 * med))
        if gaps:
            rep.find("imu.gaps", name, "warn", f"{gaps} gaps > 3x the nominal period")
    a_norm = float(np.median(np.linalg.norm(np.asarray(a, float), axis=1)))
    if abs(a_norm - GRAVITY) < 1.5:
        rep.find("imu.accel_units", name, "pass", f"|a| median {a_norm:.2f} m/s^2")
    elif abs(a_norm - 1.0) < 0.15:
        rep.find("imu.accel_units", name, "fail",
                 f"|a| median {a_norm:.2f}: accelerometer is in g, MVIS expects m/s^2")
    elif a_norm < 1e-6:
        rep.find("imu.accel_units", name, "warn", "accelerometer is all zero: gyro-only IMU?")
    else:
        rep.find("imu.accel_units", name, "warn", f"|a| median {a_norm:.2f} m/s^2 is not ~g")
    w_max = float(np.max(np.abs(np.asarray(w, float)))) if len(w) else 0.0
    if w_max > 35.0:          # ~2000 deg/s in rad/s: almost surely deg/s
        rep.find("imu.gyro_units", name, "fail",
                 f"|w| max {w_max:.1f}: gyro looks like deg/s, MVIS expects rad/s")


def _compressed_info(rep, topic, m):
    """Stream info for a CompressedImage topic, from decoding its first frame as the readers do."""
    from mvis.io.compressed_image import as_raw_image, compressed_codec
    info = {"compressed": m.format}
    try:
        raw = as_raw_image(m, topic, {topic})       # the finding below replaces the log warning
        info.update(width=raw.width, height=raw.height, encoding=raw.encoding)
        if compressed_codec(m) == "jpeg":
            rep.find("camera.lossy", topic, "warn",
                     "JPEG images are lossy and can shift detected corners; prefer raw or PNG")
    except ValueError as e:
        rep.find("camera.first_image", topic, "fail", str(e))
    return info


def inspect_ros1(path: str, rep: Report):
    """A ROS1 or ROS2 bag (`io.rosbag_reader`)."""
    from mvis.io.rosbag_reader import open_bag
    reader, deserialize = open_bag(path)
    with reader as r:
        rep.duration_s = r.duration * 1e-9
        by_topic = {}
        for c in r.connections:
            by_topic.setdefault(c.topic, []).append(c)
        for topic, conns in sorted(by_topic.items()):
            mt = conns[0].msgtype
            kind = "camera" if mt in IMAGE_TYPES else "imu" if mt == IMU_TYPE else "other"
            n = sum(c.msgcount for c in conns)
            s = Stream(topic, kind, n)
            if kind == "other":
                rep.streams.append(s)
                continue
            limit = 2 if kind == "camera" else IMU_SAMPLE
            t, w, a = [], [], []
            for conn, bt, raw in r.messages(connections=conns):
                m = deserialize(raw, conn.msgtype)
                # the readers' clock: header stamp from the raw bytes, record time when it is 0
                t.append(header_stamp_s(raw, bt))
                if kind == "camera":
                    if not s.info:
                        s.info = ({"width": int(m.width), "height": int(m.height),
                                   "encoding": m.encoding}
                                  if mt == IMAGE_TYPES[0] else _compressed_info(rep, topic, m))
                else:
                    wi, ai = ros1_imu_sample(m)
                    w.append(wi)
                    a.append(ai)
                if len(t) >= limit:
                    break
            s.start_s = t[0] if t else None
            if kind == "imu" and len(t) > 1:
                s.rate_hz = round(1.0 / float(np.median(np.diff(t))), 2)
                _imu_checks(rep, topic, t, w, a)
            elif rep.duration_s:
                s.rate_hz = round(n / rep.duration_s, 2)
            if s.start_s is not None and rep.duration_s:
                s.end_s = s.start_s + rep.duration_s   # bag span; header clock origin
            rep.streams.append(s)


def _csv_stamps(p):
    out = []
    with open(p) as f:
        for row in csv.reader(f):
            if row and not row[0].startswith("#"):
                out.append(row)
    return out


def inspect_euroc(path: str, rep: Report):
    root = os.path.join(path, "mav0") if os.path.isdir(os.path.join(path, "mav0")) else path
    span = []
    for d in sorted(os.listdir(root)):
        csvp = os.path.join(root, d, "data.csv")
        if not os.path.isfile(csvp):
            continue
        rows = _csv_stamps(csvp)
        if not rows:
            continue
        t = np.array([int(r[0]) * 1e-9 for r in rows])
        kind = "imu" if d.startswith("imu") else "camera" if d.startswith("cam") else "other"
        s = Stream(d, kind, len(rows), start_s=float(t[0]), end_s=float(t[-1]))
        if len(t) > 1:
            s.rate_hz = round(1.0 / float(np.median(np.diff(t))), 2)
        if kind == "camera":
            first = os.path.join(root, d, "data", rows[0][1].strip())
            try:
                from PIL import Image
                with Image.open(first) as im:
                    s.info = {"width": im.width, "height": im.height, "mode": im.mode}
            except Exception as e:            # noqa: BLE001 — report, don't crash
                rep.find("camera.first_image", d, "fail", f"cannot read {first}: {e}")
        elif kind == "imu":
            wa = [euroc_imu_values(r) for r in rows[:IMU_SAMPLE]]
            _imu_checks(rep, d, t[:IMU_SAMPLE], [v[0] for v in wa], [v[1] for v in wa])
        span += [s.start_s, s.end_s]
        rep.streams.append(s)
    if span:
        rep.duration_s = max(span) - min(span)


def inspect_images(path: str, rep: Report):
    files = sorted(f for f in os.listdir(path) if f.lower().endswith(IMG_EXT))
    s = Stream(os.path.basename(path.rstrip("/")), "camera", len(files))
    try:
        from PIL import Image
        with Image.open(os.path.join(path, files[0])) as im:
            s.info = {"width": im.width, "height": im.height, "mode": im.mode}
    except Exception as e:                    # noqa: BLE001
        rep.find("camera.first_image", s.name, "fail", f"cannot read first image: {e}")
    rep.streams.append(s)


def _overlap_checks(rep: Report):
    cams = [s for s in rep.streams if s.kind == "camera"]
    imus = [s for s in rep.streams if s.kind == "imu"]
    if not cams:
        rep.find("data.cameras", "dataset", "fail", "no camera stream found")
    if rep.format in ("ros1", "ros2", "euroc") and not imus:
        rep.find("data.imus", "dataset", "warn", "no IMU stream: camera-only calibration only")
    if imus and rep.duration_s is not None and rep.duration_s < MIN_DURATION_S:
        rep.find("data.duration", "dataset", "warn",
                 f"{rep.duration_s:.0f} s recording; cam-IMU needs >= {MIN_DURATION_S:.0f} s of motion")
    for c in cams:
        for i in imus:
            if None in (c.start_s, i.start_s):
                continue
            off = abs(c.start_s - i.start_s)
            if off > 5.0:
                rep.find("time.cam_imu_clock", f"{c.name}~{i.name}", "fail",
                         f"first stamps {off:.1f} s apart: different clocks?")


def inspect(path: str) -> Report:
    fmt = detect_format(path)
    rep = Report(path=os.path.abspath(path), format=fmt, supported_modes=FORMAT_MODES[fmt])
    if fmt in ("ros1", "ros2"):
        inspect_ros1(path, rep)
    elif fmt == "euroc":
        inspect_euroc(path, rep)
    elif fmt == "images":
        inspect_images(path, rep)
    _overlap_checks(rep)
    return rep.finish()


def format_text(rep: Report) -> str:
    dur = f"{rep.duration_s:.1f} s" if rep.duration_s is not None else "?"
    lines = [f"{rep.path}", f"  format {rep.format}  modes {', '.join(rep.supported_modes)}"
             f"  duration {dur}  status {rep.status.upper()}", "  streams:"]
    for s in rep.streams:
        if s.kind == "other":
            continue
        rate = f"{s.rate_hz:.1f} Hz" if s.rate_hz else "?"
        info = " ".join(f"{k}={v}" for k, v in s.info.items())
        lines.append(f"    {s.kind:<6} {s.name:<32} n={s.count:<7} {rate:<10} {info}")
    if rep.findings:
        lines.append("  findings:")
        for f in rep.findings:
            if f.status != "pass" or f.id.startswith("imu.accel"):
                lines.append("    " + format_finding(f))
    return "\n".join(lines)

