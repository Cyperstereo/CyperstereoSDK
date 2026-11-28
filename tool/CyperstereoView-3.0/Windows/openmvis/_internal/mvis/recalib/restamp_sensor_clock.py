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
"""Write a ROS1 bag whose camera header stamps sit on the sensor clock.

Input: a bag whose camera frames are stamped by the driver's publish cadence
(e.g. the ELP stereo under usb_cam: 25 Hz polling of a 29.6 Hz sensor) and the
sensor-clock sidecar `<summary>_sensor_clock.json` written by `mvis cam-imu`
for that bag (`camera_input.cameras[i].retime_sensor_clock: true`). Output: the
same bag, byte-identical except that every message of the listed camera topics
(image + its camera_info, matched by header stamp) carries the lattice capture
time phi + n T_s as header stamp; frames the fit rejected are dropped. Bag record
times are kept, so the message order and all other topics are untouched; the
stream is copied raw (no re-serialisation of images).

A provenance sidecar `<out>.sensor_clock_provenance.yaml` records the source,
the fitted clocks, counts and the sha256 of the source bag when asked for.

    python -m mvis.recalib.restamp_sensor_clock in.bag out.bag \
        --sidecar results/.../cam_imu_sensor_clock.json \
        [--topic 0=/elp/.../left/image_raw,/elp/.../left/camera_info ...] [--sha256]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np

try:
    from rosbags.rosbag1 import Reader, Writer
    from rosbags.typesys import Stores, get_typestore
except ModuleNotFoundError as exc:  # pragma: no cover
    _ROSBAGS_ERROR = exc

    def _missing_rosbags(*_a, **_k):
        raise ModuleNotFoundError("ROS1 bag support requires 'rosbags': "
                                  "pip install 'pymvis[ros1]'") from _ROSBAGS_ERROR

    Reader = Writer = get_typestore = _missing_rosbags
    Stores = None

from mvis.io.sensor_clock import read_sidecar
from mvis.recalib.bagio import header_ns
from mvis.recalib.restamp import set_header_stamp_ns

log = logging.getLogger("mvis.recalib.restamp_sensor_clock")

_MATCH_NS = 500_000        # a map stamp matches a message within 0.5 ms (frames
                           # are >= 33 ms apart; float64 seconds carry ~0.25 µs)


class _StampIndex:
    """Nearest-stamp lookup: message header stamp (ns) -> mapped value, within
    `_MATCH_NS`. Exact integer keys fail on ~10 % of frames because the sidecar
    stores seconds as float64 (rounding flips at the microsecond)."""

    def __init__(self, stamps_ns, values=None):
        o = np.argsort(stamps_ns)
        self.t = np.asarray(stamps_ns, np.int64)[o]
        self.v = None if values is None else [values[i] for i in o]

    def find(self, t_ns: int):
        if len(self.t) == 0:
            return None
        i = int(np.searchsorted(self.t, t_ns))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and abs(int(self.t[j]) - int(t_ns)) <= _MATCH_NS:
                if best is None or abs(int(self.t[j]) - t_ns) < abs(int(self.t[best]) - t_ns):
                    best = j
        return None if best is None else (self.v[best] if self.v is not None else True)

    def __len__(self):
        return len(self.t)


def load_stamp_maps(sidecar_path: str) -> dict:
    """{camera index: {'topic', 'map' {stamp key -> capture ns}, 'dropped' set,
    'clock' {...}}} from a `mvis cam-imu` sensor-clock sidecar."""
    sc = read_sidecar(sidecar_path, check_format=True)
    out = {}
    for c, rec in sc["cameras"].items():
        m = np.asarray(rec["map"], float).reshape(-1, 2)
        out[int(c)] = {
            "topic": rec.get("topic"),
            "map": _StampIndex(np.rint(m[:, 0] * 1e9).astype(np.int64),
                               [int(round(b * 1e9)) for b in m[:, 1]]),
            "dropped": _StampIndex(np.rint(np.asarray(rec.get("dropped_stamps", []), float) * 1e9).astype(np.int64)),
            "clock": {k: rec.get(k) for k in ("period_s", "phase_s", "rate_hz",
                                               "coherence", "scatter_ms")},
        }
    return out


def merge_shared_clock(maps: dict, cams: list[int], *, require_all: bool = True) -> dict:
    """Join the maps of cameras that share one sensor clock (e.g. a stereo module
    split from one side-by-side stream) into ONE lattice: common period (mean) and
    phase (mean), each frame's sensor index taken from whichever camera mapped it.
    Independently fitted phases differ by pass-1 noise (bag 4 whole-bag fit: L−R
    0.67 ms) although the frames are the same hardware frame; the joint lattice
    restores identical stamps on both sides and covers the union of the frames.
    Frames mapped on both sides with different indices, and different frames that
    land on the same sensor index, are dropped (counted).
    By default (`require_all=True`) only frames mapped by EVERY camera of the group
    are kept: a frame one side's own fit rejected is a bad frame, and putting it
    back with the other side's index costs accuracy (bag 2 union: rms 0.80 → 1.09
    px, Dw y −0.13 %; both-sides-only reproduces the mapped solve). `False` takes
    the union.
    Returns a map record like `load_stamp_maps` entries, plus 'merge' stats."""
    cams = [int(c) for c in cams]
    T = float(np.mean([maps[c]["clock"]["period_s"] for c in cams]))
    # a lattice phase is only defined modulo T: express every camera's phase as a
    # sub-period offset from the first camera's, average those, and index all
    # capture times against the common phase
    phi0 = float(maps[cams[0]]["clock"]["phase_s"])
    deltas = [((maps[c]["clock"]["phase_s"] - phi0 + T / 2) % T) - T / 2 for c in cams]
    phi = phi0 + float(np.mean(deltas))
    idx = {}                                  # stamp ns -> set of indices
    seen = {}                                 # stamp ns -> number of cameras mapping it
    for c in cams:
        m = maps[c]["map"]
        for t_ns, cap_ns in zip(m.t, m.v):
            n = int(round((cap_ns * 1e-9 - phi) / T))
            idx.setdefault(int(t_ns), set()).add(n)
            seen[int(t_ns)] = seen.get(int(t_ns), 0) + 1
    one_sided = [t for t, k in seen.items() if k < len(cams)]
    if require_all:
        for t in one_sided:
            idx.pop(t)
    stamps, caps, conflict = [], [], []
    owner = {}                                # sensor index -> stamps claiming it
    for t_ns, ns in idx.items():
        if len(ns) == 1:
            owner.setdefault(next(iter(ns)), []).append(t_ns)
        else:
            conflict.append(t_ns)
    # two DIFFERENT published frames on one sensor index: each side mapped only one
    # of them, so the per-frame check above cannot see it (bag 7: one pair, 40 ms
    # apart). One index is wrong and nothing says which -> drop all claimants.
    n_collision = 0
    for n, ts in sorted(owner.items()):
        if len(ts) == 1:
            stamps.append(ts[0]); caps.append(int(round((phi + n * T) * 1e9)))
        else:
            conflict.extend(ts); n_collision += len(ts)
    o = np.argsort(stamps)
    stamps = [stamps[i] for i in o]; caps = [caps[i] for i in o]
    dropped = set()
    for c in cams:
        dropped.update(int(t) for t in maps[c]["dropped"].t)
    dropped.update(conflict)
    if require_all:
        dropped.update(one_sided)
    dropped -= set(stamps)
    return {"topic": None,
            "map": _StampIndex(np.asarray(stamps, np.int64), caps),
            "dropped": _StampIndex(np.asarray(sorted(dropped), np.int64)),
            "clock": {"period_s": T, "phase_s": phi, "rate_hz": 1.0 / T,
                      "coherence": float(np.mean([maps[c]["clock"]["coherence"] for c in cams])),
                      "scatter_ms": float(np.mean([maps[c]["clock"]["scatter_ms"] for c in cams]))},
            "merge": {"cameras": cams, "n_frames": len(stamps), "n_conflict": len(conflict), "n_index_collision": n_collision,
                      "n_one_sided": len(one_sided), "require_all": bool(require_all),
                      "phase_spread_ms": float((max(deltas) - min(deltas)) * 1e3)}}


def companion_topics(topic: str) -> list[str]:
    """Topics stamped with the same frame as an image topic: its camera_info."""
    base = topic.rsplit("/", 1)[0]
    return [topic, base + "/camera_info"]


def restamp_bag(in_bag: str, out_bag: str, maps: dict, topics_by_cam: dict,
                *, sha256: bool = False, keep_unmatched: bool = False) -> dict:
    """Copy `in_bag` to `out_bag` re-stamping the topics in `topics_by_cam`
    ({camera index: [topics]}) from `maps` (see `load_stamp_maps`). Frames of
    those topics that the map does not cover (no board detection / rejected by
    the fit) have NO trustworthy capture time and are dropped — a frame left
    with its driver stamp is off by up to two sensor periods and poisons any
    solve on the corrected bag (bag 4: Dw −0.6 %, rms 0.65 → 0.87 px with 17
    such frames). `keep_unmatched=True` keeps them as recorded instead.
    Returns the provenance dict (also written next to `out_bag`)."""
    if os.path.exists(out_bag):
        raise FileExistsError(f"{out_bag} exists — remove it first (never overwrites a bag)")
    ts = get_typestore(Stores.ROS1_NOETIC)
    topic_cam = {t: c for c, tl in topics_by_cam.items() for t in tl}
    for c in topic_cam.values():
        if c not in maps:
            raise ValueError(f"no stamp map for camera {c}")
    stats = {t: {"restamped": 0, "dropped": 0, "unmatched": 0} for t in topic_cam}
    n_pass = 0
    t0 = time.time()
    with Reader(in_bag) as reader, Writer(out_bag) as writer:
        conns = {}
        for conn in reader.connections:
            ext = getattr(conn, "ext", None)
            kw = dict(latching=getattr(ext, "latching", None),
                      callerid=getattr(ext, "callerid", None))
            md5 = getattr(conn, "digest", None)
            if conn.msgtype not in ts.types and md5:
                conns[conn.id] = writer.add_connection(conn.topic, conn.msgtype, msgdef="",
                                                       md5sum=md5, **kw)
            else:
                conns[conn.id] = writer.add_connection(conn.topic, conn.msgtype,
                                                       typestore=ts, **kw)
        for conn, t_ns, raw in reader.messages():
            c = topic_cam.get(conn.topic)
            if c is None:
                writer.write(conns[conn.id], t_ns, raw); n_pass += 1
                continue
            msg = ts.deserialize_ros1(raw, conn.msgtype)
            h_ns = header_ns(msg)
            m = maps[c]
            t_new = m["map"].find(h_ns)
            if t_new is not None:
                set_header_stamp_ns(msg, t_new)
                writer.write(conns[conn.id], t_ns, ts.serialize_ros1(msg, conn.msgtype))
                stats[conn.topic]["restamped"] += 1
            elif m["dropped"].find(h_ns):
                stats[conn.topic]["dropped"] += 1          # fit outlier: drop the frame
            elif keep_unmatched:
                writer.write(conns[conn.id], t_ns, raw)    # as recorded (driver stamp)
                stats[conn.topic]["unmatched"] += 1
            else:
                stats[conn.topic]["unmatched"] += 1        # no capture time known: drop
    prov = {
        "tool": "mvis.recalib.restamp_sensor_clock", "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_bag": os.path.abspath(in_bag), "output_bag": os.path.abspath(out_bag),
        "source_size": os.path.getsize(in_bag), "output_size": os.path.getsize(out_bag),
        "cameras": {int(c): {"topics": topics_by_cam[c], "clock": maps[c]["clock"],
                             "n_map": len(maps[c]["map"]), "n_dropped_in_map": len(maps[c]["dropped"]),
                             **({"shared_clock": maps[c]["merge"]} if "merge" in maps[c] else {})}
                    for c in topics_by_cam},
        "topics": stats, "passthrough_messages": n_pass,
        "unmatched_frames": "kept as recorded" if keep_unmatched else "dropped",
        "seconds": round(time.time() - t0, 1),
        "note": ("header.stamp of the listed topics moved onto the sensor lattice "
                 "phi + n*T_s; bag record times unchanged; 'dropped' = fit outliers, "
                 "'unmatched' = frames without a recovered capture time (no board "
                 "detection / outside the solved window), "
                 + ("kept with their driver stamp" if keep_unmatched else "removed")),
    }
    if sha256:
        prov["source_sha256"] = _sha256(in_bag)
    prov_path = out_bag + ".sensor_clock_provenance.yaml"
    try:
        import yaml
        with open(prov_path, "w") as f:
            yaml.safe_dump(prov, f, sort_keys=False)
    except Exception:                                      # pragma: no cover
        with open(prov_path.replace(".yaml", ".json"), "w") as f:
            json.dump(prov, f, indent=1)
    prov["provenance"] = prov_path
    log.info("restamp: %s -> %s in %.0f s; %s", in_bag, out_bag, prov["seconds"], stats)
    return prov


def _sha256(path: str) -> str:
    from mvis.io.fileutil import compute_sha256
    return compute_sha256(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("in_bag"); ap.add_argument("out_bag")
    ap.add_argument("--sidecar", required=True, help="<summary>_sensor_clock.json from mvis cam-imu")
    ap.add_argument("--topic", action="append", default=[],
                    help="CAM=/topic[,/topic2] — topics re-stamped from camera CAM's map; "
                         "default: the sidecar's image topic + its camera_info")
    ap.add_argument("--sha256", action="store_true", help="record the source bag's sha256")
    ap.add_argument("--keep-unmatched", action="store_true",
                    help="keep frames the map does not cover with their recorded stamp "
                         "(default: drop them — their capture time is unknown)")
    ap.add_argument("--shared-clock", default=None,
                    help="comma-separated camera indices that share one sensor clock (a stereo "
                         "module): their maps are merged into one lattice and applied to all "
                         "their topics, so both sides get identical stamps")
    ap.add_argument("--shared-clock-union", action="store_true",
                    help="with --shared-clock: keep frames mapped by ANY camera of the group "
                         "(default: only frames every camera mapped)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    maps = load_stamp_maps(a.sidecar)
    topics = {}
    for spec in a.topic:
        c, tl = spec.split("=", 1)
        topics[int(c)] = [t for t in tl.split(",") if t]
    if not topics:
        for c, m in maps.items():
            if not m["topic"]:
                raise SystemExit(f"sidecar has no topic for camera {c}; pass --topic {c}=...")
            topics[c] = companion_topics(m["topic"])
    if a.shared_clock:
        group = [int(x) for x in a.shared_clock.split(",")]
        merged = merge_shared_clock(maps, group, require_all=not a.shared_clock_union)
        log.info("shared clock %s: %d frames, %d index conflicts, phase spread %.3f ms",
                 group, merged["merge"]["n_frames"], merged["merge"]["n_conflict"],
                 merged["merge"]["phase_spread_ms"])
        key = group[0]
        all_topics = [t for c in group for t in topics.pop(c, [])]
        maps = {c: m for c, m in maps.items() if c not in group}
        maps[key] = merged
        topics[key] = all_topics
    prov = restamp_bag(a.in_bag, a.out_bag, maps, topics, sha256=a.sha256,
                       keep_unmatched=a.keep_unmatched)
    print(json.dumps({k: prov[k] for k in ("topics", "unmatched_frames", "passthrough_messages",
                                           "seconds", "provenance")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
