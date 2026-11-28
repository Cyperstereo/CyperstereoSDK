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
"""Extract a GiLabs Ego calibration episode (recording.mcap) into the
looperrobotics-style layout so the existing prep + calibration pipeline runs
unchanged:

    <out>/imu_cam/cam0/<ns>.png     # decoded H.265 frames, ns = sensor timestamp
    <out>/imu_cam/cam1/<ns>.png
    <out>/imu_cam/imu.csv           # timestamp,omega_x..,alpha_z (EuRoC column order)

then:  build_mav0.py --root <out>  ->  <out>/mav0/{cam0,cam1,imu0}

MCAP contents (visio_schema / foxglove protobuf):
  /ego/camera/{0,1}  foxglove.CompressedVideo  (H.265 Annex-B, ~29.4 Hz)
  /ego/imu/0/raw     visio_schema ImuRaw       (batched ~1 Hz x ~470 samples,
                                                first_sample_time + t_offset_ns)

Frame <-> timestamp mapping: each CompressedVideo message is one access unit.
The stream may start mid-GOP (no leading VPS/SPS/PPS/IDR); ffmpeg silently
drops undecodable leading AUs, which would misalign the mapping. We therefore
find the first message containing an IRAP/parameter-set NAL, feed the stream
from there, and assert decoded-frame count == fed-message count. B-frames
would also break the 1:1 order assumption -> assert monotonic timestamps and
equal counts (calibration streams are expected to be all-intra or IPPP).

Usage:
    python extract_mcap.py --mcap .../recording.mcap --out .../extracted/<ep>
"""
import argparse
import csv

import cv2
import sys
from pathlib import Path


def nal_types(au: bytes):
    """Yield H.265 nal_unit_type for each NAL in an Annex-B access unit."""
    i = 0
    n = len(au)
    while i < n - 4:
        if au[i:i+3] == b"\x00\x00\x01":
            yield (au[i+3] >> 1) & 0x3F
            i += 3
        elif au[i:i+4] == b"\x00\x00\x00\x01":
            yield (au[i+4] >> 1) & 0x3F
            i += 4
        else:
            i += 1


def has_entrypoint(au: bytes) -> bool:
    """True if the AU carries a VPS/SPS/PPS or IRAP picture (decodable start)."""
    for t in nal_types(au):
        if t in (32, 33, 34) or 16 <= t <= 23:  # VPS/SPS/PPS or BLA/IDR/CRA
            return True
    return False


def extract_camera(msgs, cam_dir: Path) -> int:
    """msgs: list of (timestamp_ns, bytes). Decode to <ns>.png; return count.

    Packet-by-packet PyAV decode: each MCAP message is one access unit, fed as
    its own packet with pts = the message timestamp; decoded frames carry that
    pts back out, so the frame<->timestamp mapping is exact regardless of what
    the decoder drops. The streams contain mid-stream reference breaks (capture
    drops -> "Could not find ref with POC n"): those pictures are undecodable
    and are correctly skipped (ffmpeg 6 used to emit them as garbage frames;
    ffmpeg >= 7 drops them, which broke the old concatenated-stream 1:1 map).
    """
    import av
    ts = [t for t, _ in msgs]
    assert all(b > a for a, b in zip(ts, ts[1:])), "non-monotonic frame timestamps"

    cam_dir.mkdir(parents=True, exist_ok=True)
    codec = av.CodecContext.create("hevc", "r")
    n = skipped = 0

    # Dirty-GOP tracking: a decode failure OR a recording gap (recorder never
    # wrote some frames — the decoder then "succeeds" via error concealment,
    # propagating garbage until the next keyframe) marks the stream dirty; only
    # frames from clean packets are kept. RASL leading pictures are always
    # dropped (undecodable after their CRA when history is broken).
    dts = sorted(b - a for a, b in zip(ts, ts[1:]))
    dt_med = dts[len(dts) // 2] if dts else 0
    clean = {}
    dirty = True   # until the first IRAP
    prev_t = None

    def emit(frames):
        nonlocal n, skipped
        for fr in frames:
            assert fr.pts is not None, "decoder returned frame without pts"
            if not clean.get(fr.pts):
                skipped += 1
                continue
            cv2.imwrite(str(cam_dir / f"{fr.pts}.png"),
                        fr.to_ndarray(format="gray"))
            n += 1

    for t, d in msgs:
        if prev_t is not None and dt_med and (t - prev_t) > 1.5 * dt_med:
            dirty = True                     # recording gap -> concealment ahead
        prev_t = t
        nals = set(nal_types(d))
        if any(16 <= x <= 23 for x in nals):
            dirty = False                    # IRAP: clean random-access point
        rasl = bool(nals & {8, 9})
        clean[t] = not dirty and not rasl
        pkt = av.Packet(d)
        pkt.pts = t
        try:
            emit(codec.decode(pkt))
        except av.error.InvalidDataError:
            dirty = True
            continue
    try:
        emit(codec.decode(None))   # flush
    except (av.error.InvalidDataError, av.error.EOFError):
        pass
    if n < len(msgs):
        print(f"  {cam_dir.name}: kept {n}/{len(msgs)} "
              f"(dropped {len(msgs) - n}: undecodable or dirty-GOP)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcap", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    try:  # optional extra: the module itself must import without it
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory
    except ImportError as e:
        raise SystemExit(f"{e}. Reading MCAP needs the mcap extra: pip install pymvis[mcap]")

    out = Path(args.out) / "imu_cam"
    cams = {}   # topic -> list[(ns, bytes)]
    imu = []    # (ns, wx, wy, wz, ax, ay, az)

    with open(args.mcap, "rb") as fh:
        r = make_reader(fh, decoder_factories=[DecoderFactory()])
        for schema, channel, msg, p in r.iter_decoded_messages():
            if schema.name == "foxglove.CompressedVideo":
                cams.setdefault(channel.topic, []).append(
                    (p.timestamp.ToNanoseconds(), bytes(p.data)))
            elif schema.name.endswith("ImuRaw"):
                t0 = p.first_sample_time.ToNanoseconds()
                for s in p.samples:
                    imu.append((t0 + s.t_offset_ns,
                                s.angular_velocity.x, s.angular_velocity.y,
                                s.angular_velocity.z,
                                s.linear_acceleration.x, s.linear_acceleration.y,
                                s.linear_acceleration.z))

    for topic in sorted(cams):
        cam_idx = topic.rstrip("/").split("/")[-2] if topic.endswith("raw") \
            else topic.split("/")[-1]
        n = extract_camera(cams[topic], out / f"cam{cam_idx}")
        print(f"  cam{cam_idx}: {n} frames")

    if not imu:
        print("ERROR: no IMU messages in this recording (cameras only)")
        sys.exit(3)
    imu.sort()
    assert all(b[0] > a[0] for a, b in zip(imu, imu[1:])), "duplicate IMU stamps"
    with open(out / "imu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "omega_x", "omega_y", "omega_z",
                    "alpha_x", "alpha_y", "alpha_z"])
        w.writerows(imu)
    print(f"  imu.csv: {len(imu)} samples "
          f"({(imu[-1][0]-imu[0][0])/1e9:.1f} s, "
          f"{len(imu)/((imu[-1][0]-imu[0][0])/1e9):.1f} Hz)")


if __name__ == "__main__":
    main()
