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
"""Convert a looperrobotics dataset into an EuRoC ``mav0/`` tree so the
existing MVIS EuRoC loaders read it unchanged — no MVIS code changes.

RIG-AGNOSTIC: camera dirs are auto-discovered (``cam*`` under ``imu_cam/``, any
count) and any image extension is accepted (png/jpg/jpeg, filename stem = ns
stamp). Works for SA16(96s record) (2 cams, PNG) and SA16(7.3) (3 cams, cam2 =
4.29 Hz rolling-shutter JPG) alike; future rigs need no script changes.

PER-CAMERA ROTATION (``--rotate camN=cw|ccw|180``, repeatable): materializes
rotated grayscale PNGs instead of symlinks. Needed when a rolling-shutter
camera's files are stored rotated relative to the sensor: the RS model sweeps
image ROWS, so a 90°-rotated file makes the readout run along stored columns —
tr collapses toward 0, becomes intrinsics-model-dependent, and reproj inflates.
(SA16(7.3) cam2: portrait 1088x1920 files from a landscape sensor; --rotate
cam2=cw recovers tr 33.3 ms = 30.6 us/line and drops reproj 3.1 -> 2.0 px.)

The same ``mav0/`` tree feeds BOTH stages:
  * camera-only  -> ``EurocImageSource`` (io/euroc_image_source.py)
  * cam-IMU      -> ``load_euroc``       (io/cam_imu_loader.py)

Source (``imu_cam/`` = the cam-IMU recording):

    <root>/imu_cam/camN/<ns>.<ext>   # N auto-discovered; ext in png/jpg/jpeg
    <root>/imu_cam/imu.csv           # timestamp,omega_xyz,alpha_xyz (ns, rad/s, m/s^2)

Output:

    <out>/camN/data.csv        # "#timestamp [ns],filename"  + data/<file> (symlinks)
    <out>/imu0/data.csv        # looperrobotics imu.csv, header commented with '#'

    <out>/camN_camonly/        # optional decimated view for camera-only (--camonly-hz);
                               #   data/ symlinks the full-rate tree; data.csv is thinned.
                               #   SKIPPED for cameras already at/below the target rate
                               #   (their full-rate dir doubles as the camera-only input).

The looperrobotics imu.csv column order already matches EuRoC exactly
(timestamp, wx,wy,wz, ax,ay,az), so imu0/data.csv is the same rows with the
header line prefixed by '#' (the loader skips '#' rows but would choke on a
bare "timestamp,..." header).

The camera-only stage doesn't need a full 20 Hz — with no time-offset
estimation, ~4 Hz is plenty for intrinsics + extrinsics. The EuRoC image source
(`EurocImageSource`) ignores `factor_hz`, so we decimate at the data layer
instead: a `camN_camonly/` view whose `data/` symlinks the full-rate
`camN/data/` and whose `data.csv` lists only every k-th frame. cam-IMU keeps
the full-rate `camN/` dirs.

Usage:
    python -m mvis.datasets.converters.looper_mav0 --root <dataset>   # full + 4 Hz camonly
    python -m mvis.datasets.converters.looper_mav0 --root <dataset> --camonly-hz 0
    python -m mvis.datasets.converters.looper_mav0 --root <dataset> --out <mav0>
"""
import argparse
import csv
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def discover_cams(src: Path) -> list[Path]:
    """All ``cam*`` dirs under the recording, in name order (cam0, cam1, ...)."""
    cams = sorted(d for d in src.iterdir() if d.is_dir() and d.name.startswith("cam"))
    if not cams:
        raise SystemExit(f"no cam*/ dirs found under {src}")
    return cams


def _frames(src_dir: Path) -> list[Path]:
    """Image files (any supported extension) sorted by their ns-stamp stem."""
    imgs = [p for p in src_dir.iterdir()
            if p.suffix.lower() in IMG_EXTS and p.stem.isdigit()]
    if not imgs:
        raise SystemExit(f"no {'/'.join(sorted(IMG_EXTS))} frames in {src_dir}")
    return sorted(imgs, key=lambda p: int(p.stem))


def build_cam(src_dir: Path, out_dir: Path, rotate: str | None = None) -> int:
    data = out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    frames = _frames(src_dir)
    if rotate:
        import cv2
        rot = {"cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
               "180": cv2.ROTATE_180}[rotate]
    with open(out_dir / "data.csv", "w") as f:
        f.write("#timestamp [ns],filename\n")
        for p in frames:
            if rotate:                     # materialize rotated grayscale PNGs
                name = p.stem + ".png"
                out = data / name
                if out.is_symlink():       # replace a previous symlinked build
                    out.unlink()
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                cv2.imwrite(str(out), cv2.rotate(img, rot))
            else:
                name = p.name
                link = data / name
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(p.resolve())
            f.write(f"{p.stem},{name}\n")
    return len(frames)


def _relink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target)


def source_hz(full_dir: Path) -> tuple[list[tuple[int, str]], float]:
    """(sorted data.csv rows, mean frame rate) of an already-built cam dir."""
    rows = []
    with open(full_dir / "data.csv") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ts_s, name = line.split(",")
                rows.append((int(ts_s), name))
    rows.sort()
    dur = (rows[-1][0] - rows[0][0]) * 1e-9
    return rows, ((len(rows) - 1) / dur if dur > 0 else 0.0)


def build_cam_decimated(full_dir: Path, out_dir: Path, cam_hz: float) -> int:
    """Camera-only view of an already-built full-rate cam dir, thinned to ~cam_hz.

    `data/` symlinks the full-rate `data/` (no image copies); `data.csv` lists
    every k-th frame, where k = round(source_hz / cam_hz)."""
    rows, src_hz = source_hz(full_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _relink(out_dir / "data", (full_dir / "data").resolve())
    step = max(1, round(src_hz / cam_hz)) if cam_hz > 0 else 1
    kept = rows[::step]
    with open(out_dir / "data.csv", "w") as f:
        f.write("#timestamp [ns],filename\n")
        for ts, name in kept:
            f.write(f"{ts},{name}\n")
    return len(kept)


def build_imu(src_csv: Path, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(src_csv) as f:
        rows = list(csv.reader(f))
    with open(out_dir / "data.csv", "w") as f:
        f.write("#timestamp [ns],w_x [rad s^-1],w_y,w_z,a_x [m s^-2],a_y,a_z\n")
        for r in rows[1:]:            # rows[0] is the looperrobotics header
            if r:
                f.write(",".join(r) + "\n")
    return len(rows) - 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="looperrobotics recording dir (or extracted MCAP episode)")
    ap.add_argument("--out", default=None,
                    help="mav0 output dir (default <root>/mav0)")
    ap.add_argument("--camonly-hz", type=float, default=4.0,
                    help="frame rate for the decimated camera-only view "
                         "(0 = skip; full-rate cam-IMU tree is always built). "
                         "Cameras already at/below this rate get no _camonly "
                         "view — use their full-rate dir for camera-only too.")
    ap.add_argument("--rotate", action="append", default=[],
                    metavar="CAM=cw|ccw|180",
                    help="rotate a camera's images (repeatable, e.g. --rotate "
                         "cam2=cw); needed when files are stored rotated vs the "
                         "sensor so the RS readout matches image rows")
    a = ap.parse_args()
    rotations = {}
    for spec in a.rotate:
        cam, _, how = spec.partition("=")
        if how not in ("cw", "ccw", "180"):
            raise SystemExit(f"--rotate {spec}: want CAM=cw|ccw|180")
        rotations[cam] = how
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "mav0"
    src = root / "imu_cam"
    if not src.is_dir():        # flat recordings (e.g. looperrobotics/1..10):
        src = root             # cam*/ + imu.csv live directly under <root>
    for cam_dir in discover_cams(src):
        cam = cam_dir.name
        rot = rotations.pop(cam, None)
        n = build_cam(cam_dir, out / cam, rotate=rot)
        _, hz = source_hz(out / cam)
        note = f", rotated {rot}" if rot else ""
        print(f"{cam}: {n} frames ({hz:.2f} Hz{note}) -> {out / cam}")
        if a.camonly_hz > 0:
            if round(hz / a.camonly_hz) <= 1:
                print(f"{cam}_camonly: skipped ({hz:.2f} Hz already <= "
                      f"~{a.camonly_hz:g} Hz target; use {out / cam})")
                continue
            m = build_cam_decimated(out / cam, out / f"{cam}_camonly", a.camonly_hz)
            print(f"{cam}_camonly: {m} frames (~{a.camonly_hz:g} Hz) -> {out / f'{cam}_camonly'}")
    if rotations:
        raise SystemExit(f"--rotate names unknown cameras: {sorted(rotations)}")
    n = build_imu(src / "imu.csv", out / "imu0")
    print(f"imu0: {n} samples -> {out / 'imu0'}")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
