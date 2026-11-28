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
"""`mvis check` — environment and run-config lint before a solve.

The solver parses configs leniently (unknown keys are only logged). This
checker is strict: every finding has a stable ``id`` and a status
``pass | warn | fail``, and a caller should not start a solve until there are no
failures.
"""
from __future__ import annotations

import importlib
import importlib.metadata as _md
import logging
import os
import shutil
import subprocess
import sys
import time

from mvis.core.config_parser import load_yaml
from mvis.core.gravity import DEFAULT_HINT
from mvis.doctor.findings import Finding

TAG_FAMILIES = {"t36h11", "t36h11b1"}
MODES = {"cam", "cam-imu"}


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.msgs = []

    def emit(self, record):
        self.msgs.append(record.getMessage())


# optional extra -> (import name, pip extra) ; `needs` keys come from config_needs()
EXTRAS = {"apriltag": ("aprilgrid", "apriltag"), "ros1": ("rosbags", "ros1")}
_CV2_DISTS = ("opencv-python", "opencv-contrib-python",
              "opencv-python-headless", "opencv-contrib-python-headless")


def config_needs(path: str) -> set:
    """Optional components a run config needs: apriltag, aruco, charuco, ros1."""
    raw = load_yaml(path)
    needs = set()
    bt = (raw.get("board") or {}).get("board_type", "chessboard")
    if bt.startswith("apriltag"):
        needs.add("apriltag")
    if bt.startswith("aruco"):
        needs.add("aruco")
    if bt == "charuco":
        needs |= {"aruco", "charuco"}
    inp = raw.get("input") or {}
    cams = inp.get("cameras") or raw.get("cameras") or []
    if inp.get("format") in ("ros1", "ros2") or any(
            c.get("source_type") in ("rosbag_topic", "ros1bag", "ros2bag") for c in cams):
        needs.add("ros1")
    return needs


def check_env(needs=None, smoke: bool = False) -> list:
    """Installation verification (Stage 0). `needs` (from config_needs) turns a
    missing optional extra from a warning into a failure; `smoke` runs a short
    synthetic camera solve."""
    needs = set(needs or ())
    out = []
    out.append(Finding("env.python", "python", "pass", f"{sys.version.split()[0]} at {sys.executable}"))
    import numpy as np
    major = int(np.__version__.split(".")[0])
    out.append(Finding("env.numpy", "numpy", "pass" if major < 2 else "fail",
                       f"numpy {np.__version__}" + ("" if major < 2 else
                       " — GTSAM 4.2 wheels segfault on numpy>=2; pin numpy<2")))
    try:
        import gtsam  # noqa: F401
        gv = _version("gtsam")
        tested = gv.startswith("4.2")
        out.append(Finding("env.gtsam", "gtsam", "pass" if tested else "warn",
                           f"gtsam {gv or '?'}" + ("" if tested else
                           " — MVIS is tested with gtsam 4.2.x: pip install 'gtsam>=4.2,<4.3'")))
    except Exception as e:                      # noqa: BLE001
        out.append(Finding("env.gtsam", "gtsam", "fail", f"gtsam import failed: {e}"))
    out += _check_cv2("aruco" in needs, "charuco" in needs)
    for key, (mod, extra) in EXTRAS.items():
        try:
            importlib.import_module(mod)
            out.append(Finding(f"env.extra.{key}", mod, "pass", f"{mod} {_version(mod)}"))
        except ImportError:
            what = {"apriltag": "AprilTag boards", "ros1": "ROS1 and ROS2 bags"}[key]
            out.append(Finding(f"env.extra.{key}", mod, "fail" if key in needs else "warn",
                               f"{mod} not installed (needed for {what}): "
                               f"pip install 'pymvis[{extra}]'"))
    import mvis
    out += _check_install(mvis.__file__, mvis.__version__, os.getcwd())
    out.append(_check_cli())
    if smoke:
        out.append(smoke_solve())
    return out


def _check_install(mvis_file: str, version: str, cwd: str) -> list:
    """Where `mvis` is imported from. A source checkout (`pip install -e .` or
    PYTHONPATH) has `pyproject.toml` two levels above the package: then cwd must
    lie in that checkout, or the user is running another worktree's code, and
    the git sha says which commit. An installed wheel (site-packages) has no
    checkout to compare with, so it just reports the package version."""
    src = os.path.dirname(os.path.dirname(os.path.abspath(mvis_file)))
    repo = os.path.dirname(src)
    if not os.path.isfile(os.path.join(repo, "pyproject.toml")):
        return [Finding("env.mvis_checkout", "mvis", "pass",
                        f"installed package pymvis {_version('pymvis') or version} from {src}")]
    out = []
    cwd = os.path.abspath(cwd)
    same = cwd == repo or cwd.startswith(repo + os.sep)
    out.append(Finding("env.mvis_checkout", "mvis", "pass" if same else "warn",
                       f"mvis imported from {repo}" + ("" if same else
                       f" but cwd is {cwd}: another checkout/worktree? set PYTHONPATH=<repo>/src")))
    try:
        sha = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        out.append(Finding("env.git_sha", "mvis", "pass", sha))
    except Exception:                           # noqa: BLE001
        out.append(Finding("env.git_sha", "mvis", "warn", "not a git checkout"))
    return out


def _version(mod: str) -> str:
    try:
        return _md.version(mod)
    except _md.PackageNotFoundError:
        return ""


def _check_cv2(need_aruco: bool, need_charuco: bool = False) -> list:
    try:
        import cv2
    except Exception as e:                      # noqa: BLE001
        return [Finding("env.cv2", "opencv", "fail", f"cv2 import failed: {e}")]
    dists = {}
    for d in _CV2_DISTS:
        try:
            dists[d] = _md.version(d)
        except _md.PackageNotFoundError:
            pass
    msg = f"cv2 {cv2.__version__} from " + (", ".join(f"{d} {v}" for d, v in dists.items()) or "?")
    status = "pass"
    if len(set(dists.values())) > 1:
        status = "warn"
        msg += " — several OpenCV wheels with different versions share `cv2`; " \
               "keep one (pip uninstall the others, then reinstall it)"
    out = [Finding("env.cv2", "opencv", status, msg)]
    has_aruco = hasattr(cv2, "aruco")
    if need_aruco or not has_aruco:
        out.append(Finding("env.cv2_aruco", "opencv", "pass" if has_aruco else
                           ("fail" if need_aruco else "warn"),
                           "cv2.aruco available" if has_aruco else
                           "cv2.aruco missing (needed for ArUco boards): pip install 'pymvis[aruco]'"))
    if need_charuco and has_aruco:
        ok = hasattr(cv2.aruco, "CharucoDetector")
        out.append(Finding("env.cv2_charuco", "opencv", "pass" if ok else "fail",
                           "cv2.aruco.CharucoDetector available" if ok else
                           f"ChArUco boards need OpenCV >= 4.7 (found {cv2.__version__})"))
    return out


def _check_cli() -> Finding:
    exe = shutil.which("mvis")
    if not exe:
        return Finding("env.cli", "mvis", "warn",
                       "`mvis` not on PATH — activate the env, or use `python -m mvis.cli.main`")
    try:
        with open(exe, "rb") as f:
            first = f.readline().decode(errors="replace").strip()
    except OSError:
        first = ""
    interp = first[2:].split()[0] if first.startswith("#!") else ""
    if interp and os.path.realpath(os.path.dirname(interp)) != \
            os.path.realpath(os.path.dirname(sys.executable)):
        return Finding("env.cli", "mvis", "warn",
                       f"`mvis` on PATH ({exe}) runs {interp}, not this env's "
                       f"{sys.executable} — activate the right env")
    return Finding("env.cli", "mvis", "pass", exe)


def smoke_solve(seed: int = 0) -> Finding:
    """Recover known pinhole-radtan intrinsics from a short synthetic chessboard
    sequence with the same init + solve steps `mvis run` (mode cam) uses for one
    camera. Proves OpenCV, GTSAM and the MVIS solver run end to end."""
    t0 = time.time()
    try:
        import cv2
        import numpy as np
        from mvis.calibration.initializer import Initializer
        from mvis.calibration.optimizer import Optimizer, OptimizerOptions
        from mvis.core.types import FrameObservations, Observation
        from mvis.detection.chessboard import Chessboard

        board = Chessboard(rows=7, cols=5, square_size=0.03)
        K = np.array([[500.0, 0, 320], [0, 500, 240], [0, 0, 1]])
        dist = np.array([-0.1, 0.01, 0.001, -0.001, 0.0])
        size = (640, 480)
        pts3 = board.all_points_3d().astype(np.float32)
        rng = np.random.RandomState(seed)
        frames = []
        while len(frames) < 30:
            rvec = rng.uniform(-0.6, 0.6, 3)
            tvec = np.array([rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.5, 0.9)])
            uv = cv2.projectPoints(pts3, rvec, tvec, K, dist)[0].reshape(-1, 2)
            if not np.all((uv > 20) & (uv < np.array(size) - 20)):
                continue
            uv = uv + rng.randn(*uv.shape) * 0.3
            frames.append(FrameObservations(
                camera_id=0, timestamp=0.1 * len(frames), image_size=size,
                observations=[Observation(point_id=i, pixel=uv[i], row=int(uv[i, 1]))
                              for i in range(len(uv))]))
        log = logging.getLogger("mvis")
        level, log.level = log.level, logging.WARNING
        try:
            init = Initializer(board)
            K0, d0 = init.init_intrinsics_opencv(frames, size, "radtan")
            poses = init.init_board_poses(frames, K0, d0, "radtan")
            res = Optimizer(board, OptimizerOptions(max_iterations=30, verbose=False)) \
                .solve_single_camera(frames, K0, d0, poses, "radtan", size)
        finally:
            log.setLevel(level)
        fx, fy = res.cameras[0].intrinsics[:2]
        err = max(abs(fx - 500) / 500, abs(fy - 500) / 500)
        # a broken install is off by tens of percent or crashes; 3 % / 0.6 px
        # (noise 0.3 px per axis -> 0.42 px expected) leaves room for the
        # focal/distortion spread of a 30-frame sequence
        ok = err < 0.03 and res.overall_rms < 0.6
        return Finding("env.smoke", "solver", "pass" if ok else "fail",
                       f"synthetic camera solve: fx {fx:.1f} fy {fy:.1f} (truth 500), "
                       f"RMS {res.overall_rms:.2f} px, {time.time() - t0:.1f} s"
                       + ("" if ok else " — solver does not recover known intrinsics"))
    except Exception as e:                      # noqa: BLE001
        return Finding("env.smoke", "solver", "fail",
                       f"synthetic camera solve crashed: {type(e).__name__}: {e}")


def _check_charuco(bc) -> list:
    """ChArUco board block: a known dictionary with room for the board's markers, sizes
    in metres, marker inside the square. rows/cols count squares (not inner corners)."""
    from mvis.detection.charuco_board import dictionary_problem
    out = []
    name = bc.aruco_dict
    problem = dictionary_problem(name, bc.rows, bc.cols)
    if problem:
        out.append(Finding("board.aruco_dict", "board", "fail", problem))
    if not 0.002 < bc.square_size < 1.0:
        out.append(Finding("board.square_size", "board", "fail",
                           f"square_size {bc.square_size} must be in metres"))
    if not 0 < bc.marker_size < bc.square_size:
        out.append(Finding("board.marker_size", "board", "fail",
                           f"marker_size {bc.marker_size} must be in (0, square_size)"))
    out.append(Finding("board.charuco_layout", "board", "pass",
                       f"ChArUco {bc.rows} x {bc.cols} squares = "
                       f"{(bc.rows - 1) * (bc.cols - 1)} inner corners, {name}"))
    return out


def _exists(out, fid, subject, path, what):
    if path and not os.path.exists(path):
        out.append(Finding(fid, subject, "fail", f"{what} not found: {path}"))


def check_config(path: str) -> list:
    out = []
    raw = load_yaml(path)
    mode = raw.get("mode")
    if mode is None:
        out.append(Finding("config.mode", "mode", "fail",
                           "no `mode:` key — `mvis run` silently falls back to camera-only"))
        mode = "cam-imu" if "imu_topics" in (raw.get("input") or {}) else "cam"
    elif mode not in MODES:
        out.append(Finding("config.mode", "mode", "fail", f"unknown mode {mode!r}"))
        return out

    h = _Collect()
    log = logging.getLogger("mvis.core.config_parser")
    log.addHandler(h)
    try:
        from mvis.core.config import CamImuRunConfig, CamRunConfig
        cfg = (CamImuRunConfig if mode == "cam-imu" else CamRunConfig).from_yaml(path)
    except Exception as e:                      # noqa: BLE001
        out.append(Finding("config.parse", path, "fail", f"config does not parse: {e}"))
        return out
    finally:
        log.removeHandler(h)
    for m in h.msgs:
        if "unknown key" in m:
            out.append(Finding("config.unknown_key", "config", "fail", m))

    b = raw.get("board") or {}
    if b.get("board_type", "chessboard") == "apriltag":
        fam = b.get("tag_family")
        if fam is None:
            out.append(Finding("board.tag_family", "board", "warn",
                               "tag_family not set; default t36h11 (Kalibr 2-bit border)"))
        elif fam not in TAG_FAMILIES:
            out.append(Finding("board.tag_family", "board", "fail",
                               f"tag_family {fam!r} not in {sorted(TAG_FAMILIES)}"))
        sp = b.get("tag_spacing")
        if sp is not None and not (0.0 < sp < 1.0):
            out.append(Finding("board.tag_spacing", "board", "fail",
                               f"tag_spacing {sp} must be a ratio (gap/tag size), not metres or mm"))
        ts = b.get("tag_size")
        if ts is not None and not (0.005 < ts < 1.0):
            out.append(Finding("board.tag_size", "board", "fail",
                               f"tag_size {ts} must be in metres"))
    elif b.get("board_type") == "charuco":
        out += _check_charuco(cfg.board)

    if mode == "cam-imu":
        ci, cin = cfg.input, cfg.camera_input
        _exists(out, "input.path", "input", ci.path, "dataset")
        if ci.format not in ("ros1", "ros2", "euroc"):
            out.append(Finding("input.format", "input", "fail",
                               f"cam-IMU input format {ci.format!r}: only ros1, ros2 or euroc"))
        if not ci.imu_topics:
            out.append(Finding("input.imu_topics", "input", "fail", "no imu_topics"))
        if not cin.path:
            out.append(Finding("camera_input.path", "camera_input", "fail",
                               "no camera prior: run camera-only first and point camera_input.path at it"))
        else:
            _exists(out, "camera_input.path", "camera_input", cin.path, "camera prior")
        est_imu = (raw.get("estimate") or {}).get("imu") or {}
        if raw.get("solve", {}).get("estimate_t_ga") is False or est_imu.get("t_ga") is False:
            out.append(Finding("estimate.t_ga", "estimate", "warn",
                               "t_ga (gyro-accel offset) disabled; MVIS default practice is on"))
        g, src, place = cfg.solve.gravity, cfg.solve.gravity_source, cfg.solve.gravity_place
        # a pass, not a warn: the default is legal and the run log warns about it (the published
        # cam-IMU example configs set gravity: since 2026-11-25)
        out.append(Finding("solve.gravity", "solve", "pass",
                           DEFAULT_HINT if src == "default" else
                           f"|g| {g:.4f} m/s^2 ({src}{', ' + place if place else ''})"))
        if not (raw.get("solve") or {}).get("rms_gate_px"):
            out.append(Finding("solve.rms_gate_px", "solve", "warn",
                               "rms_gate_px unset/0: a diverged solve can still report ok=true"))
        if not cfg.output.path:
            out.append(Finding("output.path", "output", "fail", "no output.path"))
        elif not os.path.dirname(cfg.output.path):
            out.append(Finding("output.path", "output", "warn",
                               f"output.path {cfg.output.path!r} is a bare filename: the summary, "
                               "its _calib.yaml and sidecars go to the current directory; "
                               "give a folder, e.g. results/<run>/cam_imu.json"))
    else:
        cams = (raw.get("input") or {}).get("cameras") or raw.get("cameras") or []
        if not cams:
            out.append(Finding("input.cameras", "input", "fail", "no cameras listed"))
        for c in cams:
            _exists(out, "camera.source_path", f"cam{c.get('camera_id', '?')}",
                    c.get("source_path"), "camera source")
            if c.get("source_type") == "video":
                out.append(Finding("camera.source_type", f"cam{c.get('camera_id')}", "fail",
                                   "source_type video is not supported; extract frames to images"))
    if not any(f.status == "fail" for f in out):
        out.append(Finding("config.ok", path, "pass", f"mode {mode}: no errors"))
    return out
