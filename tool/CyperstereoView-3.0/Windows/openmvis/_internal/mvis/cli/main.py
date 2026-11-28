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
"""MVIS CLI entry point."""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

from mvis.core import shutter
from mvis.core.config import CalibrationConfig
from mvis.core.gravity import DEFAULT_HINT
from mvis.core.types import FrameObservations
from mvis.detection.board_factory import create_board
from mvis.io.data_source import stream_cameras
from mvis.io.output_writer import OutputWriter
from mvis.calibration.initializer import Initializer
from mvis.calibration.optimizer import Optimizer, OptimizerOptions
from mvis.calibration.lm_params import LM_LAMBDA_LOWER_DEFAULT, LM_LAMBDA_UPPER_DEFAULT
from mvis.analysis.report import print_report
from mvis.io.cam_imu_loader import read_imu_only
from mvis.io.result_schema import calyx_dw_decomposed, cam_imu_sensor_rows, imu_diag
from mvis.io.segment_select import select_motion_window
from mvis.io.sensor_clock import load_retime_map, write_sensor_clock_sidecar

logger = logging.getLogger("mvis")


def _detect_all_frames(config: CalibrationConfig, board):
    """Step 0 with caching: this was the SEVENTH persistence path — the
    one detection site with no cache at all. Now it goes through the shared
    _make_cache seam like every other site; a second `mvis cam` run on the
    same inputs never decodes a frame."""
    from mvis.io.capture_key import board_fingerprint, capture_key
    from mvis.io.detection_pipeline import _make_cache

    # one capture key per camera: source and topic (every frame is detected; factor_hz
    # selects after detection, see _subsample_by_factor_hz). The key starts with the
    # source path so the cache identity stats that file: a re-recorded source at the
    # same path misses.
    sources_key = "||".join(
        capture_key(getattr(c, 'source_path', ''), [getattr(c, 'topic', '') or ''],
                    cam=c.camera_id, type=getattr(c, 'source_type', '') or None)
        for c in config.cameras)
    board_cfg = board_fingerprint(board)
    cache = _make_cache(str(Path(config.output_file).parent / "detections"))
    if cache.is_valid(sources_key, board_cfg):
        logger.info("Using cached detections — skipping data loading")
        frames, _hist, image_sizes = cache.load(sources_key, board_cfg)
        return _subsample_by_factor_hz(config, defaultdict(list, frames)), image_sizes

    all_frames: dict[int, list[FrameObservations]] = defaultdict(list)
    image_sizes: dict[int, tuple[int, int]] = {}
    for cam_id, image, timestamp in stream_cameras(config.cameras):
        observations = board.detect(image)
        if observations:
            h, w = image.shape[:2]
            image_sizes[cam_id] = (w, h)
            all_frames[cam_id].append(FrameObservations(
                camera_id=cam_id, timestamp=timestamp,
                observations=observations, image_size=(w, h)))
    for c in config.cameras:
        logger.info(f"Camera {c.camera_id}: detected board in "
                    f"{len(all_frames[c.camera_id])} frames")
    cache.save(dict(all_frames), sources_key, board_cfg, {}, image_sizes)
    return _subsample_by_factor_hz(config, all_frames), image_sizes


def _subsample_by_factor_hz(config, all_frames):
    """Each camera's ``factor_hz`` (> 0) on its detected frames, on the common-origin
    grid of ``frame_selector.select_frame_indices_by_hz`` (one origin for all cameras,
    so synced cameras keep the same instants). Cameras without a rate keep every frame.
    Before G7 the ROS1 and timestamped-image readers subsampled before detection on a
    per-camera grid, and the other readers ignored factor_hz."""
    from mvis.io.frame_selector import subsample_frames_by_hz
    rates = {c.camera_id: getattr(c, "factor_hz", None) for c in config.cameras}
    if not any(r and r > 0 for r in rates.values()):
        return all_frames
    hz = {cid: (r if r and r > 0 else float("inf")) for cid, r in rates.items()}
    return defaultdict(list, subsample_frames_by_hz(dict(all_frames), hz))


def calibrate(config: CalibrationConfig) -> None:
    """`mvis calibrate` (legacy flat CalibrationConfig): detect, init and solve in-process.
    `mvis cam` / `mvis run` use the pipeline driver instead (calibrate_cam, G8)."""
    board = create_board(config.board)
    initializer = Initializer(board)

    # Step 0: Detect board features (cached — see _detect_all_frames)
    logger.info("Detecting board features...")
    all_frames, image_sizes = _detect_all_frames(config, board)

    if not all_frames:
        logger.error("No board detections found in any camera!")
        sys.exit(1)

    # Step 1: Initialize
    logger.info("Initializing calibration parameters...")
    all_intrinsics = {}
    all_dist_coeffs = {}
    all_board_poses = {}
    distortion_models = {}

    for cam_config in config.cameras:
        cam_id = cam_config.camera_id
        frames = all_frames.get(cam_id, [])
        if not frames:
            continue

        dm = cam_config.distortion_model
        distortion_models[cam_id] = dm
        img_size = image_sizes.get(cam_id, (640, 480))

        max_init = config.optimizer.max_init_frames
        if max_init > 0 and len(frames) > max_init:
            from mvis.calibration.init_frame_selector import select_init_frames
            init_frames = select_init_frames(
                frames, max_init, img_size,
                strategy=config.optimizer.init_frame_selection)
            logger.info(
                f"Camera {cam_id}: subsampling {len(frames)} -> {len(init_frames)} "
                f"frames for init (max_init_frames={max_init}, "
                f"strategy={config.optimizer.init_frame_selection})"
            )
        else:
            init_frames = frames

        intrinsics, dist_coeffs = initializer.init_intrinsics_opencv(
            init_frames, img_size, dm
        )
        all_intrinsics[cam_id] = intrinsics
        all_dist_coeffs[cam_id] = dist_coeffs

        poses = initializer.init_board_poses(
            frames, intrinsics, dist_coeffs, dm,
            use_subprocess=getattr(config.optimizer, "pnp_subprocess", False))
        all_board_poses[cam_id] = poses

    # Per-tag corner subsampling for the optimization. Init above ran on the full
    # corner set (better PnP / intrinsic seed); the solve uses the thinned set.
    # Mirrors CalibrationPipeline; "all" is a no-op. (config.corner_subsample is the
    # camera-only canonical field on CalibrationConfig / CamRunConfig.)
    cs_mode = getattr(config, "corner_subsample", "all")
    from mvis.detection.charuco_board import CharucoBoard
    if isinstance(board, CharucoBoard):
        cs_mode = "all"           # ChArUco ids are chessboard corners: nothing to thin
    if cs_mode != "all":
        from mvis.detection.corner_subsample import filter_frames
        before = sum(len(f.observations) for fr in all_frames.values() for f in fr)
        all_frames = filter_frames(all_frames, cs_mode)
        after = sum(len(f.observations) for fr in all_frames.values() for f in fr)
        logger.info(
            f"Corner subsample ({cs_mode}): {before} -> {after} observations "
            f"({100.0 * after / max(before, 1):.1f}%)")

    # Step 2: Optimize
    logger.info("Optimizing...")

    # Build rolling-shutter camera map. Mirrors pipeline_cam.py:556-557:
    # explicit shutter_type == "rolling" enables RS readout estimation.
    # "auto" and "global" are treated as global shutter here — auto-detection
    # via detection statistics is only available through scripts/run_calibration.py.
    rs_cameras: dict[int, float] = {}
    if config.optimizer.optimize_rolling_shutter:
        for cam_config in config.cameras:
            if shutter.is_rolling(getattr(cam_config, "shutter_type", "auto")):
                rs_cameras[cam_config.camera_id] = 0.05  # prior sigma (s)
        if rs_cameras:
            logger.info(
                f"Rolling shutter enabled for cameras: {sorted(rs_cameras)}")
        elif any(getattr(c, "shutter_type", "") == "auto" for c in config.cameras):
            logger.warning(
                "shutter_type=auto is not supported by `mvis calibrate` — "
                "treating as global shutter. Use `mvis cam` for auto-detection.")

    opts = OptimizerOptions(
        max_iterations=config.optimizer.max_iterations,
        pixel_sigma=config.optimizer.pixel_sigma,
        huber_threshold=config.optimizer.huber_threshold,
        absolute_error_tol=config.optimizer.absolute_error_tol,
        relative_error_tol=config.optimizer.relative_error_tol,
        lm_lambda_lower=getattr(config.optimizer, "lm_lambda_lower",
                                LM_LAMBDA_LOWER_DEFAULT),
        lm_lambda_upper=getattr(config.optimizer, "lm_lambda_upper",
                                LM_LAMBDA_UPPER_DEFAULT),
        rs_cameras=rs_cameras or None,
        board_heights=bool(getattr(config.optimizer, "optimize_board_points", False)),
        verbose=True,
    )
    optimizer = Optimizer(board, opts)

    camera_ids = sorted(all_frames.keys())

    # double_sphere / omni_radtan: the pipeline's Stage 2 two-phase schedule
    # (omni_schedule.solve_phased) for each such camera; a single camera keeps
    # that fit, a rig starts its joint solve from it. Before #26 this path
    # solved omni in one go from the grid seed, and before G1 double_sphere.
    from mvis.calibration.omni_schedule import (
        PHASED_MODELS, poses_by_frame, solve_phased)
    phased_results = {}
    for cam_id in camera_ids:
        dm = distortion_models.get(cam_id)
        if dm not in PHASED_MODELS:
            continue
        r = solve_phased(
            optimizer, all_frames[cam_id], all_intrinsics[cam_id],
            all_dist_coeffs[cam_id], all_board_poses[cam_id], dm,
            image_sizes.get(cam_id, (0, 0)), cam_id)
        phased_results[cam_id] = r
        all_intrinsics[cam_id] = r.cameras[0].intrinsics
        all_dist_coeffs[cam_id] = r.cameras[0].distortion
        fitted = poses_by_frame(r, len(all_frames[cam_id]), all_board_poses[cam_id])
        all_board_poses[cam_id] = [p if p is not None else p0   # PnP seed where the
                                   for p, p0 in zip(fitted, all_board_poses[cam_id])]  # fit has none

    if len(camera_ids) == 1 and camera_ids[0] in phased_results:
        result = phased_results[camera_ids[0]]
    elif len(camera_ids) == 1:
        cam_id = camera_ids[0]
        result = optimizer.solve_single_camera(
            all_frames[cam_id],
            all_intrinsics[cam_id],
            all_dist_coeffs[cam_id],
            all_board_poses[cam_id],
            distortion_models[cam_id],
            image_sizes.get(cam_id, (0, 0)),
        )
    else:
        extrinsics = initializer.init_extrinsics(
            all_frames, all_board_poses, config.reference_camera
        )
        result = optimizer.solve_multi_camera(
            dict(all_frames),
            all_intrinsics,
            all_dist_coeffs,
            all_board_poses,
            extrinsics,
            distortion_models,
            config.reference_camera,
            image_sizes,
        )

    # kb4: log the fit's valid domain (WARNING when it folds inside the image)
    from mvis.calibration.pipeline_cam import _log_kb_valid_domain
    _log_kb_valid_domain(result, image_sizes)

    # Step 3: Report
    report = print_report(result)
    print(report)

    # Step 4: Save
    OutputWriter.write(result, config.output_file, config.output_format)
    logger.info(f"Results saved to {config.output_file}")


def board_pose_summary(T_BinW) -> dict:
    """Solved board pose in the gravity-aligned world (g = −z). The world's yaw and
    origin are gauge, so only the board's attitude to gravity is comparable across
    runs: `g_inB` (unit gravity direction in board coordinates), the tilt of the
    board normal from vertical (`normal_tilt_deg`, 90 = a vertical board) and the
    in-plane direction of gravity (`g_inplane_deg`, atan2 of g_inB y over x)."""
    T = np.asarray(T_BinW.matrix() if hasattr(T_BinW, "matrix") else T_BinW, float)
    R = T[:3, :3]
    g_inB = R.T @ np.array([0.0, 0.0, -1.0])
    return dict(T_BinW=np.round(T, 6).tolist(),
                g_inB=[round(float(x), 6) for x in g_inB],
                normal_tilt_deg=round(float(np.degrees(np.arccos(np.clip(-g_inB[2], -1, 1)))), 4),
                g_inplane_deg=round(float(np.degrees(np.arctan2(g_inB[1], g_inB[0]))), 4))


def write_board_shape(rep: dict, path, source: str = "") -> dict:
    """Write the cam-IMU estimated board shape as a `board.heights_file`-compatible
    YAML (`board_shape.heights_m`, plus sigma and observation counts) and return
    the compact summary block for cam_imu.json."""
    compact = {k: (round(float(rep[k]), 4) if rep.get(k) is not None else None)
               for k in ("rms_mm", "p2p_mm", "offset_rms_mm")}
    compact["n_seen"] = int(rep["n_seen"])
    if path:
        import os

        import yaml
        sig = rep.get("sigma_m")
        doc = {"board_shape": {
            "model": "pose + per-corner height",
            "source": f"cam-IMU estimate (estimate.board_points), {source}",
            "heights_m": [float(x) for x in rep["heights_m"]],
            "sigma_m": None if sig is None else [float(x) for x in sig],
            "n_obs": [int(x) for x in rep["n_obs"]],
            "initial_m": [float(x) for x in rep["initial_m"]]}}
        # Written before the solve's other outputs, so its folder may not exist yet (a fresh output path).
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
        compact["file"] = path
        logger.info("wrote estimated board shape %s (rms %.2f mm, p-p %.2f mm, "
                    "%d corners seen)", path, rep["rms_mm"] or 0, rep["p2p_mm"] or 0,
                    rep["n_seen"])
    return compact


def load_board_heights(path: str, n_points: int) -> np.ndarray:
    """Per-corner out-of-plane heights (m) from a `mvis cam` calibration YAML
    (`board_shape.heights_m`) or an .npz with `dp` (N x 3 board-frame offsets,
    z column used; NaN = unobserved -> 0)."""
    if path.endswith(".npz"):
        dp = np.asarray(np.load(path)["dp"], float)
        h = np.nan_to_num(dp[:, 2])
    else:
        import yaml
        with open(path) as f:
            y = yaml.safe_load(f)
        h = np.asarray(y["board_shape"]["heights_m"], float)
    if len(h) != n_points:
        raise ValueError(f"board heights_file {path}: {len(h)} heights for a "
                         f"{n_points}-corner board")
    return h


def apply_board_heights(pts_B: np.ndarray, board_cfg) -> np.ndarray:
    """Board corners as (x, y, h) when board.heights_file is set, else unchanged."""
    path = getattr(board_cfg, "heights_file", None)
    if not path:
        return pts_B
    h = load_board_heights(path, len(pts_B))
    out = pts_B.copy()
    out[:, 2] += h
    logger.info(f"Board heights from {path}: rms {np.std(h)*1e3:.2f} mm, "
                f"peak-to-peak {(h.max()-h.min())*1e3:.2f} mm")
    return out


def _resolve_window(ci, _board):
    """Return (t0, window, abs_window) for the loaders.

    motion mode (default): runs the IMU pre-scan + selector → abs_window;
    manual mode: reproduces today's behaviour exactly (t0, window-or-1e9, None).
    If motion selector returns None (short bag), falls back to manual window.
    """
    if getattr(ci, "window_mode", "manual") == "motion":
        streams = read_imu_only(ci.path, ci.imu_topics, fmt=ci.format)
        sel = select_motion_window(streams, getattr(ci, "segment_length", 80.0))
        if sel is not None:
            if ci.window or ci.t0:          # never silently drop an explicit window
                logger.info("window_mode=motion: selected [%.1f, %.1f] s, "
                            "overriding configured t0=%.1f/window=%.1f "
                            "(set window_mode: manual to keep them)",
                            sel[0], sel[1], ci.t0, ci.window)
            return ci.t0, 1e9, sel          # abs_window drives the loader
        logger.info("window_mode=motion: bag < segment_length; "
                    "falling back to manual window (t0=%.1f)", ci.t0)
    win = ci.window if ci.window else 1e9
    return ci.t0, win, None


def _apply_imu_gyro_only(imu_topics, aux_imus, imu_overrides):
    """Set ``gyro_only`` on aux IMUs from the per-IMU config overrides (``input.imus``,
    matched by ``imu_topics`` order; index 0 = base). Raises if the base IMU is flagged
    ``gyro_only`` — the base defines gravity + metric scale and needs the accelerometer.
    Returns the (mutated) ``aux_imus`` list. ``aux_imus`` correspond to ``imu_topics[1:]``."""
    if not imu_overrides:
        return aux_imus
    if imu_overrides[0].get("gyro_only"):
        raise ValueError(
            "gyro_only is not allowed on the base IMU (imu_topics[0]); the base defines "
            "gravity + metric scale and needs the accelerometer. Set it on an aux IMU.")
    for i, aux in enumerate(aux_imus):
        ov = imu_overrides[i + 1] if i + 1 < len(imu_overrides) else {}
        if ov.get("gyro_only"):
            aux["gyro_only"] = True
    return aux_imus


def _override_source(config, bag: str):
    """Point a cam-IMU run config at a ROS1 bag, in place, so ONE config (e.g. a euroc
    forward/repeatability config) can also feed the ros1-only apply-calib / roundtrip. Flips
    ``input.format`` -> ``ros1`` and ``input.path`` -> ``bag``; camera/IMU topics still
    auto-derive from the camchain when its ``rostopic`` entries match the bag topics (the MIC
    case). Pass explicit ``cam_topics`` in the config if a rig's bag topics differ."""
    config.input.format = "ros1"
    config.input.path = bag
    return config


def _bag_name(bag: str, i: int) -> str:
    """``bag<NN>`` from a source path (so the repeatability study's ``discover()`` finds it): the
    trailing number of a ros1 bag filename (``..._09.bag`` -> ``bag09``), or — for a EuRoC
    ``.../MIC04_imucalib1/mav0`` dir, whose basename carries no identity — the FIRST number of
    the parent dir (``MIC04_…`` -> ``bag04``). Falls back to ``bag<i+1>`` (1-based)."""
    import os
    import re
    p = bag.rstrip("/")
    stem = os.path.splitext(os.path.basename(p))[0]
    if stem == "mav0":                              # euroc: the sequence id is the parent dir
        m = re.search(r"(\d+)", os.path.basename(os.path.dirname(p)))
    else:
        m = re.search(r"(\d+)\D*$", stem)           # ros1: the trailing number
    return f"bag{m.group(1)}" if m else f"bag{i + 1:02d}"


def batch_plan(config, out_dir: str):
    """``[(name, bag, output_json)]`` for each bag in ``config.input.bags`` (or the single
    ``input.path``), output templated to ``<out_dir>/bag<NN>/cam_imu.json``."""
    import os
    bags = config.input.bags or ([config.input.path] if config.input.path else [])
    return [(_bag_name(b, i), b, os.path.join(out_dir, _bag_name(b, i), "cam_imu.json"))
            for i, b in enumerate(bags)]


def calibrate_cam_imu_batch(config, out_dir: str) -> dict:
    """Run :func:`calibrate_cam_imu` for each bag in the config's bag set, writing a discoverable
    ``<out_dir>/bag<NN>/cam_imu.json`` per bag. Returns ``{name: summary}``."""
    import copy
    import os
    results = {}
    for name, bag, out_json in batch_plan(config, out_dir):
        cfg = copy.deepcopy(config)
        cfg.input.bags = []                 # avoid re-entry; this clone runs one source
        cfg.input.path = bag                # inherit the config's format (ros1 bag OR euroc mav0)
        cfg.output.path = out_json
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        results[name] = calibrate_cam_imu(cfg)
    return results


def calibrate_cam_imu(config) -> dict:
    """Run the joint cam-IMU pipeline from one CamImuRunConfig.

    Single config drives: dataset load (ROS1/EuRoC), prior camera intrinsics
    (MVIS/Kalibr via R4), the joint solve, and the complete output (R5
    `_calib.yaml` + Kalibr `camchain-imucam`).
    """
    import os
    import time
    import json
    from mvis.detection.board_factory import create_board
    from mvis.io.camera_input import load_camera_intrinsics
    from mvis.io.cam_imu_loader import load_ros1, load_euroc
    from mvis.io.cam_imu_output import write_cam_imu_calibration
    from mvis.calibration.pipeline_cam_imu import run_cam_imu_calibration_multi

    ci, cin, out_cfg = config.input, config.camera_input, config.output
    if config.solve.gravity_source == "default":
        logger.warning(DEFAULT_HINT)
    board = create_board(config.board)
    pts_B = apply_board_heights(np.asarray(board.all_points_3d(), float), config.board)

    specs = load_camera_intrinsics(cin.path)
    cam_topics = ci.cam_topics or [s["rostopic"] for s in specs]
    imu_topics = ci.imu_topics
    t0, win, abs_window = _resolve_window(ci, board)
    if ci.format == "euroc":          # map topics -> mav0 subdirs (cam0/imu0)
        cam_subdirs = [t.strip("/").split("/")[0] for t in cam_topics]
        imu_subdirs = [t.strip("/").split("/")[0] for t in imu_topics]
        loaded, imus = load_euroc(ci.path, cam_subdirs, imu_subdirs, board,
                                  t0=t0, window=win, abs_window=abs_window,
                                  min_obs=ci.min_obs, cache_dir=ci.cache_dir)
    else:
        loaded, imus = load_ros1(ci.path, cam_topics, imu_topics, board,
                                 t0=t0, window=win, abs_window=abs_window,
                                 min_obs=ci.min_obs, cache_dir=ci.cache_dir)
    t_imu, w_m, a_m = imus[0]["t"], imus[0]["w"], imus[0]["a"]
    # additional IMUs (imu_topics[1:]) are calibrated as auxiliaries — their
    # T_IstoI / inter-IMU td / intrinsics are solved jointly with the base.
    aux_imus = [dict(t_imu=im["t"], w_m=im["w"], a_m=im["a"], name=im["name"])
                for im in imus[1:]]
    # per-IMU overrides (input.imus, by imu_topics order) — e.g. gyro_only for a bad accel
    _apply_imu_gyro_only(imu_topics, aux_imus, getattr(ci, "imus", []))
    cam_over = getattr(cin, "cameras", None) or []   # per-camera overrides (by order)
    cams = []
    for i, (s, lc) in enumerate(zip(specs, loaded)):
        ov = cam_over[i] if i < len(cam_over) else {}
        cam = dict(
            t_cam=lc["t_cam"], ids_list=lc["ids_list"], uv_list=lc["uv_list"],
            frames=lc.get("frames"),
            cal_vec=np.array(s["intrinsics"] + s["distortion_coeffs"], float),
            is_fisheye=s["is_fisheye"], model=s["distortion_model"],
            image_size=tuple(s["resolution"]),
            pin_intrinsics=not ov.get("free_intrinsics", cin.free_intrinsics),
            estimate_readout=ov.get("estimate_readout", cin.estimate_readout),
            rolling_mode=ov.get("rolling_mode", cin.rolling_mode),
            # prior readout: only used to bracket rows when solve.knot_hz > 0
            tr_hint=float(s.get("rolling_shutter_readout", 0.0) or 0.0),
            # stamps not on the sensor clock -> two-pass sensor-clock retiming
            retime_sensor_clock=bool(ov.get("retime_sensor_clock", False)))
        # a saved stamp map (`<summary>_sensor_clock.json` of an earlier retiming
        # run) re-stamps this camera directly, skipping pass 1
        if ov.get("retime_map"):
            cam["retime_map"] = load_retime_map(ov["retime_map"], i)
        # per-camera vision-factor rate (e.g. base dense, aux sparse). Only set when
        # given; otherwise the solver falls back to solve.factor_hz / state_hz.
        if ov.get("factor_hz") is not None:
            cam["factor_hz"] = ov["factor_hz"]
        # per-camera vision factor ("interp" | "imu"); unset = solve.vision_pose
        if ov.get("vision_pose"):
            if ov["vision_pose"] not in ("interp", "imu"):
                raise ValueError(f"camera_input.cameras[{i}].vision_pose must be "
                                 f"'interp' or 'imu', got {ov['vision_pose']!r}")
            cam["vision_pose"] = ov["vision_pose"]
        cams.append(cam)
    logger.info("frames/cam: %s, base imu %d, aux imus %s",
                [len(c["t_cam"]) for c in cams], len(t_imu),
                [a["name"] for a in aux_imus] or "none")

    t0 = time.time()
    r = run_cam_imu_calibration_multi(cams, pts_B, t_imu, w_m, a_m,
                                      aux_imus=aux_imus or None,
                                      config=config.solve)
    d = r.diagnostics
    summary = dict(input=os.path.basename(ci.path), ok=bool(r.ok),
                   reproj_rms_px=round(float(d["reproj_rms_px"]), 3),
                   reproj_median_px=(round(float(d["reproj_median_px"]), 3)
                                     if d.get("reproj_median_px") is not None else None),
                   reproj_p95_px=(round(float(d["reproj_p95_px"]), 3)
                                  if d.get("reproj_p95_px") is not None else None),
                   **_reproj_extra(d, "_px"),
                   solve_s=round(time.time() - t0, 1), cams=[])
    # solver health for `mvis verdict` (ok alone does not mean a good fit)
    summary["diagnostics"] = {
        k: (round(float(d[k]), 4) if isinstance(d.get(k), float) else d.get(k))
        for k in ("iterations", "cam_chi2_dof", "imu_chi2_dof", "n_states", "lm_no_step",
                  "sensor_clock_pass1_lm_no_step")
        if d.get(k) is not None}
    summary["diagnostics"]["max_iterations"] = int(config.solve.max_iterations)
    summary["board"] = board_pose_summary(r.T_BinW)
    if d.get("board_heights"):
        bs_path = (os.path.splitext(out_cfg.path)[0] + "_board_shape.yaml"
                   if out_cfg.path else None)
        summary["board"]["shape"] = write_board_shape(d["board_heights"], bs_path,
                                                      os.path.basename(ci.path))
    if d.get("sensor_clock"):
        # compact report in the summary; the per-frame stamp maps go to a sidecar
        # that `camera_input.cameras[i].retime_map` (or the bag restamp tool) reuses
        sc_path = (os.path.splitext(out_cfg.path)[0] + "_sensor_clock.json"
                   if out_cfg.path else None)
        summary["diagnostics"]["sensor_clock"] = write_sensor_clock_sidecar(
            d["sensor_clock"], sc_path, cam_topics)
    _summary_sensor_fields(r, summary)

    # ensure all output directories exist (CLI may target a fresh results/ tree)
    for _p in (out_cfg.path, out_cfg.calib_yaml, out_cfg.kalibr_yaml,
               out_cfg.report_html, out_cfg.report_txt):
        if _p and os.path.dirname(_p):
            os.makedirs(os.path.dirname(_p), exist_ok=True)

    if out_cfg.path:
        with open(out_cfg.path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("wrote summary %s", out_cfg.path)

    # Attach this run + estimates to the bag's capture db
    # (the loader caches all cameras of the bag in ONE file). Additive: a
    # failure here never breaks the summary/YAML outputs.
    _dbp = (loaded[0].get("cache_path") if loaded else None)
    if r.ok and _dbp and os.path.exists(_dbp):
        try:
            from mvis.mvisdb.estimates import estimates_from_cam_imu_result
            from mvis.mvisdb.manager import DatabaseManager
            from mvis.mvisdb.mvisdatabase import CalibChoice, Run
            _mgr = DatabaseManager()
            _db = _mgr.load(_dbp)
            _db.run = Run(CalibChoice.CAM_IMU, status="ok",
                          description=os.path.basename(ci.path),
                          state_hz=getattr(config.solve, "state_hz", None))
            _es = estimates_from_cam_imu_result(
                r, state_hz=getattr(config.solve, "state_hz", None))
            # shim-era captures declare no IMUs (frames-only cache API), and
            # result_imu is FK-gated on the sealed core — persist what the
            # capture declares, drop the rest loudly. The fix is the
            # capture-completeness item: the loader building the db natively
            # with IMU streams before first save.
            from mvis.mvisdb.mvisdatabase import SensorType as _ST
            _declared = {s.imu_id for s in _db.imus}
            _dropped = [ss.sensor for ss in _es.sensor_states
                        if ss.sensor[0] is _ST.IMU and ss.sensor[1] not in _declared]
            if _dropped:
                _es.sensor_states = [ss for ss in _es.sensor_states
                                     if ss.sensor not in _dropped]
                logger.warning(
                    "capture db lacks IMU declarations (shim-era file); "
                    "IMU states %s not persisted — camera estimates and "
                    "extrinsics vs IMU still attached", _dropped)
            _db.estimate_states = _es
            # 2c-12: fold the solve's per-frame selection story into the
            # capture flags (parity with the cam-only ledger). Cam order in
            # the result matches the loader's stream order.
            _n_flag = 0
            for _c, _selmap in enumerate(getattr(r, "frame_selection", []) or []):
                try:
                    _stream = _db.camera(_c)
                except Exception:
                    continue
                for _f in _stream.frames:
                    _reason = _selmap.get(float(_f.timestamp))
                    if _reason is not None and _f.is_selected_to_use:
                        _f.is_selected_to_use = False
                        _f.exclude_reason = _reason
                        _n_flag += 1
            if _n_flag:
                logger.info("selection story: %d frames flagged in overlay",
                            _n_flag)
            _rid = _mgr.save(_db, _dbp)
            logger.info("estimates attached to %s as run %s",
                        os.path.basename(_dbp), _rid)
        except Exception as e:
            logger.warning("mvisdb estimate attach skipped: %s", e)
    calib_yaml = out_cfg.calib_yaml or (
        os.path.splitext(out_cfg.path)[0] + "_calib.yaml" if out_cfg.path else None)
    kalibr_yaml = _kalibr_camchain_path(
        out_cfg, [e["calib"].model for e in r.cameras],
        [e["calib"].intr for e in r.cameras])
    if calib_yaml:
        write_cam_imu_calibration(r, calib_yaml, kalibr_path=kalibr_yaml,
                                  cam_topics=cam_topics, imu_topics=imu_topics)
        logger.info("wrote %s%s", calib_yaml,
                    f" + {kalibr_yaml}" if kalibr_yaml else "")
    # I5-3 shadow: the camchain must be reproducible as an export VIEW of the
    # attached run. Exact compare except T_cn_cnm1 (recomposed through Pose3,
    # matches to float precision) and rostopic (the writer has the bag topics, the db view /camN/...).
    if kalibr_yaml and r.ok and _dbp and '_rid' in dir() and _rid:
        try:
            import numpy as _np
            import yaml as _y
            from mvis.mvisdb.exports import export_kalibr_camchain
            _vp = kalibr_yaml + f".export_check_{_rid}.yaml"
            export_kalibr_camchain(_mgr, _dbp, _rid, _vp)
            _a = _y.safe_load(open(kalibr_yaml)); _b = _y.safe_load(open(_vp))
            assert set(_a) == set(_b), "camera set differs"
            for _k in _a:
                _ta = _a[_k].pop("T_cn_cnm1", None)
                _tb = _b[_k].pop("T_cn_cnm1", None)
                assert (_ta is None) == (_tb is None), f"{_k}: T_cn_cnm1 presence"
                if _ta is not None:
                    assert _np.allclose(_ta, _tb, atol=1e-9), f"{_k}: T_cn_cnm1"
                _a[_k].pop("rostopic", None); _b[_k].pop("rostopic", None)
                assert _a[_k] == _b[_k], f"{_k}: fields differ"
            os.unlink(_vp)
            logger.info("I5 camchain export view verified against writer")
        except AssertionError as _e:
            logger.error("I5 CAMCHAIN VIEW MISMATCH (%s) — kept %s", _e, _vp)
            raise SystemExit(3)
        except Exception as _e:
            logger.warning("camchain view check skipped: %s", _e)
    # R6: analysis report (HTML + text), with optional vs-reference section.
    from mvis.analysis.cam_imu_report import write_cam_imu_report
    report_html = out_cfg.report_html or (
        os.path.splitext(out_cfg.path)[0] + "_report.html" if out_cfg.path else None)
    report_txt = out_cfg.report_txt
    ref = getattr(config, "reference", None) or None
    # per-camera detections + thumbnails for the scene's camera view / coverage
    cam_data = [dict(frames=lc.get("frames"), thumb_dir=lc.get("thumb_dir"),
                     image_size=c["image_size"]) for lc, c in zip(loaded, cams)]
    txt = write_cam_imu_report(r, html_path=report_html, text_path=report_txt,
                               reference=ref, board=board,
                               config=getattr(config, "_source_yaml", None) or config,
                               cam_data=cam_data,
                               result_files=[p for p in (calib_yaml, kalibr_yaml, out_cfg.path)
                                             if p and os.path.exists(p)])
    print(txt)
    if report_html:
        logger.info("wrote report %s", report_html)
    return summary


def _r3(v):
    return None if v is None or not np.isfinite(float(v)) else round(float(v), 3)


def _reproj_extra(e: dict, suffix: str) -> dict:
    """The cam_imu.json reprojection fields after rms/median/p95: mean, std, p90 and max over all
    corners, the corner count, and the inlier population (`reproj_stats.inlier_stats`). Nothing for
    a result made before these were recorded, so old results keep their exact summary."""
    if "reproj_mean_px" not in e:
        return {}
    out = {f"reproj_{k}{suffix}": _r3(e.get(f"reproj_{k}_px")) for k in ("mean", "std", "p90", "max")}
    out["reproj_n_corners"] = int(e["reproj_n_corners"])
    inl = e.get("reproj_inlier")
    if inl:
        out["reproj_inlier"] = {**{k: _r3(inl.get(k)) for k in ("mean", "median", "std", "p90", "p95", "max")},
                                "n_corners": int(inl["n_corners"]), "n_excluded": int(inl["n_excluded"]),
                                "threshold_px": _r3(inl["threshold_px"])}
    return out


def _summary_sensor_fields(r, summary: dict) -> None:
    """Add the per-sensor fields of the cam_imu.json summary, in place and in file order:
    the ``cams`` rows, the base-IMU model and Dw/Da diagonals, ``t_ga`` and the aux ``imus``
    rows. Values come from `result_schema.cam_imu_sensor_rows` (shared with the report),
    rounded here for the summary."""
    d = r.diagnostics
    rows = cam_imu_sensor_rows(r, require_sigma=False)
    for c, cr in enumerate(rows["cams"]):
        e, cal = cr["entry"], cr["calib"]
        cd = dict(name=f"cam{c}", td_ms=round(1e3 * cal.td, 3),
                  reproj=round(float(e["reproj_rms_px"]), 3),
                  reproj_median=(round(float(e["reproj_median_px"]), 3)
                                 if e.get("reproj_median_px") is not None else None),
                  reproj_p95=(round(float(e["reproj_p95_px"]), 3)
                              if e.get("reproj_p95_px") is not None else None),
                  **_reproj_extra(e, ""),
                  intr=[round(float(v), 5) for v in cal.intr],
                  p_CinI_mm=[round(1e3 * v, 2) for v in cr["p_m"]],
                  rotvec=[round(v, 5) for v in cr["rotvec"]])
        if e.get("corner_accounting") is not None:
            cd["corner_accounting"] = e["corner_accounting"]
        if cr["baseline_m"] is not None:
            cd["p_CtoC0_mm"] = [round(1e3 * v, 2) for v in cr["baseline_m"]]
        summary["cams"].append(cd)
    model, ii = rows["base"]["model"], rows["base"]["intr"]
    if ii is not None:
        summary["imu_model"] = model
        summary["Dw_diag"], summary["Da_diag"] = imu_diag(ii, 5)
        if model == "calyx":       # decomposed physical view (user default)
            summary.update(calyx_dw_decomposed(ii.Dw(), 5))
    if d.get("gravity") is not None:      # |g| the solve held fixed (mvis.core.gravity)
        summary["gravity_mps2"] = round(float(d["gravity"]), 6)
        summary["gravity_source"] = d.get("gravity_source", "default")
        if d.get("gravity_site"):
            summary["gravity_site"] = d["gravity_site"]
        if d.get("gravity_place"):
            summary["gravity_place"] = d["gravity_place"]
    sr = d.get("seed_report") or {}
    if "g_norm_ok" in sr:   # init checks judge the seed only; ok does not use them
        summary["init_gates"] = {"g_norm_ls": round(float(sr["g_norm_ls"]), 4),
                                 "g_norm_ok": bool(sr["g_norm_ok"]),
                                 "scale": round(float(sr["scale"]), 4),
                                 "scale_ok": sr.get("scale_ok")}
    if d.get("t_ga") is not None:                  # gyro<->accel time offset
        summary["t_ga_ms"] = round(1e3 * d["t_ga"], 4)
        if d.get("t_ga_sigma") is not None:
            summary["t_ga_sigma_ms"] = round(1e3 * d["t_ga_sigma"], 4)

    # aux IMUs (use-all-imus): per-aux T_IstoI lever/rotation, inter-IMU td, Dw/Da —
    # needed so the cam-IMU regression can compare every IMU, not just the base.
    summary["imus"] = []
    for ar in rows["aux"]:
        m = ar["entry"]
        a = dict(name=m.get("name", "aux"),
                 td_ms=round(1e3 * float(m.get("td", 0.0)), 3))
        if m.get("t_ga") is not None:              # aux IMU's own gyro<->accel offset
            a["t_ga_ms"] = round(1e3 * float(m["t_ga"]), 4)
            if m.get("t_ga_sigma") is not None:
                a["t_ga_sigma_ms"] = round(1e3 * float(m["t_ga_sigma"]), 4)
        if "chi2_dof" in m:
            a["chi2_dof"] = round(float(m["chi2_dof"]), 3)
        if ar["p_m"] is not None:
            a["p_IsinI_mm"] = [round(1e3 * v, 2) for v in ar["p_m"]]
            a["rotvec"] = [round(v, 5) for v in ar["rotvec"]]
        if ar["intr"] is not None:
            a["Dw_diag"], a["Da_diag"] = imu_diag(ar["intr"], 5)
            if model == "calyx":
                a.update(calyx_dw_decomposed(ar["intr"].Dw(), 5))
        summary["imus"].append(a)


def _kalibr_camchain_path(out_cfg, models, cal_vecs=None) -> str | None:
    """The Kalibr camchain-imucam path to write (explicit `output.kalibr_yaml`,
    else derived from `output.path`), or None. Kalibr has no equivalent for
    a model outside `KALIBR_EXPORTABLE`, or for some values (ucm alpha outside [0, 1), given
    `cal_vecs`): then the camchain is skipped with a warning instead of
    failing a finished solve — the MVIS yaml still carries the full result."""
    import os
    from mvis.core.camera_models import KALIBR_EXPORTABLE
    from mvis.io.cam_imu_output import kalibr_export_issue
    path = out_cfg.kalibr_yaml or (
        os.path.splitext(out_cfg.path)[0] + "_camchain-imucam.yaml"
        if out_cfg.path else None)
    bad = sorted({m for m in models if m not in KALIBR_EXPORTABLE})
    if path and bad:
        logger.warning("Kalibr camchain-imucam not written (%s): model(s) %s "
                       "have no Kalibr equivalent", path, ", ".join(bad))
        return None
    issues = [f"cam{c}: {i}" for c, (m, v) in enumerate(zip(models, cal_vecs or []))
              if (i := kalibr_export_issue(m, v))]
    if path and issues:
        logger.warning("Kalibr camchain-imucam not written (%s): %s",
                       path, "; ".join(issues))
        return None
    return path


def calibrate_cam(path: str, output: str | None = None) -> None:
    """`mvis cam` / `mvis run` (mode: cam): the camera-only driver
    ``mvis.calibration.cam_run.run_cam`` (the one the regression runs), which also writes
    the result to the config's ``output_file``. The run folder (report, logs, detection
    cache) is ``output``, default ``<output_file parent>/ablation``; run_cam clears it
    first, except the caches, when MVIS created it and refuses any other existing
    folder. Exits nonzero when the solve fails."""
    from mvis.calibration.cam_run import OutputDirRefused, configure_process, run_cam
    from mvis.core.config import CamRunConfig
    configure_process()     # OpenCV threads off, as in the regression's driver
    try:
        rc = run_cam(path, output, result_file=CamRunConfig.from_yaml(path).output_file)
    except OutputDirRefused as e:     # the run folder is not one MVIS created
        raise SystemExit(f"mvis cam: {e}") from None
    if rc:
        raise SystemExit(rc)


def _run_dispatch(path: str) -> None:
    """`mvis run`: dispatch on the config's top-level ``mode:`` field
    ("cam" → camera-only, "cam-imu" → joint cam-IMU). Defaults to "cam"."""
    import yaml as _yaml
    from mvis.core.config import CamImuRunConfig
    with open(path) as f:
        mode = (_yaml.safe_load(f) or {}).get("mode", "cam")
    if mode == "cam-imu":
        print(json.dumps(calibrate_cam_imu(CamImuRunConfig.from_yaml(path)), indent=1))
    elif mode == "cam":
        calibrate_cam(path)
    else:
        raise SystemExit(
            f"unknown mode {mode!r} in {path} (expected 'cam' or 'cam-imu')")


_EXIT = {"pass": 0, "warn": 1, "fail": 2, "GREEN": 0, "YELLOW": 1, "RED": 2}


def _write_json(path, d) -> None:
    """Write an agent-tool report to ``path``, creating parent folders."""
    from mvis.io.fileutil import write_json
    write_json(path, d, indent=1)


def _agent_tool(args) -> int:
    """`mvis inspect / check / verdict`: print text or JSON, return the exit code."""
    from dataclasses import asdict
    if args.command == "inspect":
        from mvis.doctor.inspect import inspect, format_text
        rep = inspect(args.path)
        d = rep.to_dict()
        if args.out:
            _write_json(args.out, d)
        print(json.dumps(d, indent=1) if args.json else format_text(rep))
        return _EXIT[rep.status]
    if args.command == "check":
        from mvis.doctor.check_config import check_env, check_config, config_needs
        from mvis.doctor.findings import format_finding, worst_status
        if args.smoke or not (args.env or args.config):
            args.env = True
        needs = config_needs(args.config) if args.config else set()
        found = (check_env(needs, smoke=args.smoke) if args.env else []) + \
            (check_config(args.config) if args.config else [])
        status = worst_status(f.status for f in found)
        if args.json:
            print(json.dumps({"status": status,
                              "findings": [asdict(f) for f in found]}, indent=1))
        else:
            for f in found:
                print(format_finding(f))
            print(f"check: {status.upper()}")
        if args.out:
            _write_json(args.out, {"status": status, "findings": [asdict(f) for f in found]})
        return _EXIT[status]
    from mvis.analysis.verdict import assess, format_text
    reference = None
    if args.config:
        import yaml as _yaml
        with open(args.config) as f:
            reference = (_yaml.safe_load(f) or {}).get("reference") or None
    v = assess(args.result, reference)
    v.independent_check = args.independent_check
    d = v.to_dict()
    if args.out:
        _write_json(args.out, d)
    print(json.dumps(d, indent=1) if args.json else format_text(v))
    return _EXIT[v.grade]


def main():
    """CLI entry point (the `mvis` console script): parse the subcommand
    (calibrate / cam / cam-imu / run / apply-calib / roundtrip / …) and
    dispatch to its handler. Exits nonzero via SystemExit on unknown modes."""
    parser = argparse.ArgumentParser(
        description="MVIS: open-source multi-visual-inertial sensor calibration (camera and camera-IMU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "environment:\n"
            "  MVIS_BLAS_THREADS   BLAS (OpenBLAS/OMP/MKL) threads for this process; "
            "default 1 —\n"
            "                      the solvers run per-factor Python on small matrices, so "
            "extra\n"
            "                      BLAS threads only spin (S0: ~590 s sys per cam-IMU solve) "
            "and make\n"
            "                      results thread-count dependent. 0 = BLAS default. An "
            "exported\n"
            "                      OPENBLAS/OMP/MKL_NUM_THREADS is never overridden "
            "(doc/blas_threads.md).\n"
            "  MVIS_EIGEN_CACHE_CPU  CPU on which Eigen reads the cache sizes it blocks GTSAM's\n"
            "                      dense products with; default the lowest allowed CPU, so a\n"
            "                      solve does not depend on the core it starts on (hybrid\n"
            "                      CPUs). off = read wherever the first product runs.\n"
        ),
    )
    from mvis import __version__
    parser.add_argument("--version", action="version", version=f"mvis {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    # Calibrate command
    cal_parser = subparsers.add_parser("calibrate", help="Run calibration")
    cal_parser.add_argument(
        "--config", "-c", required=True, help="Path to YAML config file"
    )

    # Camera-only (sectioned CamRunConfig shape; mirror of cam-imu). Accepts both
    # the new sectioned shape AND the legacy flat CalibrationConfig YAML.
    cam_parser = subparsers.add_parser(
        "cam", help="Run camera-only calibration (CamRunConfig YAML)")
    cam_parser.add_argument(
        "--config", "-c", required=True, help="Path to camera YAML config")
    cam_parser.add_argument(
        "--output", "-o", default=None,
        help="run folder for the report, logs and detection cache "
             "(default <output_file parent>/ablation). An existing folder is cleared "
             "first, except caches, only when MVIS created it; any other existing "
             "folder is refused")

    # Cam-IMU command: joint camera-IMU calibration from one YAML.
    camimu_parser = subparsers.add_parser(
        "cam-imu", help="Run joint camera-IMU calibration from a YAML config")
    camimu_parser.add_argument(
        "--config", "-c", required=True, help="Path to cam-IMU YAML config")
    camimu_parser.add_argument(
        "--bag", help="override the config's source with this ROS1 bag (format->ros1, path->bag)")

    # apply-calib: apply a calibration result to a raw bag → corrected bag.
    rc_parser = subparsers.add_parser(
        "apply-calib", help="Apply a calib result to a raw bag → corrected bag")
    rc_parser.add_argument("-c", "--config", dest="config_path",
                           help="recalib config YAML (calib/input/output/flags)")
    rc_parser.add_argument("--calib",
                           help="calibration result YAML to apply (or set `calib:` in --config)")
    rc_parser.add_argument("--in", dest="in_bag")
    rc_parser.add_argument(
        "--out", dest="out_bag",
        help="corrected output bag (default <input>.corrected.bag; an existing bag "
             "is not overwritten)")
    rc_parser.add_argument("--map", dest="map_path",
                           help="YAML: {base_imu_topic, mapping:{topic:sensor}}")
    rc_parser.add_argument("--keep-bias", action="store_true",
                           help="keep the static IMU biases (config remove_bias: false)")
    rc_parser.add_argument("--undistort", action="store_true",
                           help="undistort mono8 images (config undistort: true)")
    rc_parser.add_argument("--no-imu-intrinsics", action="store_true",
                           help="skip the inverse IMU intrinsic model; only the bias is "
                                "removed (config imu_intrinsics: false)")
    rc_parser.add_argument("--no-td", action="store_true",
                           help="keep the original header and record stamps (config td: false)")
    rc_parser.add_argument("--no-t-ga", action="store_true",
                           help="skip the accel t_ga re-interpolation (config t_ga: false)")
    rc_parser.add_argument("--strict", action="store_true",
                           help="reject missing or empty mapped topics, wrong message types, "
                                "unsupported undistortion input, duplicate IMU stamps on a t_ga "
                                "topic and non-finite corrected IMU values instead of passing "
                                "them through (config strict: true)")

    # Round-trip validation: calibrate → apply-calib → recalibrate → PASS/FAIL report.
    rt_parser = subparsers.add_parser(
        "roundtrip",
        help="Validate a calib end-to-end: calibrate → apply-calib → recalibrate → report")
    rt_parser.add_argument("-c", "--config", dest="config_path", required=True,
                           help="cam-IMU run config YAML (ROS1 bag)")
    rt_parser.add_argument("--out", help="output dir (default results/roundtrip/<name>)")
    rt_parser.add_argument("--bag", help="round-trip from this ROS1 bag, overriding the config "
                           "source (lets a euroc forward config round-trip)")
    rt_parser.add_argument("--keep-corrected", action="store_true",
                           help="keep the corrected bag (deleted by default)")
    rt_parser.add_argument("--keep-bias", action="store_true")

    # Unified runner: dispatch on the config's top-level `mode:` (cam | cam-imu).
    run_parser = subparsers.add_parser(
        "run", help="Run calibration; dispatch on the config's mode: (cam | cam-imu)")
    run_parser.add_argument(
        "--config", "-c", required=True, help="Path to run YAML config")

    # Agent-facing tools: deterministic, --json output, exit 0 ok / 1 warn / 2 fail.
    insp_parser = subparsers.add_parser(
        "inspect", help="Inspect a dataset (ROS1 bag, EuRoC mav0, ROS2, image folder)")
    insp_parser.add_argument("path", help="dataset path")
    insp_parser.add_argument("--json", action="store_true", help="machine-readable output")
    insp_parser.add_argument("--out", help="also write the JSON report here")

    chk_parser = subparsers.add_parser(
        "check", help="Lint the environment and/or a run config before solving")
    chk_parser.add_argument("-c", "--config", help="run config YAML to lint")
    chk_parser.add_argument("--env", action="store_true",
                            help="verify the installation (with -c: extras that config needs)")
    chk_parser.add_argument("--smoke", action="store_true",
                            help="also run a short synthetic camera solve (implies --env)")
    chk_parser.add_argument("--json", action="store_true")
    chk_parser.add_argument("--out", help="also write the JSON findings here (e.g. 00_env.json)")

    vd_parser = subparsers.add_parser(
        "verdict", help="Grade a finished result GREEN / YELLOW / RED")
    vd_parser.add_argument("result", help="camera-only result YAML or cam-IMU summary JSON")
    vd_parser.add_argument("-c", "--config",
                           help="cam-IMU run config (its `reference:` block is compared)")
    vd_parser.add_argument("--independent-check", default="none",
                           choices=["none", "roundtrip", "holdout", "repeatability", "cross-tool"],
                           help="independent verification already performed")
    vd_parser.add_argument("--json", action="store_true")
    vd_parser.add_argument("--out", help="also write the JSON verdict here")

    # Datasets command (catalog download / verify / list)
    from mvis.datasets.cli import add_datasets_subparser
    add_datasets_subparser(subparsers)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    from mvis.cli.blas_threads import describe as _describe_blas
    logger.info(_describe_blas())    # pinned in mvis.cli.__init__, before numpy loaded
    from mvis.core.eigen_cache import fix_eigen_cache_sizes, describe as _describe_eigen
    fix_eigen_cache_sizes()          # before the first GTSAM solve: mvis.core.eigen_cache
    logger.info(_describe_eigen())

    if args.command == "calibrate":
        config = CalibrationConfig.from_yaml(args.config)
        calibrate(config)
    elif args.command == "cam":
        calibrate_cam(args.config, args.output)
    elif args.command == "cam-imu":
        import os
        from mvis.core.config import CamImuRunConfig
        config = CamImuRunConfig.from_yaml(args.config)
        try:    # raw YAML text -> verbatim (copyable) Configuration report section
            config._source_yaml = open(args.config).read()
        except OSError:
            pass
        if args.bag:
            _override_source(config, args.bag)
        if config.input.bags:           # bag SET -> sweep, one discoverable dir per bag
            out_dir = os.path.dirname(config.output.path) or "results/sweep"
            results = calibrate_cam_imu_batch(config, out_dir)
            print(json.dumps({"out_dir": out_dir,
                              "bags": {k: v.get("ok") for k, v in results.items()}}, indent=1))
        else:
            summary = calibrate_cam_imu(config)
            print(json.dumps(summary, indent=1))
    elif args.command == "apply-calib":
        import os
        import yaml
        from mvis.recalib.calib_spec import load_rig_calib
        from mvis.recalib.apply import apply_calibration
        from mvis.recalib.sensor_map import MappingDeriveError, resolve_mapping

        # 1. Load config file if given; CLI flags override
        cfg = {}
        if args.config_path:
            from mvis.core.config_parser import load_yaml
            cfg = load_yaml(args.config_path)   # expands ${MVIS_DATASETS} and ~

        calib_path = args.calib or cfg.get("calib")
        in_bag     = args.in_bag or cfg.get("input")
        out_bag    = args.out_bag or cfg.get("output")
        map_path   = args.map_path  # CLI only; config uses mapping+base_imu_topic inline

        # --keep-bias inverts remove_bias; only override config if flag was explicitly given
        if args.keep_bias:
            remove_bias = False
        else:
            remove_bias = cfg.get("remove_bias", True)

        undistort = args.undistort or cfg.get("undistort", False)
        # --no-* flags switch an operation off; otherwise the config decides (default on)
        imu_intrinsics = False if args.no_imu_intrinsics else cfg.get("imu_intrinsics", True)
        td = False if args.no_td else cfg.get("td", True)
        t_ga = False if args.no_t_ga else cfg.get("t_ga", True)
        strict = args.strict or cfg.get("strict", False)

        # Validate required fields
        if not calib_path:
            parser.error("apply-calib: --calib / config 'calib:' is required")
        if not in_bag:
            parser.error("apply-calib: --in / config 'input:' is required")

        # Default output
        if not out_bag:
            base, _ = os.path.splitext(in_bag)
            out_bag = base + ".corrected.bag"

        rig = load_rig_calib(calib_path)
        rig._src_path = calib_path

        # Resolve mapping: --map file > config inline mapping > auto-derive
        try:
            base_imu_topic, mapping = resolve_mapping(rig, map_path, cfg)
        except MappingDeriveError as e:
            raise SystemExit(
                f"apply-calib: cannot auto-derive mapping: {e}\n"
                "Pass --map or re-export the calib with the current writer."
            ) from e

        summary = apply_calibration(
            in_bag, out_bag, rig, mapping, base_imu_topic,
            remove_bias=remove_bias, undistort=undistort, imu_intrinsics=imu_intrinsics,
            td=td, t_ga=t_ga, strict=strict)
        print(f"wrote {summary['out_bag']} "
              f"({summary['n_imu']} imu, {summary['n_cam']} cam, "
              f"{summary['n_pass']} passthrough); provenance {summary['provenance']}")
    elif args.command == "roundtrip":
        import sys
        from mvis.recalib.roundtrip import run_roundtrip
        try:
            rep = run_roundtrip(args.config_path, out=args.out, bag_override=args.bag,
                                keep_corrected=args.keep_corrected, keep_bias=args.keep_bias)
        except ValueError as e:
            parser.error(str(e))
        sys.exit(0 if rep["ok"] else 1)
    elif args.command == "run":
        _run_dispatch(args.config)
    elif args.command == "datasets":
        raise SystemExit(args.func(args))
    elif args.command in ("inspect", "check", "verdict"):
        raise SystemExit(_agent_tool(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
