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
"""The camera-only calibration driver behind ``scripts/run_calibration.py`` and the
camera-only regression rows (``cam_stats.build_command`` runs
``python -m mvis.calibration.cam_run``).

``run_cam(config, ...)`` loads the config, detects (with the detection cache in the
output folder), solves with CalibrationPipeline and writes ``<dataset>_<collection>``
``_calib.yaml``, ``_full.yaml``, ``.log`` and ``.html`` into the output folder, plus the
run attached to ``detections.mvisdb``. The output folder is cleared first, except the
detection caches, thumbnails and logs.

``mvis cam`` / ``mvis run`` (mode: cam) run ``run_cam`` too, with
``result_file`` set to the config's ``output_file``.

Usage:
    python -m mvis.calibration.cam_run --config examples/configs/example_cam_run.yaml -o results/x
    python scripts/run_calibration.py --config examples/configs/example_cam_run.yaml -o results/x
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import gtsam

from mvis.detection.board_factory import create_board
from mvis.calibration.pipeline_cam import CalibrationPipeline, PipelineConfig
from mvis.core.config import CamRunConfig
from mvis.io.output_writer import OutputWriter
from mvis.analysis.report import print_report
from mvis.analysis.generate_report import generate_calibration_report

logger = logging.getLogger("ablation")

# Written into every run folder run_cam creates; an existing folder is cleared
# only when it carries this marker, a detection cache, or nothing but MVIS
# outputs (see clean_output_dir).
OUTPUT_MARKER = ".mvis_output"
_CACHE_NAMES = frozenset({"detections.json", "detections.mvisdb"})
_PRESERVE_NAMES = _CACHE_NAMES | {"thumbnails", OUTPUT_MARKER}


class OutputDirRefused(FileExistsError):
    """The run folder exists and MVIS did not create it (see clean_output_dir)."""


def configure_process() -> None:
    """Process-wide setup of the driver: INFO logging to stderr, and OpenCV threading
    off (it conflicts with Rerun/tqdm). Called by main() and on import of the script."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    cv2.setNumThreads(1)


def parse_config(config_path: str):
    """Parse config and extract camera mappings.

    Uses CamRunConfig.from_yaml (dual-shape): accepts BOTH the legacy flat shape
    (cameras:/optimizer:/output_file:) and the canonical sectioned shape
    (input:/board:/solve:/output:). CamRunConfig is read-compatible with the legacy
    CalibrationConfig surface (.cameras/.optimizer/.output_file via properties), so
    the rest of this script is unchanged.
    """
    config = CamRunConfig.from_yaml(config_path)

    # Build mappings from config
    topics = {}      # topic -> cam_id
    dist_models = {} # cam_id -> distortion_model
    cam_names = {}   # cam_id -> display name
    bag_path = None

    for cam in config.cameras:
        cid = cam.camera_id
        cam_names[cid] = cam.name if cam.name else f"cam{cid}"
        dist_models[cid] = cam.distortion_model
        if cam.topic:
            topics[cam.topic] = cid
        if cam.source_path and bag_path is None:
            bag_path = cam.source_path

    camera_ids = sorted(cam_names.keys())
    return config, bag_path, topics, dist_models, cam_names, camera_ids


def force_global_shutter(cameras) -> None:
    """``--no-rs``: mark every camera config global shutter, in place, before the
    per-camera rate, the pipeline and the output YAML read it. The pipeline then
    estimates no readout and uses the global-shutter Stage-2 rate; the log records
    each camera whose configured shutter type the flag overrode."""
    from mvis.core import shutter
    for cam in cameras or []:
        if getattr(cam, "shutter_type", shutter.GLOBAL) != shutter.GLOBAL:
            logger.info(f"--no-rs: camera {getattr(cam, 'camera_id', '?')} "
                        f"shutter_type {cam.shutter_type!r} -> 'global'")
        cam.shutter_type = shutter.GLOBAL


_RUN_OUTPUT_SUFFIXES = (".html", ".log", "_calib.yaml", "_full.yaml")


def _is_mvis_output(item: Path, run_id: str | None) -> bool:
    """True iff ``item`` (a run-folder entry) is something run_cam writes for
    ``run_id`` (``<dataset>_<collection>``): its report, log and YAMLs (also the
    ``_pass1`` ones), the caches, thumbnails and the I5 export-check view."""
    if item.name in _PRESERVE_NAMES:
        return True
    if item.suffixes[-2:] == [".mvisdb", ".lock"]:
        return True
    if item.is_dir() or not run_id:
        return False
    if item.name.startswith("detections.mvisdb.export_check_"):
        return True
    return item.name.startswith(run_id) and item.name.endswith(_RUN_OUTPUT_SUFFIXES)


def _is_protected_dir(path: Path) -> bool:
    """The working directory, the home directory, or any parent of either."""
    p = path.resolve()
    return any(p == d or d.is_relative_to(p) for d in (Path.cwd().resolve(),
                                                         Path.home().resolve()))


def clean_output_dir(output_base, run_id: str | None = None) -> bool:
    """Clear the stale results of a run folder that MVIS created, keeping the
    detection caches, thumbnails and logs. The folder itself is never removed.
    Returns True when the folder is MVIS's to use (and the caller may mark it).

    A folder counts as MVIS-created when it carries :data:`OUTPUT_MARKER`, a
    detection cache, or nothing but run_cam's outputs for ``run_id`` (an empty
    folder qualifies). Any other existing folder raises :class:`OutputDirRefused`
    so ``-o <unrelated dir>`` cannot wipe it. The working directory, the home
    directory and their parents are never cleared or marked: a warning says so,
    the stale results stay, and False is returned."""
    import shutil
    base = Path(output_base)
    if not base.exists():
        return True
    if not base.is_dir():
        raise OutputDirRefused(f"output path {base} exists and is not a directory")
    if _is_protected_dir(base):
        logger.warning(f"Not clearing {base}: it is the working or home directory "
                       f"(or a parent of one); stale results there are kept")
        return False
    entries = list(base.iterdir())
    owned = ((base / OUTPUT_MARKER).exists()
             or any((base / n).exists() for n in _CACHE_NAMES)
             or all(_is_mvis_output(e, run_id) for e in entries))
    if not owned:
        raise OutputDirRefused(
            f"output directory {base} exists and was not created by MVIS "
            f"(no {OUTPUT_MARKER} marker or detection cache, and it holds files MVIS "
            f"does not write); choose another --output or remove it yourself")
    # Preserve caches + thumbnails + logs, delete only calibration results
    # (BOTH cache formats: the I2 gate found the old allowlist deleting the
    # .mvisdb every run, making cache hits impossible).
    for item in entries:
        if (item.name in _PRESERVE_NAMES or item.suffix == ".log"
                or item.suffixes[-2:] == [".mvisdb", ".lock"]):
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    return True


def run_calibration_with_ref(all_frames, image_sizes, detection_histories, board, ref_cam,
                              cam_names, dist_models, max_init_frames=0,
                              enable_rs=True, enable_td=True,
                              rerun_outliers=False,
                              camera_configs=None,
                              corner_subsample="all",
                              target_hz=5.0,
                              rerun_viz=False,
                              detection_images=None,
                              init_frame_selection="uniform",
                              calib_config=None,
                              detection_db=None):
    """Run calibration using CalibrationPipeline for a given reference camera.

    ``enable_rs=False`` (``--no-rs``) marks every camera global shutter before the
    pipeline reads the configs, so no rolling-shutter readout is estimated."""
    t0 = time.time()

    if not enable_rs:
        # run_cam passes the same CameraConfig objects as camera_configs and
        # calib_config.cameras; a direct caller may pass either.
        cams = list(camera_configs or [])
        cams += [c for c in (getattr(calib_config, "cameras", None) or [])
                 if not any(c is k for k in cams)]
        force_global_shutter(cams)

    # Subsample every camera to the same target rate. Trajectory alignment
    # (Stage 3) uses the full-rate data for accuracy.
    from mvis.io.frame_selector import subsample_frames_by_hz
    from mvis.calibration.selection_ledger import SelectionLedger, mark_frames
    ledger = SelectionLedger(db=detection_db)   # write-through when carried
    frames_sub = subsample_frames_by_hz(all_frames, target_hz=target_hz)
    if detection_db is not None:
        # M1c flip (2c-4): the drop is recorded as db flags at the moment of
        # selection; frames_sub is re-derived from the flags (same objects,
        # same order) so the flags are the authority, not a shadow.
        kept = {cid: {f.timestamp for f in fr} for cid, fr in frames_sub.items()}
        mark_frames(detection_db, kept, "driver", "hz_subsample")
        flag_sub = {}
        for cid, fr in all_frames.items():
            sel = {f.timestamp for f in detection_db.camera(cid).frames
                   if f.is_selected_to_use}
            flag_sub[cid] = [f for f in fr if f.timestamp in sel]
        assert {c: [f.timestamp for f in v] for c, v in flag_sub.items()} ==                {c: [f.timestamp for f in v] for c, v in frames_sub.items()}
        frames_sub = flag_sub
    else:
        ledger.diff(all_frames, frames_sub, "driver", "hz_subsample")

    # Diagnostic: MVIS_FROZEN_INTRINSICS="0,1" freezes cam 0 and 1
    frozen_intr = None
    frozen_env = os.environ.get("MVIS_FROZEN_INTRINSICS")
    if frozen_env:
        frozen_intr = {int(c.strip()) for c in frozen_env.split(",") if c.strip()}
        logger.info(f"Diagnostic: freezing intrinsics for cameras {frozen_intr}")

    # Diagnostic: MVIS_FROZEN_RS="0" freezes RS at Stage 2 value for cam 0
    freeze_rs = None
    freeze_rs_env = os.environ.get("MVIS_FROZEN_RS")
    if freeze_rs_env:
        freeze_rs = {int(c.strip()) for c in freeze_rs_env.split(",") if c.strip()}
        logger.info(f"Diagnostic: freezing RS at Stage 2 value for cameras {freeze_rs}")

    # Compute smart Stage 2 target_hz per camera:
    # GS cameras: 5 Hz (intrinsics don't need high frame rate)
    # RS cameras: 10 Hz (RS readout needs reasonable SLERP accuracy)
    stage2_hz = {}
    for cc in (camera_configs or []):
        cid = getattr(cc, 'camera_id', None)
        if cid is not None:
            is_rs = getattr(cc, 'shutter_type', 'global') == 'rolling'
            stage2_hz[cid] = 10.0 if is_rs else 5.0

    if calib_config is not None:
        # Single converter: optimizer numerics from config.optimizer.* (no
        # hardcodes); runtime/derived fields passed as overrides.
        config = PipelineConfig.from_calibration_config(
            calib_config,
            reference_camera=ref_cam,
            optimize_time_offset=enable_td,
            rerun_after_outlier_rejection=rerun_outliers,
            corner_subsample=corner_subsample,
            distortion_models=dist_models,
            frozen_intrinsics_cameras=frozen_intr,
            freeze_rs_cameras=freeze_rs,
            stage2_target_hz=stage2_hz if stage2_hz else None,
            rerun_visualization=rerun_viz,
        )
    else:
        # Back-compat: callers without a CalibrationConfig get the legacy defaults.
        config = PipelineConfig(
            max_init_frames=max_init_frames,
            init_frame_selection=init_frame_selection,
            max_iterations=15,
            max_iterations_single=15,
            pixel_sigma=1.0,
            huber_threshold=1.345,
            optimize_intrinsics=True,
            optimize_time_offset=enable_td,
            reference_camera=ref_cam,
            distortion_models=dist_models,
            verbose=True,
            rerun_after_outlier_rejection=rerun_outliers,
            corner_subsample=corner_subsample,
            camera_configs=camera_configs or [],
            rerun_visualization=rerun_viz,
            frozen_intrinsics_cameras=frozen_intr,
            freeze_rs_cameras=freeze_rs,
            stage2_target_hz=stage2_hz if stage2_hz else None,
        )

    pipeline = CalibrationPipeline(board, config, ledger=ledger)
    if detection_images:
        pipeline.set_detection_images(detection_images)
    # batch-B retirement: with the carrier, run() materializes the alignment
    # set from the db itself — the parameter is legacy-JSON-only now.
    result = pipeline.run(
        frames_sub, image_sizes,
        all_frames_full_rate=None if detection_db is not None else all_frames,
        db=detection_db)

    elapsed = time.time() - t0

    if result is None:
        return None, None, None

    # Build GTSAM calibration objects for HTML report (radtan / equidistant
    # only; other models have no gtsam Cal3 and are left out, not faked)
    calibrations = {}
    for cam in result.cameras:
        cid = cam.camera_id
        fx, fy, cx, cy = cam.intrinsics
        dist = cam.distortion
        if cam.distortion_model == "radtan":
            calibrations[cid] = gtsam.Cal3DS2(fx, fy, 0, cx, cy, dist[0], dist[1], dist[2], dist[3])
        elif cam.distortion_model == "equidistant":
            calibrations[cid] = gtsam.Cal3Fisheye(fx, fy, 0, cx, cy, dist[0], dist[1], dist[2], dist[3])

    # Pull the per-stage timings the pipeline recorded; fall back to a
    # single "total" bucket if the pipeline didn't run (shouldn't happen).
    pipeline_timings = {}
    try:
        pipeline_timings = {
            s["name"]: s["seconds"]
            for s in (result.timings or {}).get("sections", [])
        }
    except Exception:
        pass
    if not pipeline_timings:
        pipeline_timings = {"total": elapsed}

    extras = {
        "all_frames": all_frames, "board": board, "db": detection_db,
        "calibrations_gtsam": calibrations, "image_sizes": image_sizes,
        "detection_histories": detection_histories,
        "timings": pipeline_timings,
        "pipeline": pipeline,
    }
    return result, elapsed, extras


def build_parser() -> argparse.ArgumentParser:
    """The CLI of scripts/run_calibration.py (and python -m mvis.calibration.cam_run)."""
    parser = argparse.ArgumentParser(description="Multi-camera ablation test")
    parser.add_argument("--config", "-c", default="examples/configs/example_cam_run.yaml",
                        help="Path to calibration config YAML")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: from config output_file parent)")
    parser.add_argument("--no-rs", action="store_true",
                        help="Disable rolling shutter detection and optimization")
    parser.add_argument("--no-td", action="store_true",
                        help="Disable time offset optimization between cameras")
    parser.add_argument("--rerun-outliers", action="store_true",
                        help="After Stage 4, filter outlier frames + observations "
                             "and rerun warm-started.")
    parser.add_argument("--rerun-viz", action="store_true",
                        help="Launch Rerun viewer after calibration to display "
                             "board poses, camera trajectory, and detection images.")
    parser.add_argument("--corner-subsample", default=None,
                        choices=["all", "top_left", "tl", "bl", "br", "tr",
                                 "bottom_left", "bottom_right", "top_right",
                                 "two_diag", "two_anti"],
                        help="Per-tag corner subsampling for AprilGrid "
                             "(default: from config corner_subsample, else two_anti).")
    parser.add_argument("--dataset-name", default=None,
                        help="Dataset name for output artefacts "
                             "(default: parent dir of config path).")
    parser.add_argument("--collection-name", default=None,
                        help="Collection name for output artefacts "
                             "(default: config filename stem).")
    parser.add_argument("--target-hz", type=float, default=5.0,
                        help="Uniform subsample rate applied to every camera "
                             "for optimization (default: 5.0).")
    parser.add_argument("--result-file", default=None,
                        help="Also write the result to this file, in the config's "
                             "output_format (mvis cam passes the config's output_file).")
    return parser


def run_cam(config_path, output=None, *, no_rs=False, no_td=False, rerun_outliers=False,
            rerun_viz=False, corner_subsample=None, dataset_name=None, collection_name=None,
            target_hz=5.0, result_file=None) -> int:
    """Run one camera-only calibration; return the exit code (0 ok, 1 solve failed).

    The keyword arguments are the CLI flags of build_parser() (``--no-rs`` -> no_rs, ...),
    with the same defaults. ``output`` defaults to ``<config output_file parent>/ablation``;
    ``dataset_name`` / ``collection_name`` default to the config's parent folder name and
    file stem. ``result_file``, when given, also gets the result in the config's
    ``output_format`` (the file ``mvis cam`` users and ``camera_input:`` read). Raises SystemExit(3) when the mvisdb export view does not reproduce the
    calib YAML (the I5 shadow)."""

    # Parse config
    config, bag_path, topics, dist_models, cam_names, camera_ids = parse_config(config_path)
    if no_rs:
        # --no-rs: global shutter everywhere (per-camera rate, pipeline, output YAML)
        force_global_shutter(config.cameras)

    # Build per-camera target_hz.
    # Priority: config per-camera value > CLI --target-hz > smart default.
    # Smart default: RS=10Hz, GS=5Hz, ref+td=max(base, 10Hz).
    # Note: ref_cam and enable_td are resolved later; smart defaults that
    # depend on them are applied after ref_cam is known (see below).
    per_cam_hz = {}
    for cam in config.cameras:
        cam_rate = getattr(cam, "factor_hz", None)
        if cam_rate is not None and cam_rate > 0:
            per_cam_hz[cam.camera_id] = cam_rate
        elif target_hz != 5.0:
            per_cam_hz[cam.camera_id] = target_hz
        else:
            # Smart default based on shutter type
            is_rs = getattr(cam, 'shutter_type', 'global') == 'rolling'
            per_cam_hz[cam.camera_id] = 10.0 if is_rs else 5.0

    # Derive dataset/collection from config path if not explicitly given.
    cfg_path = Path(config_path)
    dataset_name = dataset_name or cfg_path.parent.name
    collection_name = collection_name or cfg_path.stem
    run_id = f"{dataset_name}_{collection_name}"

    # Determine output directory
    if output:
        output_base = output
    else:
        output_base = str(Path(config.output_file).parent / "ablation")

    # Create board from config
    board = create_board(config.board)

    # Clean old calibration results of a run folder MVIS created (keeps the
    # detection caches); refuses any other existing folder, never rmtree's it.
    mark_output = clean_output_dir(output_base, run_id)

    cache_path = f"{output_base}/detections.json"
    Path(output_base).mkdir(parents=True, exist_ok=True)
    if mark_output:     # never claim the working / home directory
        (Path(output_base) / OUTPUT_MARKER).write_text(
            "run folder written by mvis.calibration.cam_run; cleared on the next run\n")

    # Resolve reference camera early (needed for detection diversity filter)
    ref_cam = config.reference_camera if config.reference_camera in camera_ids \
              else camera_ids[0]
    enable_td = not no_td and getattr(config, "optimize_time_offset", True)

    # Apply ref-camera boost: ref cam needs ≥10 Hz for time-offset calibration
    if enable_td and ref_cam in per_cam_hz:
        per_cam_hz[ref_cam] = max(per_cam_hz[ref_cam], 10.0)
    logger.info(f"Per-camera target_hz: {per_cam_hz}")

    logger.info(f"Config: {config_path}")
    logger.info(f"Bag: {bag_path}")
    logger.info(f"Cameras: {cam_names}")
    logger.info(f"Output: {output_base}")

    # Unified data loader — dispatches to rosbag, euroc, or image paths.
    from mvis.io.data_loader import DataLoader
    loader = DataLoader(config)

    # Streaming detection: reads images one at a time, detects, discards.
    # On cache hit, no images are loaded at all.
    # All subsampling (target_hz, diversity) happens post-detection.
    from mvis.io.detection_pipeline import DetectionPipeline

    thumb_dir = Path(cache_path).parent / "thumbnails" if cache_path else None
    store_dir = getattr(getattr(config, "input", None), "cache_dir", None)
    stream_store = None
    if store_dir:
        from mvis.io.stream_store import StreamDetectionStore
        stream_store = StreamDetectionStore(store_dir)
    det_pipeline = DetectionPipeline(
        board, thumb_dir=thumb_dir, cache_path=cache_path,
        stream_store=stream_store)

    logger.info("Loading + detecting...")
    det_result = det_pipeline.run_streaming(
        loader, cam_names=cam_names, cache_key=loader.cache_key,
        ref_cam=ref_cam, enable_td=enable_td,
        target_hz=per_cam_hz)

    all_frames = det_result.all_frames
    image_sizes = det_result.image_sizes
    detection_histories = det_result.detection_histories
    detection_db = det_result.db          # carrier (2c-3); may be None
    # Filter None entries for report compatibility
    all_detection_images = {
        cid: [img for img in imgs if img is not None]
        for cid, imgs in det_result.detection_images.items()
    }

    import gc
    gc.collect()

    # Step 3: Log memory usage
    try:
        import psutil
        mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
        logger.info(f"Memory after loading: {mem_mb:.0f} MB")
    except ImportError:
        pass

    # Log video info
    for cid in camera_ids:
        n = len(all_frames.get(cid, []))
        if n > 0:
            ts = [f.timestamp for f in all_frames[cid]]
            dur = ts[-1] - ts[0] if len(ts) > 1 else 0
            hz = (n - 1) / dur if dur > 0 else 0
            logger.info(f"  {cam_names[cid]}: {n} frames, {dur:.1f}s, {hz:.1f}Hz effective")

    logger.info("=" * 60)
    logger.info(f"Calibration: {run_id} — ref={cam_names[ref_cam]} (cam{ref_cam})")
    logger.info("=" * 60)

    # Resolve options: CLI flag overrides the config value (camera-only canonical).
    corner_mode = corner_subsample or getattr(config, "corner_subsample", "two_anti")
    from mvis.detection.charuco_board import CharucoBoard
    if isinstance(board, CharucoBoard):
        corner_mode = "all"       # ChArUco ids are chessboard corners: nothing to thin
    do_rerun_outliers = rerun_outliers or getattr(config, "rerun_outliers", False)
    do_rerun_viz = rerun_viz or getattr(config, "rerun_visualization", False)
    appendix_max_frames = getattr(getattr(config, "report", None), "appendix_max_frames", 0)

    result, elapsed, extras = run_calibration_with_ref(
        all_frames, image_sizes, detection_histories, board, ref_cam,
        cam_names, dist_models, max_init_frames=config.optimizer.max_init_frames,
        enable_rs=not no_rs, enable_td=enable_td,
        rerun_outliers=do_rerun_outliers,
        corner_subsample=corner_mode,
        target_hz=per_cam_hz,
        camera_configs=config.cameras,
        rerun_viz=do_rerun_viz,
        detection_images=all_detection_images,
        init_frame_selection=config.optimizer.init_frame_selection,
        calib_config=config,
        detection_db=detection_db)

    if result is None:
        logger.error(f"Calibration failed for {run_id}")
        return 1

    # Outputs live directly under output_base, named <dataset>_<collection>.
    out_dir = output_base
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    html_path = f"{out_dir}/{run_id}.html"

    # File 1: calibration-only YAML (deployable, minimal).
    calib_yaml_path = f"{out_dir}/{run_id}_calib.yaml"
    OutputWriter.write_calibration_yaml(result, calib_yaml_path)
    if result_file:
        Path(result_file).parent.mkdir(parents=True, exist_ok=True)
        OutputWriter.write(result, result_file, config.output_format)
        logger.info(f"Results also saved to {result_file}")

    # File 2: full replay YAML (board poses, 3D corners, timing, filter stats).
    full_yaml_path = f"{out_dir}/{run_id}_full.yaml"
    pipeline_obj = extras.get("pipeline")
    filter_stats = {}
    if pipeline_obj:
        if hasattr(pipeline_obj, "pose_filter_stats") and pipeline_obj.pose_filter_stats:
            pfs = pipeline_obj.pose_filter_stats
            filter_stats["pose_filter"] = {
                "ref_kept": pfs.n_ref_kept, "ref_dropped": pfs.n_ref_dropped,
                "threshold_tags": pfs.threshold_tags,
            }
        if hasattr(pipeline_obj, "corner_filter_stats") and pipeline_obj.corner_filter_stats:
            cfs = pipeline_obj.corner_filter_stats
            filter_stats["corner_filter"] = {
                "corners_dropped": cfs.n_corners_dropped,
                "min_views": cfs.min_views,
                "obs_in": cfs.obs_in, "obs_out": cfs.obs_out,
            }
    OutputWriter.write_full_yaml(
        result, full_yaml_path,
        board=board,
        corner_subsample=corner_mode,
        config_path=config_path,
        detection_filter_stats=filter_stats or None,
    )

    # Attach this run + its estimates to the capture .mvisdb the
    # detection cache wrote, so reports can read load(path, run_id).
    # Additive: a failure here must never break the YAML/HTML outputs.
    mvisdb_path = Path(out_dir) / "detections.mvisdb"
    if mvisdb_path.exists():
        try:
            from mvis.mvisdb.estimates import estimates_from_calibration_result
            from mvis.mvisdb.manager import DatabaseManager
            from mvis.mvisdb.mvisdatabase import CalibChoice, Run
            _mgr = DatabaseManager()
            _db = extras.get("db")
            if _db is None:
                _db = _mgr.load(mvisdb_path)
            _db.run = Run(CalibChoice.CAM_ONLY, description=run_id, status="ok")
            _db.estimate_states = estimates_from_calibration_result(result)
            _pipe = extras.get("pipeline")
            if _pipe is not None and getattr(_pipe, "ledger", None) is not None:
                _n = _pipe.ledger.apply_to_db(
                    _db, solve_frames=extras.get("all_frames"))
                logger.info(f"Selection ledger: {_n} frames flagged in overlay")
            _rid = _mgr.save(_db, mvisdb_path)
            logger.info(f"Estimates attached to {mvisdb_path.name} as run {_rid}")
            # I5 shadow: the calib YAML must be reproducible as an export
            # VIEW of the run just attached — rendered from state(run_id)
            # and compared against the writer's file, ignoring the
            # documented lossy / derived fields (max error, image_size, so3;
            # see normalise_calib_yaml_pair — kb4 valid_domain is compared
            # strictly when the capture stores the stream resolution).
            try:
                import yaml as _yaml
                from mvis.mvisdb.exports import (
                    export_calib_yaml, normalise_calib_yaml_pair)
                _view_path = str(mvisdb_path) + f".export_check_{_rid}.yaml"
                export_calib_yaml(_mgr, mvisdb_path, _rid, _view_path)
                with open(calib_yaml_path) as _f1, open(_view_path) as _f2:
                    _a, _b = _yaml.safe_load(_f1), _yaml.safe_load(_f2)
                normalise_calib_yaml_pair(_a, _b)
                assert _a == _b, "I5 export view != writer calib YAML"
                os.unlink(_view_path)
                logger.info("I5 export view verified against writer YAML")
            except AssertionError:
                logger.error("I5 EXPORT VIEW MISMATCH — kept %s", _view_path)
                # SystemExit escapes the attach block's broad except: a view
                # that cannot reproduce the writer's YAML must fail the run
                # loudly during the shadow phase, not log-and-continue
                raise SystemExit(3)
        except Exception as e:
            logger.warning(f"mvisdb estimate attach skipped: {e}")

    report = print_report(result)
    with open(f"{out_dir}/{run_id}.log", "w") as f:
        f.write(f"Run: {run_id}\n")
        f.write(f"Reference camera: {ref_cam} ({cam_names[ref_cam]})\n")
        f.write(f"Corner subsample: {corner_mode}, target_hz={per_cam_hz}\n")
        f.write(f"Time: {elapsed:.1f}s\n\n")
        f.write("=== Timing Breakdown ===\n")
        for k, v in extras.get("timings", {}).items():
            f.write(f"  {k}: {v:.1f}s\n")
        f.write("\n=== Video Information ===\n")
        for cid in camera_ids:
            frames_list = all_frames.get(cid, [])
            if frames_list:
                ts = [fr.timestamp for fr in frames_list]
                dur = ts[-1] - ts[0] if len(ts) > 1 else 0
                hz = (len(ts) - 1) / dur if dur > 0 else 0
                f.write(f"  {cam_names[cid]}: {len(frames_list)} frames, "
                        f"duration={dur:.1f}s, effective={hz:.1f}Hz\n")
        f.write("\n")
        f.write(report)

    # Build calibration_info for the config section in the report
    topic_by_cam = {cid: topic for topic, cid in topics.items()}
    cam_cfgs = {c.camera_id: c for c in config.cameras}
    cameras_info = {}
    for cid in camera_ids:
        cc = cam_cfgs.get(cid)
        cameras_info[cid] = {
            "name": cam_names.get(cid, ""),
            "source_type": cc.source_type if cc else "",
            "source_path": cc.source_path if cc else "",
            "topic": topic_by_cam.get(cid, ""),
            "distortion_model": dist_models.get(cid, ""),
        }
    calibration_info = {
        "config_path": config_path,
        "reference_camera": ref_cam,
        "output_report_path": html_path,
        "calibration_yaml_path": full_yaml_path,
        "cameras_info": cameras_info,
    }

    # If rerun was enabled, also write the pass-1 result + a pass-1 report
    pipeline_obj = extras.get("pipeline")
    if rerun_outliers and pipeline_obj is not None \
            and getattr(pipeline_obj, "first_pass_result", None) is not None \
            and pipeline_obj.first_pass_result is not result:
        first = pipeline_obj.first_pass_result
        first_frames = pipeline_obj.first_pass_frames
        pass1_full_yaml = f"{out_dir}/{run_id}_pass1_full.yaml"
        pass1_html = f"{out_dir}/{run_id}_pass1.html"
        OutputWriter.write_calibration_yaml(first, f"{out_dir}/{run_id}_pass1_calib.yaml")
        OutputWriter.write_full_yaml(first, pass1_full_yaml,
                                      board=board, corner_subsample=corner_mode,
                                      config_path=config_path)
        ci_pass1 = dict(calibration_info)
        ci_pass1["output_report_path"] = pass1_html
        ci_pass1["calibration_yaml_path"] = pass1_full_yaml
        ci_pass1["result_files"] = [f"{out_dir}/{run_id}_pass1_calib.yaml", pass1_full_yaml]
        generate_calibration_report(
            result=first, all_frames=first_frames, board=board,
            calibrations=extras["calibrations_gtsam"],
            image_sizes=extras["image_sizes"],
            title=f"{dataset_name} / {collection_name} (pass 1, pre-outlier)",
            output_path=pass1_html,
            calibration_info=ci_pass1,
            sample_detection_images={cid: imgs[:5] for cid, imgs
                                     in all_detection_images.items()},
            all_detection_images=all_detection_images,
            thumb_dir=thumb_dir,
            appendix_max_frames=appendix_max_frames,
        )

    # Stash corner_subsample mode so the report's 3D plot can colour-code.
    calibration_info_full = dict(calibration_info)
    calibration_info_full["corner_subsample"] = corner_mode
    calibration_info_full["result_files"] = [calib_yaml_path, full_yaml_path]
    generate_calibration_report(
        result=result, all_frames=extras["all_frames"], board=board,
        calibrations=extras["calibrations_gtsam"], image_sizes=extras["image_sizes"],
        title=f"{dataset_name} / {collection_name}", output_path=html_path,
        calibration_info=calibration_info_full,
        sample_detection_images={cid: imgs[:5] for cid, imgs
                                 in all_detection_images.items()},
        all_detection_images=all_detection_images,
        thumb_dir=thumb_dir,
        appendix_max_frames=appendix_max_frames,
    )

    logger.info(f"  time: {elapsed:.1f}s")
    for cam in result.cameras:
        logger.info(
            f"  {cam_names[cam.camera_id]}: fx={cam.fx:.2f} "
            f"RS={cam.rolling_shutter_readout:.6f}"
        )

    logger.info(f"\nResults saved to {out_dir}/ as {run_id}.{{yaml,html,log}}")
    return 0


def main(argv=None) -> int:
    configure_process()
    args = build_parser().parse_args(argv)
    # First GTSAM use of the process: Eigen's block sizes must not depend on
    # the core the solve starts on (hybrid CPUs), see mvis.core.eigen_cache.
    from mvis.core.eigen_cache import describe, fix_eigen_cache_sizes
    fix_eigen_cache_sizes()
    logger.info(describe())
    try:
        return run_cam(args.config, args.output, no_rs=args.no_rs, no_td=args.no_td,
                       rerun_outliers=args.rerun_outliers, rerun_viz=args.rerun_viz,
                       corner_subsample=args.corner_subsample, dataset_name=args.dataset_name,
                       collection_name=args.collection_name, target_hz=args.target_hz,
                       result_file=args.result_file)
    except OutputDirRefused as e:     # the run folder is not one MVIS created
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    sys.exit(main() or 0)
elif __name__ == "__mp_main__":
    # a spawned worker re-imports this module as __mp_main__ (python -m); the script
    # entry configured its workers the same way on import
    configure_process()
