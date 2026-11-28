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
"""Regenerate per-run HTML reports from existing files.

For each `<output>/<dataset>_<slug>.yaml` already on disk, rebuild the
HTML report using the current `html_report.py` logic — without re-
running calibration. Useful when the report template (plot functions,
section layout, etc.) changes and you want to refresh existing runs.

Inputs per run (must exist on disk):
    <output>/<dataset>_<slug>.yaml          # calibration result
    <output>/detections.json                # detection cache
    <output>/thumbnails/cam<N>/*.jpg        # detection thumbnails (optional)

Outputs:
    <output>/<dataset>_<slug>.html          # overwritten

When `<output>/thumbnails/` is missing (legacy runs from before
thumbnail persistence), Appendix A is skipped silently — every other
section still renders. Re-run calibration to populate the thumbnails.

Usage:
    python -m mvis.analysis.regenerate_html <run_dir> --config <run config>
    python -m mvis.analysis.regenerate_html <root> --config-root <dir of run configs>

`<root>` holds `<dataset>/<run>/` directories; `--config-root` is searched for a YAML
whose `<parent>_<stem>` prefixes each run id (the board comes from that config).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import gtsam
import numpy as np

from mvis.analysis.generate_report import generate_calibration_report
from mvis.detection.board_factory import create_board
from mvis.core.config import CalibrationConfig
from mvis.core.camera_models import is_removed_model, removed_model_message
from mvis.core.types import (
    CalibrationResult, CameraParams, ExtrinsicTransform, ReprojectionErrorStats,
)
from mvis.io.detection_cache import legacy_json_frames
from mvis.io.safe_yaml import load_yaml_file_safe

logger = logging.getLogger("regen_html")


def _apply_stored_board_poses(result, doc) -> bool:
    """The full YAML carries board_poses (output_writer.to_full_dict): use them
    instead of re-running PnP, so a regenerated report grades motion against the
    trajectory the run actually produced. Returns True if applied."""
    stored = (doc or {}).get("board_poses") or []
    if not stored:
        return False
    import gtsam
    result.board_poses, result.board_pose_times, result.board_poses_gtsam = [], [], []
    for e in stored:
        T = np.asarray(e["T_board_cam"], dtype=np.float64)
        result.board_poses.append(T)
        result.board_pose_times.append(float(e.get("timestamp", 0.0)))
        result.board_poses_gtsam.append(gtsam.Pose3(T))
    return True


def _apply_stored_board_shape(result, yaml_path: Path, board) -> bool:
    """A run that estimated the board shape wrote its per-corner heights (`board_shape.heights_m`);
    rebuild `result.board_points_refined` from them so the regenerated report scores corners against
    the fitted board, as the run's own report did. x / y are the nominal points (the solve pins them
    under a 1e-6 m prior). Returns True if applied."""
    doc = load_yaml_file_safe(yaml_path) or {}
    heights = (doc.get("board_shape") or {}).get("heights_m")
    if heights is None or len(heights) != board.num_points():
        return False
    pts = np.asarray(board.all_points_3d(), dtype=np.float64).copy()
    pts[:, 2] += np.asarray(heights, dtype=np.float64)
    result.board_points_refined = pts
    result.board_heights = np.asarray(heights, dtype=np.float64)
    n_obs = (doc.get("board_shape") or {}).get("n_obs")            # absent before 2026-11-12
    result.board_height_n_obs = (np.asarray(n_obs, int) if n_obs is not None and len(n_obs) == len(heights)
                                 else None)
    return True


def _restore_snap_sync(result) -> None:
    """The solve snaps a non-ref frame to a ref pose within `pair_sync_tolerance_s` when it estimates
    no time offset (Optimizer: snap_tol = 0 if use_dt else the tolerance) and hands the value on as
    `result.snap_sync_tolerance_s`; neither the YAML nor the db stores it. Without it a synced
    second camera (T265 fisheye2, ~2.5 ms after fisheye1) is projected part-way to the next ref
    pose, 1/target_hz away. A stored time offset marks a td solve, which interpolates."""
    from mvis.calibration.pipeline_cam import PipelineConfig
    result.snap_sync_tolerance_s = (0.0 if result.time_offsets
                                    else PipelineConfig.pair_sync_tolerance_s)


def _result_from_yaml(path: Path) -> CalibrationResult:
    """Reconstruct a CalibrationResult from a serialized YAML."""
    d = load_yaml_file_safe(path) or {}

    cams = []
    for c in d.get("cameras", []):
        intr = c.get("intrinsics", {})
        dist = c.get("distortion", {})
        if is_removed_model(dist.get("model")):
            raise ValueError(removed_model_message(dist["model"]))
        cams.append(CameraParams(
            camera_id=c.get("id"),
            intrinsics=np.array([intr.get("fx", 0), intr.get("fy", 0),
                                  intr.get("cx", 0), intr.get("cy", 0)]),
            distortion=np.array(dist.get("coeffs", []) or [0, 0, 0, 0]),
            distortion_model=dist.get("model", "radtan"),
            image_size=tuple(c.get("image_size", [0, 0])),
            rolling_shutter_readout=float(c.get("rolling_shutter_readout", 0.0)),
        ))

    extrinsics = []
    for e in (d.get("extrinsics") or {}).get("transforms", []):
        T = np.asarray(e.get("T_to_from", np.eye(4)), dtype=np.float64)
        extrinsics.append(ExtrinsicTransform(
            from_camera=e.get("from"),
            to_camera=e.get("to"),
            T_to_from=T,
            so3=np.asarray(e.get("so3", [0, 0, 0]), dtype=np.float64),
            translation=np.asarray(e.get("translation", [0, 0, 0]),
                                    dtype=np.float64),
        ))

    time_offsets = {}
    for o in (d.get("time_offsets") or {}).get("offsets", []):
        time_offsets[o.get("camera")] = float(o.get("dt", 0.0))

    rep_errors = []
    for er in (d.get("reprojection_errors") or {}).get("per_camera", []):
        rep_errors.append(ReprojectionErrorStats(
            camera_id=er.get("camera"),
            rms=float(er.get("rms", 0)),
            mean=float(er.get("mean", 0)),
            median=float(er.get("median", 0)),
            p90=float(er.get("p90", 0)),
            p95=float(er.get("p95", 0)),
            max=float(er.get("max", 0)),
            num_observations=int(er.get("num_observations", 0)),
            std=float(er.get("std", 0)),
            inlier=er.get("inlier"),
        ))

    result = CalibrationResult(
        cameras=cams,
        extrinsics=extrinsics,
        reference_camera=int((d.get("extrinsics") or {}).get(
            "reference_camera", 0)),
        time_offsets=time_offsets,
        reprojection_errors=rep_errors,
        overall_rms=float((d.get("reprojection_errors") or {}).get(
            "overall_rms", 0)),
    )
    result.timings = d.get("timings")
    result.corner_accounting = _accounting_from_doc(d)
    return result


def _accounting_from_doc(d: dict):
    """On disk `corner_accounting.per_camera` is a list; in memory {camera_id: dict}. None for a
    file written before it existed."""
    rows = (d.get("corner_accounting") or {}).get("per_camera")
    return {a["camera"]: a for a in rows} if rows else None


def _build_calibrations(result: CalibrationResult) -> dict:
    """Build per-camera GTSAM Cal3* objects from CalibrationResult. Only
    radtan / equidistant have one; other models are left out (consumers skip a
    camera with no entry) rather than faked as Cal3Fisheye."""
    out = {}
    for cam in result.cameras:
        fx, fy, cx, cy = cam.intrinsics
        d = cam.distortion
        coefs = list(d) + [0.0] * 4
        if cam.distortion_model == "radtan":
            out[cam.camera_id] = gtsam.Cal3DS2(
                fx, fy, 0, cx, cy, coefs[0], coefs[1], coefs[2], coefs[3])
        elif cam.distortion_model == "equidistant":
            out[cam.camera_id] = gtsam.Cal3Fisheye(
                fx, fy, 0, cx, cy, coefs[0], coefs[1], coefs[2], coefs[3])
        else:
            logger.debug("cam%s: no gtsam Cal3 for model %r; skipped",
                         cam.camera_id, cam.distortion_model)
    return out


def _frames_from_mvisdb(db_path: Path) -> dict:
    """Reload all_frames from the sqlite capture, the default detection cache
    (detections.json is the legacy fallback).

    The capture holds every detected frame; the latest solved run's overlay
    flags the ones the detection stage thinned away (`detection:*`). Those are
    left out, so the report scores the frames the solve started from, as the
    live report does — a thinned frame lies between ref poses 1/target_hz
    apart and is scored at an interpolated pose (mvis_t265 stereo: 11 px RMS
    for a 0.3 px solve). Frames a later stage dropped stay, as live. A capture
    with no solved run has no overlay and keeps every frame."""
    from mvis.mvisdb.bridge import frame_observations_from_db
    from mvis.mvisdb.manager import DatabaseManager
    mgr = DatabaseManager()
    db = mgr.load(db_path, mgr.latest_solved_run_id(db_path))
    out = {}
    for s in db.streams:
        thinned = {f.timestamp for f in s.frames
                   if (f.exclude_reason or "").startswith("detection:")}
        out[s.camera_id] = [fr for fr in frame_observations_from_db(db, s.camera_id)
                            if fr.timestamp not in thinned]
    return out


def _result_from_mvisdb(db_path: Path):
    """Latest solved run's estimates -> CalibrationResult, or None if the file
    carries no solved run (then the YAML path is authoritative)."""
    from mvis.mvisdb.estimates import calibration_result_from_estimates
    from mvis.mvisdb.manager import DatabaseManager
    mgr = DatabaseManager()
    es = mgr.latest_solved_state(db_path)
    if es is None:
        return None
    # image_size from the capture's streams: the estimate does not carry it,
    # and the report's kb4 valid-domain columns (and "Image Size") need it
    return calibration_result_from_estimates(
        es, image_sizes=mgr.stream_resolutions(db_path))


def _load_thumbnails(thumb_dir: Path) -> dict[int, list[np.ndarray]]:
    """Load per-camera thumbnail JPEGs."""
    out: dict[int, list[np.ndarray]] = {}
    if not thumb_dir.exists():
        return out
    for cam_dir in sorted(thumb_dir.glob("cam*")):
        try:
            cid = int(cam_dir.name[3:])
        except ValueError:
            continue
        imgs = []
        for jp in sorted(cam_dir.glob("*.jpg")):
            img = cv2.imread(str(jp))
            if img is not None:
                imgs.append(img)
        out[cid] = imgs
    return out


def _camera_sources(config_path) -> dict:
    """camera id -> {"source_path"} from the run config's input.cameras; {} when unreadable."""
    try:
        import yaml
        cfg = yaml.safe_load(open(config_path)) or {}
        return {int(c.get("camera_id", i)): {"source_path": c.get("source_path", "")}
                for i, c in enumerate((cfg.get("input") or {}).get("cameras") or [])}
    except Exception:
        return {}


def _make_title(run_id: str, config_path: Path | None) -> str:
    """Derive a readable 'dataset / collection' title from the run_id."""
    if config_path is not None:
        ds = config_path.parent.name
        if ds and run_id.startswith(ds + "_"):
            coll = run_id[len(ds) + 1:]
            return f"{ds} / {coll}"
    return run_id


def regenerate_one(out_dir: Path, config_path: Path | None = None,
                   config_root: Path | None = None) -> bool:
    """Regenerate the main + (if present) pass1 HTML for one run dir.

    `config_path` is the run's own config (board etc.); without it, `config_root`
    is searched for a YAML matching the run id."""
    main_yamls = [y for y in out_dir.glob("*.yaml")
                  if not y.stem.endswith("_pass1")
                  and not y.stem.endswith("_full")
                  and not y.stem.endswith("_calib")
                  and not y.stem.endswith("_pass1_full")
                  and not y.stem.endswith("_pass1_calib")
                  and not y.stem.endswith("_board_shape")]   # cam-IMU sidecar, not a result
    if not main_yamls:
        # Fall back to _full.yaml (new naming convention)
        full_yamls = [y for y in out_dir.glob("*_full.yaml")
                      if "_pass1_full" not in y.stem]
        if full_yamls:
            yaml_path = full_yamls[0]
            # Derive run_id by stripping _full suffix
            run_id = yaml_path.stem.removesuffix("_full")
        else:
            logger.info(f"  Skip {out_dir.name}: no calibration YAML")
            return False
    else:
        yaml_path = main_yamls[0]
        run_id = yaml_path.stem

    pass1_yaml = out_dir / f"{run_id}_pass1_full.yaml"
    if not pass1_yaml.exists():
        pass1_yaml = out_dir / f"{run_id}_pass1.yaml"
    targets = [(yaml_path, out_dir / f"{run_id}.html")]
    if pass1_yaml.exists():
        pass1_html = out_dir / f"{run_id}_pass1.html"
        targets.append((pass1_yaml, pass1_html))

    html_path = out_dir / f"{run_id}.html"

    logger.info(f"  Regen {run_id} ({len(targets)} HTMLs)...")
    try:
        result = _result_from_yaml(yaml_path)
    except Exception as e:
        logger.error(f"    parse YAML failed: {e}")
        return False

    # Find the source config to rebuild the board. The run_id is "<dataset>_<slug>";
    # without `config_path` the config is matched by run_id under `config_root`.
    candidates = ([] if config_path or config_root is None
                  else list(Path(config_root).rglob("*.yaml")))
    config = CalibrationConfig.from_yaml(str(config_path)) if config_path else None
    config_path_found: Path | None = config_path
    # Two-pass matching: first try the strict match (parent_name + stem
    # forms a prefix of run_id), then fall back to stem-only containment.
    for strict in (True, False):
        if config is not None:
            break
        for c in candidates:
            if "/gt/" in str(c) or "manifest" in c.name:
                continue
            if strict:
                if not run_id.startswith(f"{c.parent.name}_{c.stem}"):
                    continue
            else:
                if c.stem not in run_id:
                    continue
            try:
                config = CalibrationConfig.from_yaml(str(c))
                config_path_found = c
                break
            except Exception:
                continue
    if config is None:
        logger.error(f"    no source config found for {run_id}: pass --config <run config> "
                     "(one run dir) or --config-root <dir> (searched by run id)")
        return False

    board = create_board(config.board)

    # Reload detections: sqlite capture first (the default), legacy
    # detections.json as fallback.
    mvisdb_path = out_dir / "detections.mvisdb"
    cache_path = out_dir / "detections.json"
    all_frames = {}
    if mvisdb_path.exists():
        try:
            all_frames = _frames_from_mvisdb(mvisdb_path)
        except Exception as e:
            logger.warning(f"    mvisdb frame reload failed: {e}")
    if not all_frames:
        all_frames = legacy_json_frames(cache_path)
    if not all_frames:
        logger.error(f"    no detections.mvisdb/.json — can't reconstruct frames")
        return False

    image_sizes = {}
    for cid, frames in all_frames.items():
        if frames:
            image_sizes[cid] = frames[0].image_size

    calibrations = _build_calibrations(result)

    # The db run's estimates are the first authority for the main target —
    # YAML stays authoritative for pass1 and old dirs.
    if mvisdb_path.exists():
        try:
            db_result = _result_from_mvisdb(mvisdb_path)
            if db_result is not None and db_result.board_poses:
                # a writer-only diagnostic: the db has no count, the YAML does
                db_result.corner_accounting = result.corner_accounting
                result = db_result
                logger.info(
                    f"    result from mvisdb run ({len(result.board_poses)}"
                    " poses; YAML not consulted)")
                calibrations = _build_calibrations(result)
        except Exception as e:
            logger.warning(f"    mvisdb result read failed: {e}")
    # Prefer the board_poses the run itself wrote (full YAML carries them);
    # only fall back to PnP when regenerating from a calib-only YAML.
    _doc = load_yaml_file_safe(yaml_path) or {}
    if not result.board_poses and _apply_stored_board_poses(result, _doc):
        logger.info(f"    board_poses: {len(result.board_poses)} stored poses "
                    "(PnP re-run skipped)")
    # Recompute board poses via PnP so error-analysis plots and 3D
    # visualization work (only when the YAML carried none).
    from mvis.calibration.initializer import Initializer
    _init = Initializer(board)
    ref_cam = result.reference_camera
    for cid_idx, cam in (enumerate(result.cameras) if not result.board_poses else []):
        cid = cam.camera_id
        if cid not in all_frames or not all_frames[cid]:
            continue
        poses = _init.init_board_poses(
            all_frames[cid], cam.intrinsics, cam.distortion,
            cam.distortion_model)
        if cid == ref_cam:
            result.board_poses = []
            result.board_pose_times = []
            for i, (p, f) in enumerate(zip(poses, all_frames[cid])):
                if p is not None:
                    result.board_poses.append(p.matrix())
                    result.board_pose_times.append(f.timestamp)
                    result.board_poses_gtsam.append(p)
                else:
                    result.board_poses.append(None)
                    result.board_pose_times.append(f.timestamp)

    thumbs = _load_thumbnails(out_dir / "thumbnails")

    calibration_info = {
        "config_path": "(regenerated from disk)",
        "reference_camera": result.reference_camera,
        "output_report_path": str(html_path),
        "calibration_yaml_path": str(yaml_path),
        "result_files": [str(p) for p in sorted(
            {Path(yaml_path), *out_dir.glob("*_calib.yaml")})],
        "cameras_info": _camera_sources(config_path_found),   # names the recording in the report title
    }

    n_ok = 0
    for tgt_yaml, tgt_html in targets:
        try:
            tgt_result = result if tgt_yaml == yaml_path \
                else _result_from_yaml(tgt_yaml)
            tgt_calibrations = (calibrations if tgt_yaml == yaml_path
                                else _build_calibrations(tgt_result))
            if not tgt_result.board_poses:   # pass1: its own stored poses, else nothing projects
                _apply_stored_board_poses(tgt_result, load_yaml_file_safe(tgt_yaml))
            _restore_snap_sync(tgt_result)
            if _apply_stored_board_shape(tgt_result, tgt_yaml, board):
                logger.info(f"    {tgt_yaml.name}: refined board from its board_shape heights")
            generate_calibration_report(
                result=tgt_result, all_frames=all_frames, board=board,
                calibrations=tgt_calibrations, image_sizes=image_sizes,
                title=_make_title(tgt_yaml.stem, config_path_found),
                output_path=str(tgt_html),
                calibration_info=calibration_info,
                sample_detection_images={cid: imgs[:5]
                                         for cid, imgs in thumbs.items()},
                all_detection_images=thumbs,
                thumb_dir=out_dir / "thumbnails",
            )
            n_ok += 1
        except Exception as e:
            logger.error(f"    {tgt_yaml.name}: regen failed: {e}")
    return n_ok > 0


def main():
    p = argparse.ArgumentParser(description="Regenerate per-run HTML reports.")
    p.add_argument("root",
                   help="One run dir, or a root of <dataset>/<run>/ dirs to scan.")
    p.add_argument("--config", type=Path, default=None,
                   help="The run's own config (board etc.) when `root` is one run dir.")
    p.add_argument("--config-root", type=Path, default=None,
                   help="Directory searched (recursively) for the run configs when `root` "
                        "holds many runs: a YAML whose <parent>_<stem> prefixes the run id.")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    root = Path(args.root)
    if not root.exists():
        logger.error(f"Root not found: {root}")
        return 1

    # If `root` itself looks like a run dir (has *.yaml), regen it directly.
    direct_yaml = list(root.glob("*.yaml"))
    if direct_yaml and any(not y.stem.endswith("_pass1") for y in direct_yaml):
        ok = regenerate_one(root, args.config, args.config_root)
        return 0 if ok else 1

    # Otherwise treat as a root of runs: enumerate dataset/<slug>/ dirs.
    n_ok = n_fail = 0
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        for run_dir in sorted(ds_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if regenerate_one(run_dir, config_root=args.config_root):
                n_ok += 1
            else:
                n_fail += 1
    logger.info(f"Done: {n_ok} regenerated, {n_fail} skipped/failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
