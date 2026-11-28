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
"""Robust YAML config parser with nested section support.

Replaces the fragile ``BoardConfig(**data["board"])`` pattern which
crashes on unknown keys or nested sub-sections. This parser:

- Recursively parses nested dicts into their corresponding dataclass.
- Ignores unknown keys with a warning (forward compatibility).
- Uses defaults for missing keys (backward compatibility).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
from pathlib import Path
from typing import get_type_hints

import yaml

from mvis.core.config import (
    BoardConfig,
    CalibrationConfig,
    CameraConfig,
    CameraEstimateConfig,
    DetectionConfig,
    EstimateConfig,
    FrameQualityConfig,
    GuardrailConfig,
    OptimizerConfig,
    OutlierRejectionConfig,
    ReportConfig,
    TrajectoryAlignmentConfig,
)
from mvis.core.paths import DATASETS_DEFAULT, DATASETS_ENV

logger = logging.getLogger(__name__)

# Field-name aliases per config. The same concept is spelled
# differently across the two pipelines — accept BOTH spellings so a YAML written for
# one reads in the other; the canonical field wins if both are present. Keyed by
# class name to avoid imports.
_FIELD_ALIASES = {
    # camera-only canonical: pixel_sigma; per-camera rate canonical: factor_hz
    "OptimizerConfig": {"sigma_px": "pixel_sigma", "target_hz": "factor_hz"},
    "CameraConfig": {"target_hz": "factor_hz"},
    # cam-IMU canonical: sigma_px; per-camera rate canonical: factor_hz
    "CamImuCalibConfig": {"pixel_sigma": "sigma_px", "target_hz": "factor_hz"},
    # OpenCV's ChArUco / board-generator files call it `dictionary`
    "BoardConfig": {"dictionary": "aruco_dict"},
}


def _check_board_keys(data: dict, path: str) -> None:
    """Board-block checks the dataclass defaults would hide: `dictionary` and `aruco_dict`
    giving two names, or a ChArUco board without a dictionary (the DICT_6X6_250 default
    finds no corners on a board printed with another one)."""
    where = path or "board"
    d, a = data.get("dictionary"), data.get("aruco_dict")
    if d is not None and a is not None and d != a:
        raise ValueError(f"{where}: dictionary {d!r} and aruco_dict {a!r} disagree; give one")
    if data.get("board_type") == "charuco" and d is None and a is None:
        raise ValueError(f"{where}: a ChArUco board needs its dictionary "
                         "(aruco_dict: DICT_5X5_250, for example)")


_GRAVITY_LINE = re.compile(r"^(?P<indent>[ \t]*)gravity:[^#\n]*\s#(?P<comment>.*)$")
_UNIT = re.compile(r"^\s*m/s(?:²|\^2|2)\s*,?")


def _gravity_comments(text: str) -> tuple[str | None, str | None]:
    """The comments after a top-level `gravity:` and after a block-style `solve:` child `gravity:`
    in the raw config text (the YAML loader drops comments)."""
    top = solve = None
    in_solve, child_indent = False, None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line[0].isspace():                   # a top-level key ends any solve: block
            in_solve, child_indent = bool(re.match(r"solve:\s*(#.*)?$", line)), None
        elif in_solve and child_indent is None:
            child_indent = len(line) - len(line.lstrip())
        m = _GRAVITY_LINE.match(line)
        if not m:
            continue
        indent = len(m["indent"])
        if indent == 0 and top is None:
            top = m["comment"]
        elif in_solve and indent == child_indent and solve is None:
            solve = m["comment"]
    return top, solve


def gravity_place(comment: str | None) -> str | None:
    """The place a `gravity:` comment names: `# m/s², Newark, Delaware` -> "Newark, Delaware".
    The place ends at the first ';' (`# m/s², Wuhan, China; local 9.7936` -> "Wuhan, China");
    a unit-only or empty comment names none."""
    if comment is None:
        return None
    return _UNIT.sub("", comment.split(";", 1)[0], count=1).strip() or None


def _resolve_gravity(cfg, data: dict, text: str | None = None) -> None:
    """Resolve the run config's `gravity` / `site` (or `solve.gravity`, the same key) into
    cfg.solve.gravity and cfg.solve.gravity_source (mvis.core.gravity). `text`, the raw config,
    gives the place named in the comment after the `gravity:` value; without it, no place."""
    from mvis.core.gravity import resolve, site_record
    solve_g = (data.get("solve") or {}).get("gravity")
    if solve_g is not None and cfg.gravity is not None and float(solve_g) != float(cfg.gravity):
        raise ValueError(f"gravity {cfg.gravity} and solve.gravity {solve_g} disagree; give one")
    g = cfg.gravity if cfg.gravity is not None else solve_g
    cfg.solve.gravity, cfg.solve.gravity_source = resolve(g, cfg.site or None)
    cfg.solve.gravity_site = site_record(cfg.site)
    # with a site, the site names the place (regression comments such as "under 0.1 % ..." are no place)
    top, solve = _gravity_comments(text) if text and g is not None and not cfg.site else (None, None)
    cfg.solve.gravity_place = gravity_place(top) or gravity_place(solve)


def _all_corners_for_charuco(board, holder, explicit: bool) -> None:
    """ChArUco ids are chessboard corners, not tag * 4 + corner, so corner_subsample
    has nothing to thin: set it to "all" (warn only when the config asked for another)."""
    if board.board_type != "charuco" or holder.corner_subsample == "all":
        return
    if explicit:
        logger.warning(f"corner_subsample {holder.corner_subsample!r} has no meaning on a "
                       "ChArUco board; using 'all'")
    holder.corner_subsample = "all"

# Boolean aliases whose VALUE inverts: pin_tr -> estimate_readout.
# pin_tr=True ("pin the readout" = global) == estimate_readout=False, so the canonical
# is the negation. Canonical wins if both are present.
_INVERTING_BOOL_ALIASES = {
    "CamImuCalibConfig": {"pin_tr": "estimate_readout"},
    "CamImuCameraInputConfig": {"pin_tr": "estimate_readout"},
}


def _parse_dataclass(data: dict | None, cls: type, path: str = "") -> object:
    """Recursively parse a dict into a dataclass instance.

    For each field of ``cls``:
    - If the field's type is itself a dataclass and ``data`` has a
      matching dict → recurse.
    - If the field is a primitive and ``data`` has a matching value →
      assign (with basic type coercion for int↔float).
    - If ``data`` has a key not in the dataclass → log a warning
      (helps catch typos).
    - If the field is missing from ``data`` → use the dataclass default.
    """
    if data is None:
        return cls()
    if cls.__name__ == "BoardConfig":
        _check_board_keys(data, path)

    # Translate field-name aliases to the canonical name (canonical wins on conflict;
    # the alias is consumed so it doesn't trip the unknown-key warning).
    aliases = _FIELD_ALIASES.get(cls.__name__)
    if aliases and any(a in data for a in aliases):
        data = dict(data)
        for alias, canon in aliases.items():
            if alias in data:
                if canon not in data:
                    data[canon] = data[alias]
                del data[alias]

    inv = _INVERTING_BOOL_ALIASES.get(cls.__name__)
    if inv and any(a in data for a in inv):
        data = dict(data)
        for alias, canon in inv.items():
            if alias in data:
                if canon not in data:
                    data[canon] = not bool(data[alias])
                del data[alias]

    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    used_keys = set()

    for name, field in fields.items():
        if name not in data:
            continue
        used_keys.add(name)
        value = data[name]
        field_type = field.type

        # Check if the field type is a dataclass (nested config).
        # Resolve string annotations if needed.
        actual_type = None
        if isinstance(field_type, str):
            # Try to resolve from the module's namespace
            try:
                actual_type = eval(field_type, globals())
            except Exception:
                actual_type = None
        elif isinstance(field_type, type):
            actual_type = field_type
        else:
            # Could be a generic (list, dict, Optional, etc.)
            actual_type = None

        if (actual_type is not None
                and dataclasses.is_dataclass(actual_type)
                and isinstance(value, dict)):
            kwargs[name] = _parse_dataclass(
                value, actual_type, path=f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value

    # Warn about unknown keys (typo detection / forward compat).
    for key in data:
        if key not in fields and key not in used_keys:
            full_path = f"{path}.{key}" if path else key
            logger.warning(f"Config: unknown key '{full_path}' — ignored")

    return cls(**kwargs)


def _fill_camera_factor_hz(cameras, raw_cameras, optimizer):
    """Fill each camera's `factor_hz` from the global `optimizer.factor_hz` when that
    camera omitted its own in the YAML (per-camera value wins). The camera-only
    analog of the cam-IMU global/per-camera factor_hz. The
    legacy `target_hz` YAML key counts as the camera specifying its own."""
    g = getattr(optimizer, "factor_hz", None)
    if g is None:
        return
    for raw, cam in zip(raw_cameras or [], cameras or []):
        if isinstance(raw, dict) and "factor_hz" not in raw and "target_hz" not in raw:
            cam.factor_hz = g


def _unify_plural_time_offset(optimizer, raw_opt, td_in_top, set_td):
    """Fold the vestigial plural `optimizer.optimize_time_offsets` into the live
    singular top-level `optimize_time_offset` (unified-estimate). Deprecation;
    the singular (or estimate.camera.td) wins, so skip if the top-level set it."""
    if isinstance(raw_opt, dict) and "optimize_time_offsets" in raw_opt and not td_in_top:
        logger.warning("Config: optimizer.optimize_time_offsets is deprecated — "
                       "use top-level optimize_time_offset (or estimate.camera.td)")
        set_td(bool(getattr(optimizer, "optimize_time_offsets", False)))


def _apply_estimate_camera(est, optimizer, raw_opt, raw_top, set_td):
    """Map a rig-wide EstimateConfig onto the legacy camera-only knobs
    (unified-estimate U2). estimate is the source of truth: each specified key
    WINS over the legacy field (warn on explicit conflict). Keys the camera-only
    solver does not consume yet (camera.distortion/extrinsics) are recorded but
    warned; the imu sub-block is warned as no-op for camera-only."""
    cam, raw_opt, raw_top = est.camera, (raw_opt or {}), (raw_top or {})

    def conflict(name, present, path):
        """Warn that an explicit legacy knob is being overridden by `estimate.camera.<name>`."""
        if present:
            logger.warning(f"Config: both estimate.camera.{name} and {path} set "
                           f"— estimate wins")

    if cam.intrinsics is not None:
        conflict("intrinsics", "optimize_intrinsics" in raw_opt,
                 "optimizer.optimize_intrinsics")
        optimizer.optimize_intrinsics = bool(cam.intrinsics)
    if cam.tr is not None:
        conflict("tr", "optimize_rolling_shutter" in raw_opt,
                 "optimizer.optimize_rolling_shutter")
        optimizer.optimize_rolling_shutter = bool(cam.tr)
    if cam.td is not None:
        conflict("td", "optimize_time_offset" in raw_top, "optimize_time_offset")
        set_td(bool(cam.td))

    # recorded onto the dataclass but NOT consumed by the camera-only solver yet
    recorded = []
    if cam.distortion is not None:
        optimizer.optimize_distortion = bool(cam.distortion)
        recorded.append("camera.distortion")
    if cam.extrinsics is not None:
        optimizer.optimize_extrinsics = bool(cam.extrinsics)
        recorded.append("camera.extrinsics")
    if est.board_points is not None:
        # board pose + per-corner heights
        optimizer.optimize_board_points = bool(est.board_points)
        logger.info("Config: estimate.board_points=%s -> camera-only board pose + "
                    "per-corner heights", est.board_points)
    if recorded:
        logger.warning("Config: estimate." + " / estimate.".join(recorded)
                       + " recorded but NOT consumed by the camera-only solver yet "
                       "(it always estimates these).")
    imu = est.imu
    if any(v is not None for v in (imu.intrinsics, imu.model,
                                   imu.extrinsics, imu.td)):
        logger.warning("Config: estimate.imu.* has no effect on the camera-only "
                       "pipeline — ignored")


def _apply_estimate_cam_imu(cfg, data):
    """Map a rig-wide EstimateConfig onto the legacy cam-IMU knobs
    (unified-estimate U3). estimate is the source of truth: each specified key
    WINS over the legacy field (warn on explicit conflict). `cfg` is a
    CamImuRunConfig (cfg.solve = CamImuCalibConfig, cfg.camera_input)."""
    est = cfg.estimate
    cam, imu = est.camera, est.imu
    solve, cin = cfg.solve, cfg.camera_input
    raw = data or {}
    raw_solve = raw.get("solve") or {}
    raw_cin = raw.get("camera_input") or {}

    def warn(name, present, path):
        """Warn that an explicit legacy knob is being overridden by `estimate.<name>`."""
        if present:
            logger.warning(f"Config: both estimate.{name} and {path} set "
                           f"— estimate wins")

    # --- camera ---
    if cam.intrinsics is not None:
        warn("camera.intrinsics", "free_intrinsics" in raw_cin,
             "camera_input.free_intrinsics")
        cin.free_intrinsics = bool(cam.intrinsics)
    if cam.distortion is not None and cam.intrinsics is None:
        logger.warning("Config: estimate.camera.distortion is folded into "
                       "camera.intrinsics for cam-IMU — set camera.intrinsics")
    if cam.tr is not None:
        warn("camera.tr", "estimate_readout" in raw_cin or "estimate_readout"
             in raw_solve, "estimate_readout")
        cin.estimate_readout = bool(cam.tr)
        solve.estimate_readout = bool(cam.tr)
    if cam.td is not None:
        solve.estimate_td = bool(cam.td)           # base camera <-> IMU td
    if cam.extrinsics is not None:
        solve.estimate_extrinsics = bool(cam.extrinsics)   # cam-IMU T_C->I

    # --- imu ---
    if imu.intrinsics is not None:
        warn("imu.intrinsics", "imu_intrinsics" in raw_solve,
             "solve.imu_intrinsics")
        if isinstance(imu.intrinsics, bool):
            solve.imu_intrinsics = "full" if imu.intrinsics else "pinned"
        else:
            solve.imu_intrinsics = str(imu.intrinsics)
    if imu.model is not None:
        warn("imu.model", "imu_intrinsics_model" in raw_solve,
             "solve.imu_intrinsics_model")
        solve.imu_intrinsics_model = str(imu.model)
    if imu.extrinsics is not None:
        solve.estimate_aux_extrinsics = bool(imu.extrinsics)
    if imu.td is not None:
        solve.estimate_aux_td = bool(imu.td)
    # --- board ---
    if est.board_points is not None:
        solve.estimate_board_heights = bool(est.board_points)
        logger.info("Config: estimate.board_points=%s -> cam-IMU board shape "
                    "(per-corner heights) %s", est.board_points,
                    "estimated" if est.board_points else "fixed")


_ENV_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# The `core.paths` default: datasets live under ~/Datasets unless MVIS_DATASETS says
# otherwise, so shipped configs can say ${MVIS_DATASETS}/...
_ENV_DEFAULTS = {DATASETS_ENV: DATASETS_DEFAULT}


def expand_path_vars(value):
    """Expand ``${VAR}`` and a leading ``~`` in every string of a parsed YAML tree.

    Unset variables are left as written, except ``MVIS_DATASETS`` which defaults
    to ``~/Datasets``. Strings without ``${`` or a leading ``~`` are unchanged.
    """
    if isinstance(value, dict):
        return {k: expand_path_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_path_vars(v) for v in value]
    if not isinstance(value, str) or ("${" not in value and not value.startswith("~")):
        return value
    out = _ENV_VAR.sub(
        lambda m: os.environ.get(m.group(1), _ENV_DEFAULTS.get(m.group(1), m.group(0))),
        value)
    return os.path.expanduser(out)


def load_yaml(path: str) -> dict:
    """Read a config YAML and expand path variables (see ``expand_path_vars``)."""
    with open(path) as f:
        return expand_path_vars(yaml.safe_load(f) or {})


def parse_config(path: str) -> CalibrationConfig:
    """Parse a calibration YAML file into a CalibrationConfig.

    Handles nested sections (board.detection, optimizer.frame_quality,
    etc.), unknown keys (warned, not crashed), and missing sections
    (defaults applied).
    """
    data = load_yaml(path)

    config = CalibrationConfig()

    # Board (with nested detection sub-config).
    if "board" in data:
        config.board = _parse_dataclass(data["board"], BoardConfig, "board")

    # Cameras (list of CameraConfig).
    if "cameras" in data:
        config.cameras = [
            _parse_dataclass(cam, CameraConfig, f"cameras[{i}]")
            for i, cam in enumerate(data["cameras"])
        ]

    # Optimizer (with nested sub-configs).
    if "optimizer" in data:
        config.optimizer = _parse_dataclass(
            data["optimizer"], OptimizerConfig, "optimizer")

    # Report config.
    if "report" in data:
        config.report = _parse_dataclass(
            data["report"], ReportConfig, "report")

    # Top-level scalars.
    _TOP_LEVEL_SCALARS = [
        "reference_camera", "output_file", "output_format",
        "corner_subsample", "rerun_outliers", "optimize_time_offset",
        "rerun_visualization",
    ]
    for key in _TOP_LEVEL_SCALARS:
        if key in data:
            setattr(config, key, data[key])

    # Per-camera estimate: overrides (unified-estimate).
    for raw, cam in zip(data.get("cameras", []) or [], config.cameras):
        if isinstance(raw, dict) and isinstance(raw.get("estimate"), dict):
            cam.estimate = _parse_dataclass(
                raw["estimate"], CameraEstimateConfig, "cameras[].estimate")

    # Unify the vestigial plural time-offset, then apply the rig-wide estimate:.
    def _set_td(v):
        """Setter handed to the estimate/plural-td unifiers (writes the canonical field)."""
        config.optimize_time_offset = v
    _unify_plural_time_offset(config.optimizer, data.get("optimizer"),
                              "optimize_time_offset" in data, _set_td)
    if "estimate" in data:
        config.estimate = _parse_dataclass(
            data["estimate"], EstimateConfig, "estimate")
        _apply_estimate_camera(config.estimate, config.optimizer,
                               data.get("optimizer"), data, _set_td)

    _fill_camera_factor_hz(config.cameras, data.get("cameras", []), config.optimizer)
    _all_corners_for_charuco(config.board, config, "corner_subsample" in data)
    return config


def parse_cam_imu_config(path: str):
    """Parse a cam-IMU run YAML into a CamImuRunConfig.

    Recursively parses the nested sections (input, board, camera_input, solve,
    output) via ``_parse_dataclass`` — unknown keys are warned (not crashed),
    missing sections default.
    """
    from mvis.core.config import CamImuRunConfig
    data = load_yaml(path)
    text = Path(path).read_text(encoding="utf-8", errors="replace")  # for the place in the `gravity:` comment
    # `mode:` is the `mvis run` dispatch key, already consumed by the caller
    data = {k: v for k, v in data.items() if k != "mode"} if isinstance(data, dict) else data
    cfg = _parse_dataclass(data, CamImuRunConfig, "cam_imu")
    if isinstance(data, dict) and "estimate" in data:
        _apply_estimate_cam_imu(cfg, data)
    _all_corners_for_charuco(cfg.board, cfg.solve, True)
    _resolve_gravity(cfg, data if isinstance(data, dict) else {}, text)
    return cfg


def parse_cam_run_config(path: str):
    """Parse a camera-only run YAML into a CamRunConfig.

    Accepts BOTH shapes:
    - NEW sectioned: ``input.cameras`` / ``board`` / ``solve`` / ``output`` /
      ``report`` + run scalars (mirrors CamImuRunConfig).
    - LEGACY flat CalibrationConfig: ``cameras`` / ``optimizer`` / ``output_file`` /
      top-level scalars — parsed via ``parse_config`` and adapted.
    """
    from mvis.core.config import (
        CamRunConfig, CamInputConfig, CamOutputConfig,
        BoardConfig, OptimizerConfig, ReportConfig, CameraConfig,
    )
    data = load_yaml(path)

    sectioned = ("input" in data or "solve" in data
                 or (isinstance(data.get("output"), dict)))
    if not sectioned:
        # Legacy flat shape -> parse as CalibrationConfig (which applies estimate:),
        # then adapt.
        cc = parse_config(path)
        return CamRunConfig(
            input=CamInputConfig(cameras=cc.cameras),
            board=cc.board, solve=cc.optimizer,
            output=CamOutputConfig(path=cc.output_file, format=cc.output_format),
            report=cc.report, estimate=cc.estimate,
            reference_camera=cc.reference_camera,
            corner_subsample=cc.corner_subsample, rerun_outliers=cc.rerun_outliers,
            optimize_time_offset=cc.optimize_time_offset,
            rerun_visualization=cc.rerun_visualization)

    cfg = CamRunConfig()
    if isinstance(data.get("input"), dict):
        cams = data["input"].get("cameras", []) or []
        cfg.input = CamInputConfig(cameras=[
            _parse_dataclass(c, CameraConfig, f"input.cameras[{i}]")
            for i, c in enumerate(cams)],
            cache_dir=data["input"].get("cache_dir"))
    if "board" in data:
        cfg.board = _parse_dataclass(data["board"], BoardConfig, "board")
    solve = data.get("solve")
    if isinstance(solve, dict) and "corner_subsample" in solve:
        # solve.corner_subsample is an alias of the top-level key (the cam-IMU configs
        # spell it under solve:); the top-level key wins when both are set
        solve = {k: v for k, v in solve.items() if k != "corner_subsample"}
        if "corner_subsample" not in data:
            cfg.corner_subsample = data["solve"]["corner_subsample"]
    if "solve" in data:
        cfg.solve = _parse_dataclass(solve, OptimizerConfig, "solve")
    if isinstance(data.get("output"), dict):
        cfg.output = _parse_dataclass(data["output"], CamOutputConfig, "output")
    if "report" in data:
        cfg.report = _parse_dataclass(data["report"], ReportConfig, "report")
    for key in ("reference_camera", "corner_subsample", "rerun_outliers",
                "optimize_time_offset", "rerun_visualization"):
        if key in data:
            setattr(cfg, key, data[key])

    # Per-camera estimate: overrides (unified-estimate).
    for raw, cam in zip((data.get("input") or {}).get("cameras", []) or [],
                        cfg.input.cameras):
        if isinstance(raw, dict) and isinstance(raw.get("estimate"), dict):
            cam.estimate = _parse_dataclass(
                raw["estimate"], CameraEstimateConfig, "input.cameras[].estimate")

    # Unify the vestigial plural time-offset, then apply the rig-wide estimate:.
    def _set_td(v):
        """Setter handed to the estimate/plural-td unifiers (writes the canonical field)."""
        cfg.optimize_time_offset = v
    _unify_plural_time_offset(cfg.solve, data.get("solve"),
                              "optimize_time_offset" in data, _set_td)
    if "estimate" in data:
        cfg.estimate = _parse_dataclass(
            data["estimate"], EstimateConfig, "estimate")
        _apply_estimate_camera(cfg.estimate, cfg.solve,
                               data.get("solve"), data, _set_td)

    _fill_camera_factor_hz(cfg.input.cameras,
                           (data.get("input") or {}).get("cameras", []), cfg.solve)
    _all_corners_for_charuco(cfg.board, cfg, "corner_subsample" in data
                             or "corner_subsample" in (data.get("solve") or {}))
    return cfg
