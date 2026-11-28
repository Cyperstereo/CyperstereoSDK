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
"""Calibration configuration — all dataclass definitions.

This module defines the config schema. Parsing from YAML is in
``config_parser.py``; validation in ``config_validator.py``.

Backward compatibility: ``CalibrationConfig.from_yaml()`` still works
(delegates to the new parser internally).
"""

from dataclasses import dataclass, field
from typing import Optional

import yaml

from mvis.core.camera_models import resolve_model_alias


# ---------------------------------------------------------------------------
# Board detection tuning
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    """AprilTag adaptive multi-scale detection parameters."""

    min_tag_px: float = 20.0         # below this, upscale triggers
    max_scale: float | None = None   # None = auto (4/6/8 by image size)
    module_px_target: float = 25.0   # target module size for adaptive scale
    early_stop_rate: float = 0.95    # stop scales when this coverage reached
    moderate_rate: float = 0.70      # below this, try 3rd adaptive scale


@dataclass
class BoardConfig:
    """Calibration board configuration."""

    board_type: str = "chessboard"
    rows: int = 9
    cols: int = 6
    square_size: float = 0.025
    tag_family: str = "t36h11"
    tag_size: float = 0.02
    tag_spacing: float = 0.005
    aruco_dict: str = "DICT_6X6_250"
    marker_size: float = 0.02
    marker_separation: float = 0.005
    # ChArUco boards printed with OpenCV < 4.6 (other marker layout on even row counts)
    charuco_legacy: bool = False
    # ChArUco: refine each corner with cv2.cornerSubPix (OpenCV's own are ~0.5 px off).
    # Not `refine_subpixel`, which is ArUco marker-corner refinement.
    charuco_refine: bool = True
    refine_subpixel: bool = False
    # Measured board shape: YAML with `board_shape.heights_m`
    # (one out-of-plane height per corner, as `mvis cam` writes with
    # estimate.board_points) or an .npz with `dp` (N x 3 offsets). The corners
    # used by the solvers become (x, y, h) instead of (x, y, 0).
    heights_file: str | None = None
    detection: DetectionConfig = field(default_factory=DetectionConfig)


# ---------------------------------------------------------------------------
# Unified "what to calibrate" switchboard (unified-estimate)
# ---------------------------------------------------------------------------

@dataclass
class CameraEstimateConfig:
    """Per-camera "what to calibrate" knobs. Used BOTH as the rig-wide camera
    default (`estimate.camera`) and as a per-camera override on a camera entry
    (`cameras[i].estimate`) — the per-camera value wins where set, else the
    rig-wide default, else the pipeline's legacy default.

    TRI-STATE: None = unspecified (inherit); bool = estimate / hold fixed.
    """
    intrinsics: bool | None = None    # camera focal length / principal point
    distortion: bool | None = None    # distortion params (camera-only granularity;
                                      #   cam-IMU folds distortion into intrinsics)
    tr: bool | None = None            # rolling-shutter readout (RS cameras only)
    td: bool | None = None            # time offset (cam <-> IMU; cam-IMU only)
    extrinsics: bool | None = None    # cam-cam (camera-only) / cam-IMU T_C->I


@dataclass
class ImuEstimateConfig:
    """The rig-wide IMU "what to calibrate" knobs (`estimate.imu`). The IMU block
    is rig-wide only: an `input.imus[i]` entry carries just `gyro_only`.

    TRI-STATE: None = unspecified (inherit).
    """
    intrinsics: bool | str | None = None  # HOW MUCH: bool (True=full,
                                      #   False=pinned) OR staging string
                                      #   "pinned"|"dw_da"|"dw_da_r"|"full"
    model: str | None = None          # WHICH model family: "kalibr"|"rpng"|
                                      #   "calyx" (alias of imu_intrinsics_model;
                                      #   rig-wide — one family per rig today)
    extrinsics: bool | None = None    # IMU-IMU extrinsic T_IstoI (aux IMUs;
                                      #   base IMU is the gauge, not estimated)
    td: bool | None = None            # aux-IMU <-> base-IMU time offset (aux IMUs;
                                      #   base td = 0 by definition)


@dataclass
class EstimateConfig:
    """Symmetric `estimate:` block — the canonical, SOURCE-OF-TRUTH way to choose
    WHICH quantities are calibrated, usable on BOTH the camera-only
    (CalibrationConfig/CamRunConfig) and cam-IMU (CamImuRunConfig) top-level
    configs. Split by sensor kind (camera / imu) so each only carries the keys
    that apply to it; shared concepts keep the same name across both (extrinsics).

    Pattern (mirrors `factor_hz`): `estimate.camera` / `estimate.imu` are rig-wide
    DEFAULTS; a per-sensor `estimate:` on a camera/imu entry OVERRIDES only the
    keys that differ. When present, an `estimate:` value WINS over the legacy
    low-level knob it maps to (`optimize_*`, `free_intrinsics`, `estimate_readout`,
    `sigma_td_prior`, `imu_intrinsics`); the parser warns on an explicit conflict.
    When absent, legacy fields stand unchanged (existing configs behavior-identical).

    Camera-only note (unified-estimate U1): `camera.distortion` / `camera.extrinsics`
    are not yet consumed by the camera-only solver (it always estimates them) —
    recorded but the parser warns. `board_points` is honoured by both the camera-only
    and the cam-IMU solver. The `imu` sub-block has no effect on the camera-only pipeline.
    """
    camera: CameraEstimateConfig = field(default_factory=CameraEstimateConfig)
    imu: ImuEstimateConfig = field(default_factory=ImuEstimateConfig)
    board_points: bool | None = None  # board shape: per-corner heights (camera-only AND cam-IMU)


# ---------------------------------------------------------------------------
# Per-camera configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Per-camera input configuration."""

    camera_id: int = 0
    name: str = ""
    source_type: str = "images"
    source_path: str = ""
    topic: str = ""
    distortion_model: str = "radtan"
    shutter_type: str = "auto"       # auto / rolling / global
    factor_hz: float = 10.0          # per-camera frame/vision-factor rate
                                     # (YAML alias: target_hz)
    # Optional initial calibration SEED for Stage-1 init: [fx, fy, cx, cy]
    # (+ distortion coeffs). When set, Stage 1 uses these instead of the
    # OpenCV / vanishing-point init — for rigs where that init lands in a
    # basin the joint solve cannot escape (joint-pair-defect, T5 saga).
    # initial_distortion has the model's coeffs (ucm: alpha; omni_radtan: xi,k1,k2,p1,p2;
    # a legacy 4-value omni seed gets p2 = 0).
    initial_intrinsics: list | None = None
    initial_distortion: list | None = None
    # Per-camera "what to calibrate" override (unified-estimate). Tri-state keys
    # override the rig-wide estimate.camera default where set.
    estimate: "CameraEstimateConfig" = field(default_factory=CameraEstimateConfig)

    def __post_init__(self):
        # accepted kb4 / ucm spellings (kannala_brandt4, kb_4, ...) -> the id; a removed
        # model (kb8) raises here, at config load; any other name is left for
        # config_validator to accept / reject
        self.distortion_model = resolve_model_alias(self.distortion_model)


# ---------------------------------------------------------------------------
# Subsample / diversity filtering
# ---------------------------------------------------------------------------
# Optimizer sub-configs
# ---------------------------------------------------------------------------

@dataclass
class FrameQualityConfig:
    """Detection-quality gating between Stage 1 and Stage 2."""

    pose_filter: bool = True
    min_pose_tag_fraction: float = 0.20
    min_pose_tag_floor: int = 4
    min_corner_views: int = 5
    auto_fallback_min_tags: int = 10
    # Per-frame: skip frames with fewer than this many unique tags
    min_tags_per_frame: int = 6
    # Per-camera: drop corners seen < this many times within a single camera
    min_corner_views_per_camera: int = 3
    # PnP: discard poses with median reprojection error above this
    max_pnp_reprojection_px: float = 10.0
    # Validate Stage 1 intrinsics before feeding to Stage 2
    stage1_intrinsic_validation: bool = True
    # Minimum convex hull fraction of tag centers / image area
    min_hull_fraction: float = 0.01


@dataclass
class TrajectoryAlignmentConfig:
    """Stage 3 extrinsic + time-offset initialization."""

    ransac_iterations: int = 1000
    ransac_inlier_threshold_deg: float = 5.0
    cross_correlation_resolution_ms: int = 10
    max_expected_dt_s: float = 0.5
    upsample_hz: int = 100


@dataclass
class OutlierRejectionConfig:
    """Stage 5 two-pass outlier filtering."""

    enabled: bool = True
    threshold_px: float = 2.0
    percentile: float = 99.0
    frame_median_threshold_px: float | None = None
    frame_median_percentile: float = 99.0


@dataclass
class GuardrailConfig:
    """Pass-2 sanity check thresholds."""

    fx_min: float = 1.0
    fx_fy_max_asymmetry: float = 0.30
    distortion_max_abs: float = 2.0
    p95_regression_factor: float = 1.5


@dataclass
class OptimizerConfig:
    """Optimization settings with nested sub-configs."""

    max_iterations: int = 200
    max_init_frames: int = 100  # max frames for OpenCV calibrateCamera init (0 = no cap)
    init_frame_selection: str = "uniform"  # "uniform" (time-spaced) or "diverse" (position/size/orientation bins)
    # Global per-camera frame/vision-factor rate default — the camera-only analog of
    # CamImuCalibConfig.factor_hz. from_yaml fills each camera's
    # CameraConfig.factor_hz from this when that camera omits its own (per-camera
    # wins). Default 5.0 = the established per-camera default (behavior-preserving).
    factor_hz: float = 5.0          # (YAML alias: target_hz)
    optimize_intrinsics: bool = True
    optimize_distortion: bool = True
    optimize_board_points: bool = False
    optimize_extrinsics: bool = True
    optimize_time_offsets: bool = False
    optimize_rolling_shutter: bool = False
    huber_threshold: float = 1.345
    pixel_sigma: float = 1.0
    # PnP solver for board-pose init: 'iterative' (legacy) | 'ippe' (planar-
    # correct; resolves the two-fold ambiguity that flips pose basins — the
    # fix needed by low-res planar rigs like the T5 Reverb G2).
    pnp_method: str = "iterative"
    pnp_subprocess: bool = False  # board-pose PnP in a spawn child — survives
                                  # rare native OpenCV/GTSAM segfaults on
                                  # degenerate frames (seen in a field batch)
    anchor_prior_sigma: float = 1000.0
    # LM termination (cam-imu-speedup L1, 2026-10-31). GTSAM declares convergence
    # when the relative OR the absolute cost decrease of an accepted iteration is
    # below its tolerance; 1e-5 is the GTSAM default and what the cam-IMU solve
    # uses. The old 1e-10 never fired on a ~1e4-cost bundle: LM entered one more
    # iteration and burnt up to 15 rejected lambda retries (~45 s each on a
    # 100k-observation solve) before "giving up because cannot decrease error
    # with maximum lambda" — 9 / 36 board_rep cam-only runs, up to half the
    # solve time. The returned values are the last accepted iterate either way.
    absolute_error_tol: float = 1e-5
    relative_error_tol: float = 1e-5
    # LM damping bounds (GTSAM defaults 0 / 1e5). The upper bound caps the retry
    # loop of a genuinely stuck iteration: lambda x10 per rejected try, give up
    # at the bound, so the number of tries is the number of decades between the
    # lambda at the stall and the bound (GTSAM starts at 1e-5 -> 7 tries; after
    # k accepted iterations lambda is 1e-5 / 10^k, e.g. 12 tries from 1e-10).
    # The lower bound stays at GTSAM's 0 so the damping path, and hence every
    # accepted iterate, is bitwise what it was before L1; `lm_lambda_lower: 1e-6`
    # caps a late stall at 8 tries at the price of ~1e-11 relative result drift.
    lm_lambda_lower: float = 0.0
    lm_lambda_upper: float = 1e2
    # Nested sub-configs
    frame_quality: FrameQualityConfig = field(default_factory=FrameQualityConfig)
    trajectory_alignment: TrajectoryAlignmentConfig = field(
        default_factory=TrajectoryAlignmentConfig)
    outlier_rejection: OutlierRejectionConfig = field(
        default_factory=OutlierRejectionConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    """HTML report generation settings."""

    thumbnail_max_dim: int = 320
    # Appendix A: up to this many lowest-detection frames per camera, as PNG thumbnails
    # (~150 kB each, ~11 MB for 2 x 30 frames). 0 (default) leaves the appendix out.
    appendix_max_frames: int = 0


# ---------------------------------------------------------------------------
# Top-level calibration config
# ---------------------------------------------------------------------------

@dataclass
class CalibrationConfig:
    """Top-level calibration configuration."""

    cameras: list[CameraConfig] = field(default_factory=list)
    board: BoardConfig = field(default_factory=BoardConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    # Unified "what to calibrate" switchboard (unified-estimate). When present in
    # the YAML the parser writes it down into optimizer.* / optimize_time_offset.
    estimate: "EstimateConfig" = field(default_factory=EstimateConfig)
    reference_camera: int = 0
    output_file: str = "calibration_result.yaml"
    output_format: str = "yaml"

    # Per-tag corner subsampling for AprilGrid:
    #   "two_anti" (default) — 2 corners/tag (top-left + bottom-right), ~20% faster
    #                          than "all", same principal point within ~0.1 px
    #   "two_diag"           — 2 corners/tag, the other diagonal
    #   "all"                — 4 corners/tag, no speedup, most data
    #   "top_left"           — 1 corner/tag, fastest; fit is unchanged but cx, cy
    #                          shift by up to ~0.8 px in a direction set by the corner
    # Also: "bottom_left", "bottom_right", "top_right" (single corners, same shift)
    # CAMERA-only canonical. Precedence: CLI --corner-subsample > this field.
    # (Cam-IMU has its own CamImuCalibConfig.corner_subsample, default "all" — the
    # two pipelines differ intentionally: camera-only thins for speed, cam-IMU keeps
    # all corners for accuracy.)
    corner_subsample: str = "two_anti"

    # Enable outlier-driven rerun: after joint optimization, filter
    # outlier observations and rerun warm-started.
    rerun_outliers: bool = True

    # Enable camera-camera time offset optimization (multi-camera only).
    # Set true only for unsynchronized cameras.
    optimize_time_offset: bool = False

    # Launch Rerun viewer after calibration to visualize results.
    # Off by default; opt in with rerun_visualization: true in the config
    # (python -m mvis.calibration.cam_run also takes --rerun-viz).
    rerun_visualization: bool = False

    @classmethod
    def from_yaml(cls, path: str) -> "CalibrationConfig":
        """Load configuration from a YAML file.

        Delegates to ``config_parser.parse_config`` for robust nested
        parsing. Falls back to the legacy ``**kwargs`` parser if the
        new parser is unavailable.
        """
        try:
            from mvis.core.config_parser import parse_config
            return parse_config(path)
        except ImportError:
            # Fallback: legacy parser (no nested section support).
            return cls._legacy_from_yaml(path)

    @classmethod
    def _legacy_from_yaml(cls, path: str) -> "CalibrationConfig":
        """Legacy parser — kept for backward compatibility."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        config = cls()
        if "board" in data:
            bd = data["board"]
            # Strip nested sections the legacy parser can't handle.
            bd = {k: v for k, v in bd.items()
                  if not isinstance(v, dict)}
            config.board = BoardConfig(**bd)
        if "cameras" in data:
            config.cameras = [
                CameraConfig(**{k: v for k, v in c.items()
                                if not isinstance(v, dict)})
                for c in data["cameras"]
            ]
        if "optimizer" in data:
            opt = {k: v for k, v in data["optimizer"].items()
                   if not isinstance(v, dict)}
            config.optimizer = OptimizerConfig(**opt)
        if "reference_camera" in data:
            config.reference_camera = data["reference_camera"]
        if "output_file" in data:
            config.output_file = data["output_file"]
        if "output_format" in data:
            config.output_format = data["output_format"]
        return config

    def to_yaml(self, path: str) -> None:
        """Save configuration to a YAML file."""
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)


# ===========================================================================
# Camera-only RUN config — sectioned shape mirroring CamImuRunConfig
# ===========================================================================
# CamRunConfig is the sectioned camera-only run config (input / board / solve /
# output), the analog of CamImuRunConfig. It is READ-COMPATIBLE with the legacy
# flat CalibrationConfig — it exposes `.cameras` / `.optimizer` / `.output_file` /
# `.output_format` as properties, so the same calibrate() / validator / scripts
# consume either. `from_yaml` accepts BOTH the new sectioned shape and the legacy
# flat shape (delegating to the CalibrationConfig parser + adapting).

@dataclass
class CamInputConfig:
    """Camera-only input — the per-camera sources (analog of CamImuInputConfig)."""
    cameras: list[CameraConfig] = field(default_factory=list)
    # Shared detection store dir (io.stream_store), the same key as the cam-IMU
    # input.cache_dir: a camera-only and a cam-IMU run on one bag detect it once.
    cache_dir: str | None = None


@dataclass
class CamOutputConfig:
    """Where to write the camera-only calibration (analog of CamImuOutputConfig)."""
    path: str = "calibration_result.yaml"
    format: str = "yaml"


@dataclass
class CamRunConfig:
    """Top-level camera-only run config (sectioned), mirroring CamImuRunConfig."""
    input: CamInputConfig = field(default_factory=CamInputConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    solve: OptimizerConfig = field(default_factory=OptimizerConfig)
    output: CamOutputConfig = field(default_factory=CamOutputConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    # Unified "what to calibrate" switchboard (unified-estimate).
    estimate: "EstimateConfig" = field(default_factory=EstimateConfig)
    # run-level scalars (same names + defaults as the legacy CalibrationConfig)
    reference_camera: int = 0
    corner_subsample: str = "two_anti"
    rerun_outliers: bool = True
    optimize_time_offset: bool = False
    rerun_visualization: bool = False   # Rerun viewer opt-in (off by default since 2026-08-30)

    # --- legacy read-compat (so CalibrationConfig consumers work unchanged) ---
    @property
    def cameras(self) -> list:
        """Legacy alias for `input.cameras` (CalibrationConfig read-compat)."""
        return self.input.cameras

    @property
    def optimizer(self) -> "OptimizerConfig":
        """Legacy alias for `solve` (CalibrationConfig read-compat)."""
        return self.solve

    @property
    def output_file(self) -> str:
        """Legacy alias for `output.path` (CalibrationConfig read-compat)."""
        return self.output.path

    @property
    def output_format(self) -> str:
        """Legacy alias for `output.format` (CalibrationConfig read-compat)."""
        return self.output.format

    @classmethod
    def from_yaml(cls, path: str) -> "CamRunConfig":
        """Parse a camera-only run YAML — accepts the new sectioned shape AND the
        legacy flat CalibrationConfig shape (adapted)."""
        from mvis.core.config_parser import parse_cam_run_config
        return parse_cam_run_config(path)


# ===========================================================================
# Cam-IMU calibration config
# ===========================================================================
# One config system for the joint cam-IMU pipeline: dataset input, board,
# camera-intrinsics input (MVIS/Kalibr), solve knobs, output. The solve
# knobs (CamImuCalibConfig) live HERE canonically and are re-exported from
# calibration.pipeline_cam_imu for backward compatibility.

@dataclass
class CamCalibConfig:
    """Camera-only SOLVE config — the user-facing knobs, analog of CamImuCalibConfig.

    Consumed by `calibration.pipeline_cam.run_cam_calibration[_multi]`, which maps it onto
    the internal `PipelineConfig`. Names mirror `CamImuCalibConfig` where they overlap
    (`sigma_px`, `huber_px`, `max_iterations`, `corner_subsample`)."""
    max_iterations: int = 15
    sigma_px: float = 1.0            # pixel obs noise (-> PipelineConfig.pixel_sigma)
    huber_px: float = 1.345          # robust loss thresh (-> PipelineConfig.huber_threshold)
    corner_subsample: str = "two_anti"
    optimize_intrinsics: bool = True
    optimize_time_offset: bool = True
    max_init_frames: int = 100
    init_frame_selection: str = "uniform"
    reference_camera: int | None = None
    rerun_outliers: bool = False


@dataclass
class CamImuCalibConfig:
    """Knobs for the joint cam-IMU solve (defaults per the reviewed plan)."""
    rms_gate_px: float = 0.0        # if > 0: final reproj rms above this marks
                                    # the result ok=False (diverged solves can
                                    # converge numerically with huge rms —
                                    # a field batch found rms 171 px with
                                    # ok=True). 0 = disabled (legacy behavior).
    state_hz: float = 30.0          # TRAJECTORY pose-state knot rate (Hz).
                                    # (0 = every accepted base-cam frame.) Keep
                                    # dense for chord-error accuracy
                                    # (sim-calibration-study). Default raised
                                    # 0.0 -> 30.0 for a denser td<->IMU fit.
    # Per-camera vision-FACTOR rate — same concept (and name) as the camera-only
    # CameraConfig.factor_hz. Default 10.0 (modern cam-IMU
    # default): each camera lays down ~10 vision factors/s. None -> fall back to
    # state_hz (set explicitly to use the dense trajectory rate). A per-camera
    # `factor_hz` in the cams dict overrides this; lets aux cameras use FEWER
    # vision factors without coarsening the shared trajectory. (YAML alias: target_hz)
    factor_hz: float | None = 10.0
    sigma_px: float = 1.0
    huber_px: float = 2.0           # 0 disables the robust kernel
    # IMU continuous-time noise (gyro, gyro walk, accel, accel walk)
    sigma_g: float = 0.01
    sigma_bg: float = 0.001
    sigma_a: float = 0.02
    sigma_ba: float = 0.002
    # IMU intrinsics staging: 'pinned' | 'dw_da' (gyro/accel scale+
    # misalignment free) | 'dw_da_r' (+ gyro-accel rotation) | 'full' (+ Tg).
    # Default 'dw_da' — estimate gyro/accel scale + misalignment (the stage every
    # shipped cam-IMU config already selects).
    imu_intrinsics: str = "dw_da"
    # IMU-intrinsic model: 'kalibr' (lower-tri Dw/Da + R_wtoI) | 'rpng' (upper-tri
    # Dw/Da + R_atoI) | 'calyx' (full 3x3 Dw, no rotation DOF, upper-tri Da). All
    # three share the 24-dim tangent; see mvis/imu. Default 'calyx'.
    imu_intrinsics_model: str = "calyx"
    # PnP solver for the init's pose track: 'iterative' (legacy) | 'ippe'
    # (planar-specific; resolves the planar two-fold ambiguity that makes
    # ITERATIVE flip basins / diverge to absurd poses — seen on Monado MOC).
    pnp_method: str = "iterative"
    sigma_dw_da: float = 0.1        # weak prior on freed Dw/Da components
    sigma_tg: float = 0.05          # weak prior on freed Tg components
    sigma_imu_R: float = 0.1        # weak prior on freed gyro-accel rotation
    estimate_readout: bool = False  # rolling shutter: estimate the readout tr
                                    # (False = global shutter; per-cam override)
    rolling_mode: int = 0           # RS row convention: 0 top / 1 mid / 2 bottom
    max_interp_dt: float = 0.3      # frames bracketed by a longer state drop
    # Dense pose knots (the chord-error fix). The
    # vision factors interpolate the pose between the two bracketing states with a
    # constant twist (SO(3) geodesic + linear position), so with states only at the
    # base-camera frames (ELP ~24 fps -> 41 ms) the corner poses carry a model error
    # ~ angular-acceleration x dt^2 that the free gyro scale / t_ga / td / tr absorb.
    # knot_hz > 0 subdivides every base-state interval (<= max_interp_dt) evenly so
    # no interval exceeds 1/knot_hz; the extra knots have no frame of their own and
    # are tied by the IMU preintegration. 0 = off (states only at base frames).
    knot_hz: float = 0.0
    # IMU-propagated vision factor. "interp"
    # = the constant-twist chord between the two bracketing states;
    # "imu" (default) = CameraCalibImuFactor: each corner's pose is the anchor state
    # propagated through the raw base-IMU samples to the corner's capture time
    # (td + rolling-shutter row), so no chord model error and no need for dense
    # knots. Board mode only. The velocity state and (with vision_imu_gyro
    # "calibrated") the IMU intrinsics enter the vision factors.
    vision_pose: str = "imu"
    vision_imu_gyro: str = "calibrated"   # "calibrated" (live Dw/Da/Tg) | "raw" (identity)
    vision_imu_dw_column: bool = True     # False: Dw in the residual, Jacobian column zeroed
    vision_imu_anchor: str = "lower"      # "lower" = X(j-1) | "nearest" bracketing state
    # True: the vision-IMU factor integrates once and corrects linearly, like an IMU factor; it
    # integrates again only when t_ga (tga_relin_threshold_s) or td moves far. False: at every evaluation.
    vision_imu_linearized: bool = True
    vision_imu_t_ga: bool = True          # with estimate_t_ga: the factor's accel uses the live t_ga (not with frame_dt)
    # Per-frame capture-time offset (E8 follow-up): > 0 gives every vision_pose=imu
    # frame its own time offset delta_f (s) with a zero-mean prior of this sigma,
    # absorbing per-frame capture jitter (ELP ~4 ms rms) that otherwise biases the
    # gyro scale through errors-in-variables. 0 = off.
    vision_imu_frame_dt_sigma: float = 0.0
    # Sensor-clock retiming (per camera: camera_input.cameras[i].retime_sensor_clock).
    # Pass-1 prior sigma (s) on the per-frame capture offsets of the flagged cameras;
    # wide on purpose — the offsets of a polled USB camera span a whole sensor period.
    sensor_clock_frame_dt_sigma: float = 0.02
    # Spatial-temporal "what to calibrate" pins (unified-estimate U3). All default
    # True = the quantity is estimated (the historical always-free behavior). Set
    # via the estimate: block; False adds a _TINY-sigma prior holding the quantity
    # at its seed. estimate_extrinsics/td act on the per-camera bundle (cam-IMU
    # T_C->I = sig_C[0:6]; cam td = sig_C[6]); estimate_aux_* act on each aux IMU
    # (T_IstoI = _As[0:6]; inter-IMU td = _As[6]).
    estimate_extrinsics: bool = True      # cam-IMU extrinsic T_C->I (per camera)
    estimate_td: bool = True              # camera <-> base-IMU time offset
    estimate_aux_extrinsics: bool = True  # aux-IMU -> base-IMU extrinsic T_IstoI
    estimate_aux_td: bool = True          # aux-IMU <-> base-IMU time offset
    # Board shape (since 2026-10-31): estimate one height per board
    # corner along the board normal, jointly with everything else (both vision
    # factors, single-board mode). Set via `estimate.board_points: true`. The
    # heights start from board.heights_file (or flat), carry a weak prior of
    # board_height_sigma (m) and a gauge (mean + two tilts = 0) so the board
    # pose keeps the plane.
    estimate_board_heights: bool = False
    board_height_sigma: float = 0.003
    # base-IMU internal gyro<->accel time offset t_ga (the accel applying at
    # gyro time t is a_m(t + t_ga)), estimated INSIDE the ACI factor via an
    # extra Jacobian column + scalar variable. Default off — for IMUs whose
    # driver interpolates gyro/accel from different clocks (e.g. t265 BMI055).
    estimate_t_ga: bool = True      # on by default since 2026-08-09
    sigma_t_ga: float = 0.010       # weak prior on t_ga (s) when estimated
    # OPTIONAL valid-domain bounds on free camera intrinsics (one-sided hinge
    # factor, zero inside the domain): double_sphere fx,fy > 0, xi in [-1, 1],
    # alpha in [0, 1]; omni_radtan fx,fy > 0, xi >= 0; kb4 fx,fy > 0. Off by default.
    bound_intrinsics: bool = False
    # > 0: re-integrate an IMU's factors at its current t_ga whenever t_ga is more than this far
    # (s) from the point they were last integrated at (0 = never, the linear term covers it all).
    # On by default since 2026-11-02 (the regression suite showed no regression).
    tga_relin_threshold_s: float = 0.003
    # > 0: an IMU slower than this (Hz) is upsampled before integration, real samples kept
    # (mvis.imu.resample). 0 = use the samples as they are.
    imu_min_rate_hz: float = 200.0
    imu_upsample_method: str = "cubic"   # "linear" | "cubic"
    aux_imu_factor_hz: float = 0.0  # rate of aux-IMU preintegration factors
                                    # (0 = one per base interval = state_hz). Set
                                    # lower (e.g. 10) to decimate aux factors+states
                                    # ~state_hz/this, for a faster multi-IMU solve.
    aux_imu_stride: int = 3         # aux-IMU factor decimation: 1 aux factor per
                                    # this many base intervals. Not accuracy-neutral:
                                    # levers change with it. Overrides aux_imu_factor_hz.
    # OPTIONAL per-camera prior anchoring td at its SEED value (0 = off). A
    # common-mode td slide is only weakly observable through the IMU terms; the
    # anchor pins it but locks the solve to the seed. Enable only when init td
    # is gated good and multi-cam common-mode drift matters more than seed
    # robustness. RELATIVE td between cameras is vision-determined regardless.
    sigma_td_prior: float = 0.0
    # per-tag corner thinning for the JOINT-GRAPH factors (mvis.board.
    # corner_subsample): "all" | "top_left"/"tl"/"bl"/"br"/"tr" | "two_diag"
    # | "two_anti". "top_left" keeps 1 corner/tag (~4x fewer rows). PnP + init
    # always use ALL corners. Only meaningful for AprilGrid-style ids.
    # Cam-IMU canonical; default "all" (accuracy) — intentionally differs from the
    # camera-only CalibrationConfig.corner_subsample default "two_anti" (speed).
    corner_subsample: str = "all"
    # |g| the solve holds fixed, m/s^2. Set from the run config's `gravity` / `site` (mvis.core.gravity);
    # 9.81 when neither is given. gravity_source: "default" | "config" | "site".
    gravity: float = 9.81
    gravity_source: str = "default"
    gravity_site: dict | None = None     # the config `site` + its g_site, for the report (mvis.core.gravity.site_record)
    gravity_place: str | None = None     # the place the comment after `gravity:` names (config_parser.gravity_place)
    # priors on the first state (gauge handles pose)
    sigma_v0: float = 0.5
    sigma_bg0: float = 0.01
    sigma_ba0: float = 0.1
    max_iterations: int = 50
    # LM damping bounds for the joint solve (cam-imu-speedup L1; see
    # OptimizerConfig.lm_lambda_*): lower 0 = GTSAM (bitwise-preserving), upper
    # 1e2 caps the retry loop. The termination tolerances stay at the GTSAM
    # defaults (1e-5 relative / absolute).
    lm_lambda_lower: float = 0.0
    lm_lambda_upper: float = 1e2
    # feature-mode depth-margin guard: skip corners whose landmark sits within
    # z_min of the camera principal plane (the 1/z grazing band). ADAPTIVE:
    # z_min = z_min_frac * median(per-camera seed depth), floored at z_min_floor.
    # z_min_frac=0 falls back to the factor's fixed default. 0.6 validated 17/17
    # on the ArUco batch.
    z_min_frac: float = 0.6
    z_min_floor: float = 0.05
    init_kwargs: dict = field(default_factory=dict)   # forwarded to the init

    @property
    def pin_tr(self) -> bool:
        """Back-compat alias: the readout is pinned iff we don't estimate it."""
        return not self.estimate_readout


@dataclass
class CamImuInputConfig:
    """Dataset input for a cam-IMU run (ROS1 or ROS2 bag, OR EuRoC mav0 dir)."""
    format: str = "ros1"            # "ros1" | "ros2" | "euroc"
    path: str = ""                  # bag file (ros1), bag folder / .db3 / .mcap (ros2) or mav0 dir (euroc)
    bags: list[str] = field(default_factory=list)   # OPTIONAL bag SET for a repeatability sweep:
                                    # one config calibrates each bag to <out>/bag<NN>/cam_imu.json
                                    # (discoverable by the repeatability study). Empty = use `path`.
    cam_topics: list[str] = field(default_factory=list)   # ros1; empty = from
                                    # camera_input intrinsics file (rostopic)
    imu_topics: list[str] = field(default_factory=lambda: ["/imu0"])
    t0: float = 3.0                 # window start offset (s)
    window: float = 0.0             # window length (s); 0 = whole dataset (DEFAULT:
                                    # use the whole trajectory — max excitation for
                                    # td/IMU-intrinsic observability. Raise it only to
                                    # cap solve cost; regression configs pin 45 explicitly)
    min_obs: int = 4                # min board corners to keep a frame
    cache_dir: str | None = None    # detection cache dir (None = no cache)
    window_mode: str = "motion"     # "motion" = auto-select segment_length by
                                    # IMU excitation; "manual" = use t0/window.
    segment_length: float = 80.0    # motion-mode window length (s)
    # OPTIONAL per-IMU overrides, matched BY ORDER to imu_topics. Each entry is a dict;
    # `gyro_only: true` calibrates that aux IMU from its GYRO alone (rotation + td + Dw +
    # gyro bias; lever/Da/Tg pinned) — for IMUs with an unreliable/interpolated accel
    # (e.g. T265 BMI055). Not allowed on the base (imu_topics[0]); omit = full mode.
    imus: list = field(default_factory=list)


@dataclass
class CamImuCameraInputConfig:
    """Prior camera intrinsics input (MVIS *_calib.yaml OR Kalibr camchain)."""
    path: str = ""                  # path to the intrinsics file
    free_intrinsics: bool = True    # DEFAULT: estimate camera intrinsics jointly
                                    # (the provided file seeds them). Set False to
                                    # PIN — recommended for omni/fisheye where the
                                    # fx–ξ gauge drifts when freed (Phase 35), or to
                                    # trust a high-quality cam-only prior. Regression
                                    # configs keep this default (enforced by
                                    # tests/regression/test_regression_tiers.py since 2026-10-18).
    estimate_readout: bool = False  # rolling shutter: estimate the readout tr
                                    # (False = global shutter)
    rolling_mode: int = 0           # RS row convention (per-cam, if estimating tr)
    # OPTIONAL per-camera overrides, matched BY ORDER to input.cam_topics (and the
    # intrinsics-file cameras). Each entry is a dict that may set: factor_hz (the
    # per-camera vision-FACTOR rate — e.g. base cam at the trajectory rate, aux cams
    # sparser), estimate_readout, rolling_mode, free_intrinsics, retime_sensor_clock
    # (stamps not on the sensor clock -> two-pass sensor-clock retiming, needs
    # the IMU-propagated vision factor on that camera; the fitted stamp map is saved to
    # `<output.path stem>_sensor_clock.json`), retime_map (path to such a sidecar:
    # re-stamps the camera directly, no pass 1), vision_pose ("interp" | "imu": this
    # camera's vision factor, overriding solve.vision_pose — e.g. imu for rolling-
    # shutter cameras, interp for global-shutter ones). Keys omitted on an entry fall back
    # to the block-level defaults above / solve.factor_hz / state_hz. Entries beyond
    # the camera count are ignored; missing entries use the defaults.
    cameras: list = field(default_factory=list)

    @property
    def pin_tr(self) -> bool:
        """Back-compat alias: the readout is pinned iff we don't estimate it."""
        return not self.estimate_readout


@dataclass
class CamImuOutputConfig:
    """Where to write the cam-IMU calibration."""
    path: str = "cam_imu_calib.json"        # summary JSON
    calib_yaml: str | None = None           # complete MVIS _calib.yaml
    kalibr_yaml: str | None = None          # Kalibr camchain-imucam export
    report_html: str | None = None          # analysis report (HTML)
    report_txt: str | None = None           # analysis report (text)


@dataclass
class CamImuRunConfig:
    """Top-level cam-IMU run config — drives one calibration from one YAML."""
    input: CamImuInputConfig = field(default_factory=CamImuInputConfig)
    board: BoardConfig = field(default_factory=BoardConfig)
    camera_input: CamImuCameraInputConfig = field(
        default_factory=CamImuCameraInputConfig)
    solve: CamImuCalibConfig = field(default_factory=CamImuCalibConfig)
    output: CamImuOutputConfig = field(default_factory=CamImuOutputConfig)
    # Unified "what to calibrate" switchboard (unified-estimate).
    estimate: "EstimateConfig" = field(default_factory=EstimateConfig)
    # optional published/known values for the report's vs-reference section
    # ({label, baseline_mm, lever_mm{cam:[..]}, rotation_rad{cam:..}}; rotation_deg is still read).
    reference: dict = field(default_factory=dict)
    # Local gravity (mvis.core.gravity): `gravity` in m/s^2, or `site: {latitude_deg, altitude_m}`.
    # Neither: 9.81. Resolved into solve.gravity by the parser.
    gravity: float | None = None
    site: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "CamImuRunConfig":
        """Parse a cam-IMU run YAML (nested sections, unknown keys warned)."""
        from mvis.core.config_parser import parse_cam_imu_config
        return parse_cam_imu_config(path)
