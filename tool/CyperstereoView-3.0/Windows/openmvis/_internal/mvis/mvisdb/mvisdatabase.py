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
"""MVISDatabase — THE designed structure, in one file.

The whole thing for ONE run: the live object a solve mutates, and the loaded
object ``manager.load(run_id)`` returns for a stored run. Measurement fields are
written once by ingest (numpy arrays are made non-writeable); the run FLAGS live
INLINE on the tree and are the mutable working state of the run. Multi-run
is a DISK concept only — the overlay tables are how a run sleeps beside other
runs; this structure never holds two runs.

Pure Python + numpy. sqlite3 MUST NEVER be imported here —
persistence lives entirely in ``mvis.mvisdb.manager``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

import numpy as np

# ---------------------------------------------------------------------------
# Enums — StrEnum -> TEXT + CHECK in the schema; Corner is the sole IntEnum.
# ---------------------------------------------------------------------------


class SensorType(StrEnum):
    """Polymorphic discriminator: which id space an integer belongs to."""
    CAM = "cam"
    IMU = "imu"
    BOARD = "board"   # state_var only
    WORLD = "world"   # state_var only


class CameraModel(StrEnum):
    """THE model — one enum, exactly eight values. projection/distortion
    pair is a derived VIEW for Kalibr export, never stored (kb4's is pinhole-equi)."""
    RADTAN = "radtan"
    EQUIDISTANT = "equidistant"
    FOV = "fov"
    DOUBLE_SPHERE = "double_sphere"
    EUCM = "eucm"
    UCM = "ucm"
    OMNI_RADTAN = "omni_radtan"
    KB4 = "kb4"

    @classmethod
    def _missing_(cls, value):
        # an old file naming a removed model (kb8): say so, not "not a valid CameraModel"
        from mvis.core.camera_models import is_removed_model, removed_model_message
        if is_removed_model(value):
            raise ValueError(removed_model_message(value))
        return None


class CalibChoice(StrEnum):
    """Which calibration the run solves."""
    CAM_ONLY = "cam_only"
    CAM_IMU = "cam_imu"


class RunOrigin(StrEnum):
    """Where the run's estimate came from."""
    SOLVED = "solved"
    GROUND_TRUTH = "ground_truth"
    IMPORTED = "imported"


class DetectionMode(StrEnum):
    """Composable capture modes: board = complete known layout;
    tag = ID'd corners, no coplanarity assumed; free = anonymous tracks."""
    BOARD = "board"
    TAG = "tag"
    FREE = "free"


class BoardType(StrEnum):
    APRILGRID = "aprilgrid"
    ARUCO_GRID = "aruco_grid"
    CHESSBOARD = "chessboard"


class TagFamily(StrEnum):
    T36H11 = "t36h11"
    T36H11_KALIBR = "t36h11_kalibr"   # is_kalibr_tag derives from this
    T25H9 = "t25h9"
    ARUCO_4X4_50 = "aruco_4x4_50"
    ARUCO_5X5_100 = "aruco_5x5_100"
    ARUCO_6X6_250 = "aruco_6x6_250"


class Corner(IntEnum):
    """The sole IntEnum — the number IS the fact (corner order inside a tag).
    NULL/None on a chessboard (corners_per_tag == 1)."""
    BL = 0
    BR = 1
    TR = 2
    TL = 3


class ShutterType(StrEnum):
    GLOBAL = "global"
    ROLLING = "rolling"
    UNKNOWN = "unknown"


class SourceType(StrEnum):
    IMAGES = "images"
    TIMESTAMPED_IMAGES = "timestamped_images"
    EUROC = "euroc"
    ROS1BAG = "ros1bag"
    ROS2BAG = "ros2bag"


class ClockKind(StrEnum):
    """Which clock timestamps come from — recorded, never implied."""
    HEADER = "header"
    BAG = "bag"
    FILENAME_NS = "filename_ns"
    FRAME_INDEX = "frame_index"


class CalibProvenance(StrEnum):
    """How a calibration value came to be:
    'provided and pinned' is a real third state, a pinned value is not
    evidence, and truth can never be confused with an estimate."""
    UNKNOWN = "unknown"
    PRIOR_FIXED = "prior_fixed"
    PRIOR_INITIALIZED = "prior_initialized"
    ESTIMATED = "estimated"
    GROUND_TRUTH = "ground_truth"


class PoseSource(StrEnum):
    """Provenance of a per-frame pose. MONOTONIC ladder within a run:
    pnp < refined_per_camera < (refined_cam_only | refined_cam_imu).
    The db writer refuses any downgrade."""
    PNP = "pnp"
    REFINED_PER_CAMERA = "refined_per_camera"
    REFINED_CAM_ONLY = "refined_cam_only"
    REFINED_CAM_IMU = "refined_cam_imu"

    @property
    def rank(self) -> int:
        return _POSE_RANK[self]


_POSE_RANK = {
    PoseSource.PNP: 0,
    PoseSource.REFINED_PER_CAMERA: 1,
    PoseSource.REFINED_CAM_ONLY: 2,
    PoseSource.REFINED_CAM_IMU: 2,
}


class ExclusionKind(StrEnum):
    """validity = the data is bad; selection = the run chose not to."""
    VALIDITY = "validity"
    SELECTION = "selection"


class ImuIntrinsicsModel(StrEnum):
    """xi24 layout convention: kalibr 6|6|9|3, rpng 6|6|9|3 with the R
    direction inverted, calyx 9|6|9 with no R."""
    KALIBR = "kalibr"
    RPNG = "rpng"
    CALYX = "calyx"


class VarType(StrEnum):
    """Which state variable a state_var blob holds."""
    T_IMU_TO_WORLD = "T_imu_to_world"
    V = "v"
    B_G = "b_g"
    B_A = "b_a"
    T_BOARD_TO_WORLD = "T_board_to_world"
    LANDMARK_XYZ = "landmark_xyz"
    T_CAM_TO_BOARD = "T_cam_to_board"      # cam-only base state


class PolicyState(StrEnum):
    """One knob vocabulary for the whole solve."""
    FREE = "free"
    ANCHOR = "anchor"
    FIX = "fix"


# ---------------------------------------------------------------------------
# Small value types
# ---------------------------------------------------------------------------


def _freeze(a: np.ndarray | None) -> np.ndarray | None:
    """Measurement arrays are written once — make that physical."""
    if a is None:
        return None
    a = np.asarray(a)
    a.flags.writeable = False
    return a


@dataclass(slots=True)
class PoseCamToBoard:
    """Camera pose w.r.t. ONE board: T_cam_to_board . p_cam = p_board.
    A pose travels with what produced it — never the T alone."""
    T: np.ndarray                       # (4,4)
    source: PoseSource


@dataclass(slots=True)
class Residual:
    """A reprojection error is only meaningful against the pose it was computed
    from: residual_source is ALWAYS the enclosing group's pose source."""
    residual_px: float
    residual_source: PoseSource


@dataclass(slots=True)
class RelativeCalib:
    """One sensor relative to the run's reference stream."""
    T_x_to_ref: np.ndarray              # (4,4), T_<from>_to_<to>
    td_x_to_ref_s: float | None = None  # inter-camera td is LIVE
    sigma6: np.ndarray | None = None
    provenance: CalibProvenance = CalibProvenance.UNKNOWN


@dataclass(slots=True)
class PolicyEntry:
    """free | anchor(sigma) | fix — fix requires provenance != unknown."""
    state: PolicyState = PolicyState.FREE
    sigma: float | None = None          # meaningful only for ANCHOR


@dataclass(slots=True)
class SensorPolicy:
    """Per-sensor solve policy on the Run — the single knob source."""
    intrinsics: PolicyEntry = field(default_factory=PolicyEntry)
    readout: PolicyEntry = field(default_factory=PolicyEntry)
    extrinsics: PolicyEntry = field(default_factory=PolicyEntry)
    td: PolicyEntry = field(default_factory=PolicyEntry)


# ---------------------------------------------------------------------------
# Boards — pure geometry (no detect() here, ever)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BoardPt:
    """board_pt_id = tag_id * corners_per_tag + (corner_id or 0)."""
    board_pt_id: int
    tag_id: int
    corner_id: Corner | None            # None on a chessboard
    board_pt_3d: np.ndarray             # (3,), metres, board frame

    def __post_init__(self) -> None:
        self.board_pt_3d = _freeze(np.asarray(self.board_pt_3d, float))


@dataclass(slots=True)
class Board:
    """The physical target. tag_id is board-LOCAL (detector subtracts start_id)."""
    board_id: int
    board_type: BoardType
    rows: int
    cols: int
    spacing: float                      # metres, RESOLVED (never a ratio)
    size: float                         # tag/square size, metres
    start_id: int = 0
    tag_family: TagFamily | None = None  # None on a chessboard
    pts: dict[int, BoardPt] = field(default_factory=dict)

    @property
    def corners_per_tag(self) -> int:
        return 1 if self.board_type is BoardType.CHESSBOARD else 4

    @property
    def is_kalibr_tag(self) -> bool:
        return self.tag_family is TagFamily.T36H11_KALIBR

    def point_3d(self, board_pt_id: int) -> np.ndarray:
        return self.pts[board_pt_id].board_pt_3d


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CameraInfo:
    """Values known BEFORE the solve; a run's answer lives in estimate_states.
    Every calibration field is tri-state: None = unknown/empty."""
    name: str
    topic: str | None = None
    shutter: ShutterType = ShutterType.UNKNOWN
    readout_s: float | None = None
    rolling_mode: int | None = None     # 0 top / 1 mid / 2 bottom
    intrinsics: np.ndarray | None = None    # (4,) fx fy cx cy
    distortion: np.ndarray | None = None    # (N,), N = model param count - 4
    camera_model: CameraModel | None = None
    resolution: tuple[int, int] | None = None
    calib_provenance: CalibProvenance = CalibProvenance.UNKNOWN

    @property
    def is_fisheye(self) -> bool:
        """Correct over the eight ids — omni_radtan is its own id."""
        return self.camera_model is not None and self.camera_model is not CameraModel.RADTAN


@dataclass(slots=True)
class ImuInfo:
    name: str
    topic: str | None = None
    sigma_g: float | None = None
    sigma_bg: float | None = None
    sigma_a: float | None = None
    sigma_ba: float | None = None
    frequency_hz: float | None = None
    xi24: np.ndarray | None = None      # (24,)
    intrinsics_model: ImuIntrinsicsModel | None = None
    calib_provenance: CalibProvenance = CalibProvenance.UNKNOWN


@dataclass(slots=True)
class ImuMeasurements:
    """Written once by ingest; single shared timestamp per row (loader finalizes)."""
    t: np.ndarray                       # (N,)
    gyro: np.ndarray                    # (N,3)
    accel: np.ndarray                   # (N,3)

    def __post_init__(self) -> None:
        self.t = _freeze(np.asarray(self.t, float))
        self.gyro = _freeze(np.asarray(self.gyro, float))
        self.accel = _freeze(np.asarray(self.accel, float))


@dataclass(slots=True)
class Rig:
    """Reference gauge + pairing. reference_stream is DERIVED per run."""
    base_camera_id: int = 0
    base_imu_id: int | None = None
    stereo_pairs: list[tuple[int, int]] = field(default_factory=list)
    sync_kind: str | None = None
    sync_tolerance_s: float | None = None

    def stereo_partner(self, camera_id: int) -> int | None:
        for a, b in self.stereo_pairs:
            if a == camera_id:
                return b
            if b == camera_id:
                return a
        return None


# ---------------------------------------------------------------------------
# The frame tree — measurements frozen, run flags INLINE and mutable
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TagDetection:
    """One detected corner of a known board point (~1.7M per large capture).
    Measured: uv, point_id, detector_emission_order (bit-identity replay, has
    an expiry — D-emission). Run flags: is_valid / is_selected / reason /
    residual. Everything else is derived through the board reference."""
    uv: np.ndarray                      # (2,) float64, exact
    point_id: int
    detector_emission_order: int = 0
    # containment (set by the builder)
    frame_id: int = 0
    board_id: int = 0
    board: Board | None = field(default=None, repr=False, compare=False)
    # --- run flags, inline ---
    is_valid: bool = True
    is_selected: bool = True
    exclude_reason: str | None = None
    residual: Residual | None = None

    def __post_init__(self) -> None:
        self.uv = _freeze(np.asarray(self.uv, float).reshape(2))

    # --- derived (never stored — the rule with no exceptions) ---
    @property
    def row(self) -> int:
        return int(self.uv[1])          # truncation

    @property
    def tag_id(self) -> int:
        cpt = self.board.corners_per_tag if self.board else 4
        return self.point_id // cpt

    @property
    def corner_id(self) -> Corner | None:
        cpt = self.board.corners_per_tag if self.board else 4
        return None if cpt == 1 else Corner(self.point_id % cpt)

    @property
    def point_3d(self) -> np.ndarray:
        if self.board is None:
            raise ValueError("detection has no board reference wired")
        return self.board.point_3d(self.point_id)


@dataclass(slots=True)
class BoardDetection:
    """Everything this frame saw OF ONE BOARD — the pose is per board."""
    board_id: int
    frame_id: int = 0
    detections: list[TagDetection] = field(default_factory=list)
    board: Board | None = field(default=None, repr=False, compare=False)
    frame: "FrameData | None" = field(default=None, repr=False, compare=False)
    # --- run fields, inline ---
    pose_cam_to_board: PoseCamToBoard | None = None
    is_valid_to_use: bool = True
    is_selected: bool = True            # this group's OWN selection bit

    # --- derived ---
    @property
    def n_detections(self) -> int:
        return len(self.detections)

    @property
    def n_tags(self) -> int:
        cpt = self.board.corners_per_tag if self.board else 4
        return len({d.point_id // cpt for d in self.detections})

    @property
    def detect_status(self) -> str:
        return "no_detection" if not self.detections else "ok"

    @property
    def is_selected_to_use(self) -> bool:
        """Selection flows DOWN: a frame the run did not select cannot
        have a selected group."""
        frame_sel = self.frame.is_selected_to_use if self.frame is not None else True
        return frame_sel and self.is_selected

    def set_pose(self, T: np.ndarray, source: PoseSource) -> None:
        """The monotonic PoseSource ladder, enforced at the object."""
        if self.pose_cam_to_board is not None and source.rank < self.pose_cam_to_board.source.rank:
            raise ValueError(
                f"pose downgrade {self.pose_cam_to_board.source} -> {source} "
                "is a bug by construction")
        self.pose_cam_to_board = PoseCamToBoard(np.asarray(T, float), source)


@dataclass(slots=True)
class FrameData:
    """One image. Detections GROUPED BY TARGET; tag/free lists are the
    deferred modes' homes."""
    timestamp: float
    frame_id: int = 0
    camera_id: int = 0
    board_detections: list[BoardDetection] = field(default_factory=list)
    tag_feature_detections: list = field(default_factory=list)    # no schema home yet
    free_feature_detections: list = field(default_factory=list)   # no schema home yet
    # --- run flags, inline ---
    is_motion_still: bool | None = None     # ships None: no producer yet
    is_selected_to_use: bool = True
    exclude_reason: str | None = None

    @property
    def is_valid_to_use(self) -> bool:
        """Validity flows UP: any usable group makes the frame usable;
        a frame with NO groups gives any([]) == False for free."""
        return (any(g.is_valid_to_use for g in self.board_detections)
                or any(getattr(g, "is_valid_to_use", False)
                       for g in self.tag_feature_detections)
                or any(getattr(g, "is_valid_to_use", False)
                       for g in self.free_feature_detections))


@dataclass(slots=True)
class CameraStream:
    """camera info + a vector of frame data."""
    camera_id: int
    info: CameraInfo
    frames: list[FrameData] = field(default_factory=list)


@dataclass(slots=True)
class ImuStream:
    imu_id: int
    info: ImuInfo
    measurements: ImuMeasurements | None = None


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Run:
    """One experiment on this bag. run_id is None until attached to disk."""
    calib_choice: CalibChoice
    run_id: int | None = None
    origin: RunOrigin = RunOrigin.SOLVED
    description: str = ""
    solve_mode: str | None = None
    variant: str | None = None
    collection: str | None = None
    state_hz: float | None = None
    window: tuple[float, float] | None = None
    min_obs: int | None = None
    config: dict = field(default_factory=dict)
    config_sha256: str | None = None
    mvis_commit: str | None = None
    machine_id: int | None = None
    status: str = "running"
    started_at: float = 0.0
    wall_s: float | None = None
    residual_promote_px: float | None = None
    sensor_policy: dict[tuple[SensorType, int], SensorPolicy] = field(default_factory=dict)
    factor_hz: dict[tuple[SensorType, int], float] = field(default_factory=dict)

    def policy(self, sensor: tuple[SensorType, int]) -> SensorPolicy:
        return self.sensor_policy.setdefault(sensor, SensorPolicy())


# ---------------------------------------------------------------------------
# The estimate side — a projection shape, nothing stored like this
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CameraIntrinsicsState:
    intrinsics: np.ndarray              # (4,) fx fy cx cy
    distortion: np.ndarray              # (N,)
    camera_model: CameraModel
    readout_s: float | None = None
    provenance: CalibProvenance = CalibProvenance.PRIOR_INITIALIZED
    sigmas: np.ndarray | None = None    # marginals — the cam-IMU anchors read these


@dataclass(slots=True)
class ImuIntrinsicsState:
    xi24: np.ndarray                    # (24,)
    model: ImuIntrinsicsModel
    provenance: CalibProvenance = CalibProvenance.PRIOR_INITIALIZED
    sigmas: np.ndarray | None = None


@dataclass(slots=True)
class ReprojStats:
    """Per-sensor reprojection summary (schema v3) — solver output the
    calib-YAML export view needs; all optional."""
    rms: float | None = None
    mean: float | None = None
    median: float | None = None
    p90: float | None = None
    p95: float | None = None
    n_obs: int | None = None


@dataclass(slots=True)
class SensorState:
    """Everything the run knows about ONE sensor. For the reference
    stream: factor_hz is None (its rate is state_hz) and relative is None
    (it IS the frame the others are relative to)."""
    sensor: tuple[SensorType, int]
    intrinsics: CameraIntrinsicsState | ImuIntrinsicsState
    factor_hz: float | None = None
    relative: RelativeCalib | None = None
    reproj: ReprojStats | None = None


@dataclass(slots=True)
class CamOnlyBaseState:
    """Cam-only trajectory: the reference camera w.r.t. the BOARD — no world
    frame exists in cam-only."""
    t: np.ndarray                       # (F,)
    T_cam_to_board: np.ndarray          # (F,7) qw qx qy qz tx ty tz


@dataclass(slots=True)
class CamImuBaseState:
    t: np.ndarray                       # (K,)
    T_imu_to_world: np.ndarray          # (K,7)
    v: np.ndarray                       # (K,3)
    b_g: np.ndarray                     # (K,3)
    b_a: np.ndarray                     # (K,3)


@dataclass(slots=True)
class WorldFrame:
    """The gauge record that makes trajectories comparable across runs."""
    kind: str = "gravity_aligned"
    g_magnitude: float = 9.80665
    yaw_gauge: float | None = None


@dataclass(slots=True)
class EstimateStates:
    """The estimate side of the run — created at the end of Phase A
    (intrinsics/readout), completed by Phase B."""
    reference_stream: tuple[SensorType, int]
    sensor_states: list[SensorState] = field(default_factory=list)
    state_hz: float | None = None
    base_state: CamOnlyBaseState | CamImuBaseState | None = None
    boards: dict[int, np.ndarray] | None = None     # board_id -> T_board_to_world (7,)
    world_frame: WorldFrame | None = None
    overall_rms: float | None = None                # schema v3: headline stat

    def sensor(self, sensor: tuple[SensorType, int]) -> SensorState | None:
        for s in self.sensor_states:
            if s.sensor == sensor:
                return s
        return None


# ---------------------------------------------------------------------------
# MVISDatabase — the container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MVISDatabase:
    """streams + boards + imus + rig + THE run + its estimate."""
    name: str
    rig: Rig = field(default_factory=Rig)
    streams: list[CameraStream] = field(default_factory=list)
    boards: dict[int, Board] = field(default_factory=dict)
    imus: list[ImuStream] = field(default_factory=list)
    run: Run = field(default_factory=lambda: Run(CalibChoice.CAM_ONLY))
    estimate_states: EstimateStates | None = None
    _next_frame_id: int = field(default=0, repr=False, compare=False)

    # --- navigation -------------------------------------------------------
    def camera(self, camera_id: int) -> CameraStream:
        for s in self.streams:
            if s.camera_id == camera_id:
                return s
        raise KeyError(f"no camera stream {camera_id}")

    def imu(self, imu_id: int) -> ImuStream:
        for s in self.imus:
            if s.imu_id == imu_id:
                return s
        raise KeyError(f"no imu stream {imu_id}")

    @property
    def reference_stream(self) -> tuple[SensorType, int]:
        """Keyed on the RUN's calib_choice, NOT on the rig having an IMU:
        a rig with IMUs can still be solved camera-only."""
        if self.run.calib_choice is CalibChoice.CAM_IMU:
            if self.rig.base_imu_id is None:
                raise ValueError("cam_imu run but rig.base_imu_id is None")
            return (SensorType.IMU, self.rig.base_imu_id)
        return (SensorType.CAM, self.rig.base_camera_id)

    # --- building (ingest / simulator only touch measured fields) ---------
    def add_frame(self, camera_id: int, timestamp: float) -> FrameData:
        f = FrameData(timestamp=float(timestamp), frame_id=self._next_frame_id,
                      camera_id=camera_id)
        self._next_frame_id += 1
        self.camera(camera_id).frames.append(f)
        return f

    def add_board_detection(self, frame: FrameData, board_id: int,
                            detections: list[TagDetection]) -> BoardDetection:
        board = self.boards[board_id]
        g = BoardDetection(board_id=board_id, frame_id=frame.frame_id,
                           detections=detections, board=board, frame=frame)
        for d in detections:
            d.frame_id = frame.frame_id
            d.board_id = board_id
            d.board = board
        frame.board_detections.append(g)
        return g

    # --- integrity --------------------------------------------------------
    def validate(self) -> None:
        """The invariants nesting buys (D-semantics): every sensor referenced
        by the estimate exists; a GROUND_TRUTH run is internally consistent."""
        cam_ids = {s.camera_id for s in self.streams}
        imu_ids = {s.imu_id for s in self.imus}
        if self.estimate_states is not None:
            refs = [self.estimate_states.reference_stream] + [
                s.sensor for s in self.estimate_states.sensor_states]
            for kind, sid in refs:
                pool = cam_ids if kind is SensorType.CAM else imu_ids
                if kind in (SensorType.CAM, SensorType.IMU) and sid not in pool:
                    raise ValueError(f"estimate references unknown sensor ({kind}, {sid})")
            if self.estimate_states.reference_stream != self.reference_stream:
                raise ValueError("estimate reference_stream disagrees with run/rig")
        for sensor, pol in self.run.sensor_policy.items():
            for aspect in (pol.intrinsics, pol.readout, pol.extrinsics, pol.td):
                if aspect.state is PolicyState.ANCHOR and aspect.sigma is None:
                    raise ValueError(f"anchor policy without sigma on {sensor}")
        for s in self.streams:
            ts = [f.timestamp for f in s.frames]
            if any(b < a for a, b in zip(ts, ts[1:])):
                raise ValueError(f"camera {s.camera_id} frames not time-ordered")


# ---------------------------------------------------------------------------
# Columnar fast path + the identity harness (they live with the structure)
# ---------------------------------------------------------------------------


def frames_to_arrays(db: MVISDatabase, camera_id: int, *,
                     selected_only: bool = True,
                     valid_only: bool = True) -> dict[str, np.ndarray]:
    """The FAST read shape: flat numpy over one camera's usable detections.
    Solvers consume this; the dataclass tree is the browse path."""
    fi, t, pid, bid, uv, row = [], [], [], [], [], []
    for frame_index, f in enumerate(db.camera(camera_id).frames):
        if selected_only and not f.is_selected_to_use:
            continue
        for g in f.board_detections:
            if selected_only and not g.is_selected_to_use:
                continue
            if valid_only and not g.is_valid_to_use:
                continue
            for d in g.detections:
                if valid_only and not d.is_valid:
                    continue
                if selected_only and not d.is_selected:
                    continue
                fi.append(frame_index)
                t.append(f.timestamp)
                pid.append(d.point_id)
                bid.append(g.board_id)
                uv.append(d.uv)
                row.append(d.row)
    return {
        "frame_index": np.asarray(fi, np.int64),
        "timestamp": np.asarray(t, float),
        "point_id": np.asarray(pid, np.int64),
        "board_id": np.asarray(bid, np.int64),
        "uv": (np.asarray(uv, float).reshape(-1, 2) if uv
               else np.empty((0, 2), float)),
        "row": np.asarray(row, np.int64),
    }


def observation_fingerprint(db: MVISDatabase) -> str:
    """SHA256 over (camera_id, frame_index, point_id, u, v, row) as raw IEEE754
    bits, rows sorted — insertion order never changes it, one uv bit does.
    This DEFINES 'bit-identical' for the migration checks."""
    rows: list[tuple] = []
    for s in db.streams:
        for frame_index, f in enumerate(s.frames):
            for g in f.board_detections:
                for d in g.detections:
                    rows.append((s.camera_id, frame_index, d.point_id,
                                 float(d.uv[0]), float(d.uv[1]), d.row))
    rows.sort()
    h = hashlib.sha256()
    for cid, fi, pid, u, v, r in rows:
        h.update(np.int64(cid).tobytes())
        h.update(np.int64(fi).tobytes())
        h.update(np.int64(pid).tobytes())
        h.update(np.float64(u).tobytes())
        h.update(np.float64(v).tobytes())
        h.update(np.int64(r).tobytes())
    return h.hexdigest()
