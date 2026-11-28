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
"""CaptureSpec — THE identity of a stored capture.

identity = hash(source block + detection block). Change any field and you have
measured a different thing: a different ``.mvisdb`` opens. Run policy (window,
hz, solver knobs) is deliberately ABSENT; runtime knobs (threads,
verbosity) are plain function arguments, never stored, never hashed.

Pure Python — no sqlite here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from mvis.mvisdb.mvisdatabase import (
    BoardType, ClockKind, DetectionMode, SourceType, TagFamily,
)

__all__ = [
    "BoardSpec", "BoardDetectorConfig", "CameraAddress", "CaptureSpec",
    "FreeFeatureConfig", "ImuAddress", "SourceConfig", "TagDetectorConfig",
]


# --- source block: io's config (where the data is, how to address it) --------


@dataclass(slots=True, frozen=True)
class CameraAddress:
    """camera_id is MINTED here — the id<->topic binding is explicit config,
    killing the positional enumerate() where 'camera 0' differed per loader."""
    camera_id: int
    topic: str | None = None            # or subdir for image sources
    name: str = ""


@dataclass(slots=True, frozen=True)
class ImuAddress:
    imu_id: int
    topic: str | None = None
    name: str = ""


@dataclass(slots=True, frozen=True)
class SourceConfig:
    type: SourceType
    path: str
    clock: ClockKind
    cameras: tuple[CameraAddress, ...] = ()
    imus: tuple[ImuAddress, ...] = ()
    # identity includes WHAT was read, not just where from:
    source_mtime_ns: int = 0
    source_size: int = 0


# --- detection block: one config per enabled detector (a union) -------------


@dataclass(slots=True, frozen=True)
class BoardSpec:
    family: TagFamily | None
    board_type: BoardType
    rows: int
    cols: int
    spacing: float                      # metres, resolved
    size: float
    start_id: int = 0


@dataclass(slots=True, frozen=True)
class BoardDetectorConfig:
    boards: tuple[BoardSpec, ...]
    refine_subpixel: bool = True
    min_tag_px: int = 0


@dataclass(slots=True, frozen=True)
class TagDetectorConfig:
    """No layout fields — a tag config with rows/cols is unrepresentable."""
    family: TagFamily
    id_range: tuple[int, int] | None = None
    min_tag_px: int = 0


@dataclass(slots=True, frozen=True)
class FreeFeatureConfig:
    max_features: int = 300
    quality: float = 0.01
    klt_window: int = 21
    min_distance: int = 15


_DETECTOR_TYPES = (BoardDetectorConfig, TagDetectorConfig, FreeFeatureConfig)


# --- the spec ----------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CaptureSpec:
    """hash(source + detection). capture_mode is DERIVED = detection.keys()."""
    source: SourceConfig
    detection: dict[DetectionMode, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for mode, cfg in self.detection.items():
            if not isinstance(cfg, _DETECTOR_TYPES):
                raise TypeError(f"detection[{mode}] is not a detector config: {type(cfg)}")

    @property
    def capture_mode(self) -> frozenset[DetectionMode]:
        return frozenset(self.detection.keys())

    def identity_hash(self) -> str:
        """Canonical, field-order-independent SHA256. EVERY field flips it —
        that is a tested property, not an aspiration (tests/test_mvisdb_spec)."""
        payload = {
            "source": asdict(self.source),
            "detection": {
                str(mode): {"__type__": type(cfg).__name__, **asdict(cfg)}
                for mode, cfg in sorted(self.detection.items())
            },
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()
