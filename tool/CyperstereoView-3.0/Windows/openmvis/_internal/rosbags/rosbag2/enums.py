# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Rosbag2 enums."""

from __future__ import annotations

from enum import IntEnum, auto


class StoragePlugin(IntEnum):
    """Storage Plugins."""

    SQLITE3 = auto()
    MCAP = auto()


class CompressionMode(IntEnum):
    """Compession modes."""

    NONE = auto()
    FILE = auto()
    MESSAGE = auto()
    STORAGE = auto()


class CompressionFormat(IntEnum):
    """Compession formats."""

    ZSTD = auto()
