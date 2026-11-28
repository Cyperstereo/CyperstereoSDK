# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Rosbags support for rosbag2 files.

Readers and writers provide access to metadata and raw message content saved
in the rosbag2 format.

"""

from .enums import CompressionFormat, CompressionMode, StoragePlugin
from .errors import ReaderError, WriterError
from .reader import Reader
from .writer import Writer

__all__ = [
    'CompressionFormat',
    'CompressionMode',
    'Reader',
    'ReaderError',
    'StoragePlugin',
    'Writer',
    'WriterError',
]
