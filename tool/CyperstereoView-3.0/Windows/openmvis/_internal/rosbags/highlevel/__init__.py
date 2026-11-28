# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Highlevel interfaces for rosbags."""

from .anyreader import AnyReader, AnyReaderError

__all__ = [
    'AnyReader',
    'AnyReaderError',
]
