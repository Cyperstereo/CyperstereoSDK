# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Rosbag2 errors."""

from __future__ import annotations


class ReaderError(Exception):
    """Reader Error."""

    args: tuple[str]


class WriterError(Exception):
    """Writer Error."""

    args: tuple[str]
