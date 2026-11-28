# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Rosbags message serialization and deserialization.

Serializers and deserializers convert between python messages objects and
the common rosbag serialization formats. Computationally cheap functions
convert directly between different serialization formats.

"""

from .errors import SerdeError

__all__ = [
    'SerdeError',
]
