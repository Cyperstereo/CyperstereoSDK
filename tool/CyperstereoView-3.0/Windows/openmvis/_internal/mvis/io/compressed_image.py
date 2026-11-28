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
"""sensor_msgs/CompressedImage -> the fields of a raw sensor_msgs/Image.

`image_decode.ros_image_to_gray` calls `as_raw_image` before its raw decode, so a PNG frame goes
through the same rules as the raw frame it came from, 16-bit scaling included. A raw Image passes
through untouched.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import cv2
import numpy as np

logger = logging.getLogger("mvis.io.compressed_image")

_MAGIC = {"png": b"\x89PNG\r\n\x1a\n", "jpeg": b"\xff\xd8\xff"}


def compressed_codec(msg) -> str:
    """'png' or 'jpeg', from the bytes (`format` is free text and drivers get it wrong).

    ValueError for anything else: compressedDepth (a private header ahead of the PNG), video
    codecs, or data that is not what `format` says."""
    fmt = str(msg.format)
    name = fmt.lower()
    if "compresseddepth" not in name:
        head = np.frombuffer(msg.data, np.uint8)[:8].tobytes()
        for codec, magic in _MAGIC.items():
            if head.startswith(magic):
                return codec
        if "png" in name or "jpeg" in name or "jpg" in name:
            raise ValueError(f"unreadable compressed image: format {fmt!r} but the data is "
                             f"not PNG or JPEG ({np.size(msg.data)} bytes)")
    raise ValueError(f"unsupported compressed image format {fmt!r}: MVIS reads PNG and JPEG")


def as_raw_image(msg, topic: str, warned: set):
    """`msg` itself if it is a raw Image; a CompressedImage decoded to raw Image fields.

    Encoding is mono8, or mono16 for a 16-bit PNG (scaled later by `image_decode.mono16_to_mono8`).
    Colour becomes grey by the luma weights of `image_decode`, as raw rgb8/bgr8 does, so a colour
    PNG gives the pixels of the raw colour frame it encodes. Warns once per topic in `warned`
    on JPEG."""
    if not hasattr(msg, "format"):
        return msg
    try:
        codec = compressed_codec(msg)
        if codec == "jpeg" and topic not in warned:
            warned.add(topic)
            logger.warning("%s: JPEG images are lossy and can shift detected corners; "
                           "prefer raw or PNG for a calibration recording", topic)
        # IMREAD_UNCHANGED keeps 16-bit depth and ignores EXIF rotation, which would swap
        # height and width against the intrinsics.
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"unreadable compressed image: {codec} data did not decode "
                             f"({np.size(msg.data)} bytes)")
        if img.dtype not in (np.uint8, np.uint16):
            raise ValueError(f"unsupported compressed image: {img.dtype}")
        if img.ndim == 3:       # imdecode gives BGR(A); luma, keeping the bit depth
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if img.shape[2] == 3
                               else cv2.COLOR_BGRA2GRAY)
    except ValueError as e:
        stamp = msg.header.stamp
        raise ValueError(f"{topic} at {stamp.sec}.{stamp.nanosec:09d}: {e}") from None
    return SimpleNamespace(height=img.shape[0], width=img.shape[1], data=img,
                           encoding="mono8" if img.dtype == np.uint8 else "mono16")
