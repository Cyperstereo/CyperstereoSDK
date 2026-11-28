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
"""One decoder from a camera frame to the uint8 grey image the board detectors read.

Every reader goes through here: the camera-only sources of `io.data_source.make_sources`
(ROS1, ROS2, EuRoC, image folders; `DataLoader` and `mvis calibrate`), the cam-IMU loader
and the mono16 bag converter.
So one frame gives the same pixels on every path.

Rules, per encoding:

* ``mono8`` / ``8UC1``: the bytes as they are.
* ``mono16`` / ``16UC1`` (and every 16-bit image): `mono16_to_mono8`. A frame whose
  maximum is above 4095 holds its data in the upper byte (TUM-VI, any 16-bit camera):
  ``>> 8``. A frame whose maximum is at most 4095 is 10- or 12-bit data in a 16-bit
  container (IR cameras): min-max stretch per frame, floored. This is the rule the
  camera-only regression path has used since 2026-05-01.
* ``rgb8`` / ``bgr8`` / ``8UC3`` (BGR) / ``rgba8`` / ``bgra8`` / ``8UC4`` (BGRA) and their
  16-bit forms: OpenCV luma weights (0.299 R + 0.587 G + 0.114 B) in the right channel
  order.
* ``bayer_rggb8`` / ``bggr8`` / ``gbrg8`` / ``grbg8`` (and ``*16``): OpenCV demosaic to
  grey, with the cv_bridge pattern mapping.
* ``CompressedImage`` (PNG, JPEG): decoded by `compressed_image.as_raw_image`, then the
  rules above. Colour comes out by luma, as raw colour does.
* Image files: ``cv2.IMREAD_UNCHANGED`` (no EXIF rotation, 16 bits kept), then the same
  rules, with OpenCV's BGR(A) channel order.

Row padding (``step``) and ``is_bigendian`` are honoured. Anything else raises ValueError.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# A 16-bit frame whose maximum is above this holds 8 significant bits in its upper byte.
MONO16_SHIFT_ABOVE = 4095

# encoding -> (dtype, channels, cv2 colour code or None)
_ENCODINGS = {
    "mono8": (np.uint8, 1, None), "8UC1": (np.uint8, 1, None),
    "mono16": (np.uint16, 1, None), "16UC1": (np.uint16, 1, None),
    "rgb8": (np.uint8, 3, cv2.COLOR_RGB2GRAY), "bgr8": (np.uint8, 3, cv2.COLOR_BGR2GRAY),
    "8UC3": (np.uint8, 3, cv2.COLOR_BGR2GRAY),
    "rgba8": (np.uint8, 4, cv2.COLOR_RGBA2GRAY), "bgra8": (np.uint8, 4, cv2.COLOR_BGRA2GRAY),
    "8UC4": (np.uint8, 4, cv2.COLOR_BGRA2GRAY),
    "rgb16": (np.uint16, 3, cv2.COLOR_RGB2GRAY), "bgr16": (np.uint16, 3, cv2.COLOR_BGR2GRAY),
    "16UC3": (np.uint16, 3, cv2.COLOR_BGR2GRAY),
    "rgba16": (np.uint16, 4, cv2.COLOR_RGBA2GRAY), "bgra16": (np.uint16, 4, cv2.COLOR_BGRA2GRAY),
    "16UC4": (np.uint16, 4, cv2.COLOR_BGRA2GRAY),
}
# ROS names the Bayer pattern by its first row, OpenCV by the second (cv_bridge's mapping).
_BAYER = {"rggb": cv2.COLOR_BayerBG2GRAY, "bggr": cv2.COLOR_BayerRG2GRAY,
          "gbrg": cv2.COLOR_BayerGR2GRAY, "grbg": cv2.COLOR_BayerGB2GRAY}
for _p, _code in _BAYER.items():
    _ENCODINGS[f"bayer_{_p}8"] = (np.uint8, 1, _code)
    _ENCODINGS[f"bayer_{_p}16"] = (np.uint16, 1, _code)
_BAYER_CODES = set(_BAYER.values())


def mono16_to_mono8(a: np.ndarray) -> np.ndarray:
    """A 16-bit grey frame as uint8: ``>> 8`` when its maximum is above 4095, else a
    per-frame min-max stretch (integer, floored); all zeros for a constant frame."""
    a = np.asarray(a)
    hi = int(a.max()) if a.size else 0
    if hi > MONO16_SHIFT_ABOVE:
        return (a >> 8).astype(np.uint8)
    lo = int(a.min()) if a.size else 0
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    # int32 before the multiply: (4095 - 0) * 255 overflows uint16.
    return ((a.astype(np.int32) - lo) * 255 // (hi - lo)).astype(np.uint8)


def array_to_gray(img: np.ndarray, code: int | None = None) -> np.ndarray:
    """A decoded uint8/uint16 image (grey, or colour in OpenCV's BGR/BGRA order unless
    `code` says otherwise) as the uint8 grey image the detectors read."""
    img = np.asarray(img)
    if img.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"unsupported image depth {img.dtype}")
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim == 3:
        if code is None:
            code = {3: cv2.COLOR_BGR2GRAY, 4: cv2.COLOR_BGRA2GRAY}.get(img.shape[2])
            if code is None:
                raise ValueError(f"unsupported image with {img.shape[2]} channels")
        img = cv2.cvtColor(np.ascontiguousarray(img), code)
    elif img.ndim != 2:
        raise ValueError(f"unsupported image shape {img.shape}")
    elif code in _BAYER_CODES:
        img = cv2.cvtColor(np.ascontiguousarray(img), code)
    return mono16_to_mono8(img) if img.dtype == np.uint16 else img


def _raw_pixels(msg, dtype, channels) -> np.ndarray:
    """The (h, w[, c]) pixel array of a raw sensor_msgs/Image, honouring step and endianness."""
    h, w = int(msg.height), int(msg.width)
    itemsize = np.dtype(dtype).itemsize
    row = w * channels * itemsize
    step = int(getattr(msg, "step", 0) or row)
    buf = np.frombuffer(msg.data, np.uint8)
    if step < row or (h and buf.size < step * (h - 1) + row):
        raise ValueError(f"image data too short: {buf.size} bytes for {h}x{w} "
                         f"{msg.encoding} with step {step}")
    if step == row:
        rows = buf[:h * row]
    else:
        rows = np.lib.stride_tricks.as_strided(buf, shape=(h, row), strides=(step, 1))
        rows = np.ascontiguousarray(rows).reshape(-1)
    if itemsize > 1:
        order = ">" if getattr(msg, "is_bigendian", 0) else "<"
        rows = rows.view(np.dtype(dtype).newbyteorder(order)).astype(dtype)
    shape = (h, w) if channels == 1 else (h, w, channels)
    return rows.reshape(shape)


def ros_image_to_gray(msg, topic: str = "", warned: set | None = None) -> np.ndarray:
    """A sensor_msgs/Image or CompressedImage (ROS1 or ROS2, or any object with its fields)
    as a uint8 grey (h, w) image. `topic` and `warned` go to the JPEG warning of
    `compressed_image.as_raw_image`. ValueError on an unsupported encoding or bad data."""
    if hasattr(msg, "format"):
        from mvis.io.compressed_image import as_raw_image
        msg = as_raw_image(msg, topic, set() if warned is None else warned)
    data = msg.data
    if isinstance(data, np.ndarray) and data.ndim >= 2:   # already decoded (as_raw_image)
        return array_to_gray(data)
    enc = getattr(msg, "encoding", "")
    try:
        dtype, channels, code = _ENCODINGS[enc]
    except KeyError:
        raise ValueError(f"unsupported image encoding {enc!r}"
                         + (f" on {topic}" if topic else "")) from None
    return array_to_gray(_raw_pixels(msg, dtype, channels), code)


def image_file_to_gray(path) -> np.ndarray | None:
    """An image file (PNG, JPEG, ...) as a uint8 grey image; None if it cannot be read."""
    img = cv2.imread(str(Path(path)), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    return array_to_gray(img)
