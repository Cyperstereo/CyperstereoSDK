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
"""Optional pixel undistortion for the corrected bag (--undistort)."""
from __future__ import annotations
import numpy as np, cv2

# models with a cv2 undistortion path; anything else cannot be undistorted here
UNDISTORT_MODELS = ("radtan", "plumb_bob", "equidistant", "fisheye")


def check_undistortable(plan) -> None:
    """Raise before any output is written when --undistort is asked for a
    camera whose model has no cv2 path (fov / ds / eucm / ucm / omni_radtan / kb4):
    the image would pass through unchanged while provenance says undistorted."""
    bad = {t: tp.cam.model for t, tp in plan.items()
           if tp.kind == "cam" and tp.cam is not None
           and tp.cam.model not in UNDISTORT_MODELS}
    if bad:
        raise ValueError(f"--undistort supports only {UNDISTORT_MODELS}; "
                         f"unsupported camera model(s): {bad}")


def undistort_image_msg(msg, cam, *, strict=False):
    """Undistort a single-channel (mono8/8UC1) image message in place through `cam`'s model.
    Rows may be padded (step > width); the output is unpadded (step = width). Other
    encodings and models without a cv2 path pass through unchanged, or raise with `strict`."""
    if getattr(msg, "encoding", None) not in ("mono8", "8UC1"):
        if strict:
            raise ValueError(f"unsupported image encoding for undistortion: "
                             f"{getattr(msg, 'encoding', None)}")
        return msg
    if cam.model not in UNDISTORT_MODELS:
        if strict:
            raise ValueError(f"unsupported undistortion model for {cam.name}: {cam.model}")
        return msg
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    data = np.frombuffer(msg.data, np.uint8)        # bytes or a rosbags uint8 array
    if h <= 0 or w <= 0 or step < w or data.size != h * step:
        raise ValueError("invalid image dimensions, row stride, or data length")
    arr = np.ascontiguousarray(data.reshape(h, step)[:, :w])
    fx, fy, cx, cy = [float(v) for v in cam.K]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    d = np.asarray(cam.dist, float)
    if cam.model in ("radtan", "plumb_bob"):
        out = cv2.undistort(arr, K, d)
    else:                                   # equidistant / fisheye
        out = cv2.fisheye.undistortImage(arr, K, d, Knew=K)
    msg.data = np.ascontiguousarray(out, np.uint8).reshape(-1)
    msg.step = w
    return msg
