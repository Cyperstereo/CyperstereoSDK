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
"""Abstract camera model interface.

This is the single home for per-model camera math (projection + Jacobians,
unprojection, intrinsics layout). Both the camera-calib pipeline and the cam-IMU
factors/init consume it polymorphically — no `if model == ...` branching outside
this package.
"""

from abc import ABC, abstractmethod

import numpy as np
import gtsam

from mvis.core.camera_models import MODELS, canonical_dist, check_model, n_dist

# gtsam Cal3DS2/Cal3Fisheye expose a 9-col cal Jacobian [fx,fy,skew,cx,cy,d1..d4];
# our cal_vec drops skew -> 8.
_GTSAM_TO_OURS = [0, 1, 3, 4, 5, 6, 7, 8]


def _in_front(p_cam) -> bool:
    """True when a camera-frame point is strictly in front of the camera (z > 0).

    The shared front-hemisphere test for the pinhole models: their jac=False path
    divides by z, which would mirror a point behind the camera into the image. NaN z
    counts as behind. The jac=True path needs no check, gtsam's cheirality test
    already rejects z <= 0."""
    return bool(p_cam[2] > 0.0)


def _z_depth(ray, depth: float) -> np.ndarray:
    """The point on a model's bearing `ray` at z = `depth`, the one `unproject` convention.
    NaN when the model has no ray for the pixel or the ray does not point
    in front of the camera (z <= 0, which has no z-depth)."""
    if ray is None or not ray[2] > 0.0:
        return np.full(3, np.nan)
    return ray / ray[2] * depth


def _gtsam_project(cal, p_cam, jac):
    """Camera-frame project via a gtsam Cal3 at IDENTITY pose. jac=True ->
    (uv, J_point[2x3 wrt p_cam], J_intr[2x8 our order]); invalid -> (nan, None,
    None) (jac) or nan (no jac)."""
    cam_cls = (gtsam.PinholeCameraCal3Fisheye if isinstance(cal, gtsam.Cal3Fisheye)
               else gtsam.PinholeCameraCal3DS2)
    pt = gtsam.Point3(*p_cam)
    if not jac:
        try:
            return np.asarray(cam_cls(gtsam.Pose3(), cal).project(pt), float)
        except RuntimeError:
            return np.array([np.nan, np.nan])
    Jpt = np.zeros((2, 3), order="F")
    J_pose = np.zeros((2, 6), order="F")
    Jc9 = np.zeros((2, 9), order="F")
    try:
        uv = cam_cls(gtsam.Pose3(), cal).project(pt, J_pose, Jpt, Jc9)
    except RuntimeError:
        return np.array([np.nan, np.nan]), None, None
    return np.asarray(uv, float), Jpt, Jc9[:, _GTSAM_TO_OURS]


def _kernel_project(kernel, p_cam, cal_vec, n_intr, jac):
    """Camera-frame project via a custom_projections kernel. Slices the kernel's
    J_intr (2 x max(8, len(cal_vec))) to n_intr; invalid -> (nan, None, None)
    (jac) or nan."""
    if not jac:
        uv, _, _ = kernel(p_cam, cal_vec, False)
        return np.array([np.nan, np.nan]) if uv is None else np.asarray(uv, float)
    uv, J_pc, J_intr = kernel(p_cam, cal_vec, True)
    if uv is None:
        return np.array([np.nan, np.nan]), None, None
    J_intr = np.asarray(J_intr)
    assert J_intr.shape[1] >= n_intr, (
        f"kernel J_intr is {J_intr.shape[1]} wide, model needs {n_intr}")
    return np.asarray(uv, float), J_pc, J_intr[:, :n_intr]


class CameraModel(ABC):
    """Abstract base class for camera models."""

    @property
    def cal_vec(self) -> np.ndarray:
        """Full intrinsics vector [fx, fy, cx, cy, dist...] (length = n_intr)."""
        return np.concatenate([self.intrinsics(), self.distortion()])

    @property
    def n_intr(self) -> int:
        """Number of intrinsic parameters (= len(cal_vec) = num_params())."""
        return 4 + len(self.distortion())

    def unproject_to_normalized(self, pixels: np.ndarray):
        """Pixels (N,2) -> normalized image points (x/z, y/z) (N,2) + validity
        (N,) bool. Unprojects each pixel to a bearing ray via `self.unproject`
        then divides by z; used to feed a generic model to OpenCV `solvePnP` with
        identity K. `valid` is True only for rays in front (z > 0) that the model
        could unproject; invalid rows are zero. Model-agnostic — the per-model
        math lives in each subclass's `unproject`."""
        pixels = np.asarray(pixels, float).reshape(-1, 2)
        norm = np.zeros((len(pixels), 2))
        valid = np.zeros(len(pixels), bool)
        for j, px in enumerate(pixels):
            ray = self.unproject(px)
            if ray is None or not np.isfinite(ray).all() or abs(ray[2]) < 1e-9:
                continue
            norm[j] = (ray[0] / ray[2], ray[1] / ray[2])
            valid[j] = ray[2] > 0
        return norm, valid

    @abstractmethod
    def project(self, point_3d: np.ndarray, jac: bool = False):
        """Project a 3D point in camera frame to pixel coordinates.

        Args:
            point_3d: 3D point [X, Y, Z] in camera frame.
            jac: if True, also return analytic Jacobians.

        Returns:
            jac=False: 2D pixel [u, v] ([nan, nan] if out of model range).
            jac=True : (uv[2], J_point[2x3 wrt point_3d], J_intr[2 x n_intr]);
                       (uv=[nan,nan], None, None) if out of model range.
        """

    @abstractmethod
    def unproject(self, pixel: np.ndarray, depth: float = 1.0) -> np.ndarray:
        """Unproject a pixel to the camera-frame point at z-depth `depth`.

        Args:
            pixel: 2D pixel coordinate [u, v].
            depth: z of the returned point (not the range along the ray).

        Returns:
            3D point [X, Y, depth]; [nan, nan, nan] when the pixel has no ray in
            front of the camera (z <= 0, e.g. past 90 deg on a wide-FOV model —
            the bearing kernels in `custom_projections` still return those).
        """

    @abstractmethod
    def get_gtsam_calibration(self):
        """Return the underlying GTSAM calibration object."""

    @abstractmethod
    def intrinsics(self) -> np.ndarray:
        """Return intrinsic parameters [fx, fy, cx, cy]."""

    @abstractmethod
    def distortion(self) -> np.ndarray:
        """Return distortion coefficients."""

    @abstractmethod
    def distortion_model_name(self) -> str:
        """Return the distortion model name."""

    @abstractmethod
    def num_params(self) -> int:
        """Return total number of calibration parameters."""


def create_camera_model(
    distortion_model: str,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion_coeffs: np.ndarray | None = None,
) -> CameraModel:
    """Factory function to create a camera model.

    Args:
        distortion_model: a `core.camera_models.MODELS` name.
        fx, fy: Focal lengths.
        cx, cy: Principal point.
        distortion_coeffs: Distortion coefficients (default: model-specific).

    Returns:
        CameraModel instance.
    """
    from mvis.camera.pinhole_radtan import PinholeRadtan
    from mvis.camera.pinhole_equidistant import PinholeEquidistant
    from mvis.camera.fov_model import FOVModel
    from mvis.camera.double_sphere import DoubleSphere
    from mvis.camera.eucm import EUCM
    from mvis.camera.ucm import UCM
    from mvis.camera.omni_radtan import OmniRadtan
    from mvis.camera.kb4 import KannalaBrandt4

    check_model(distortion_model)
    if distortion_model == "radtan":
        coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros(4)
        return PinholeRadtan(fx, fy, cx, cy, coeffs)
    elif distortion_model == "equidistant":
        coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros(4)
        return PinholeEquidistant(fx, fy, cx, cy, coeffs)
    elif distortion_model == "fov":
        return FOVModel(fx, fy, cx, cy, distortion_coeffs)
    elif distortion_model == "double_sphere":
        return DoubleSphere(fx, fy, cx, cy, distortion_coeffs)
    elif distortion_model == "eucm":
        return EUCM(fx, fy, cx, cy, distortion_coeffs)
    elif distortion_model == "ucm":
        return UCM(fx, fy, cx, cy, distortion_coeffs)
    elif distortion_model == "omni_radtan":
        return OmniRadtan(fx, fy, cx, cy, distortion_coeffs)
    elif distortion_model == "kb4":
        return KannalaBrandt4(fx, fy, cx, cy, distortion_coeffs)
    else:   # registered but no CameraModel class yet
        raise NotImplementedError(
            f"No CameraModel class for distortion model {distortion_model!r} "
            f"(registered: {', '.join(MODELS)})")


def build_camera_model(model: str, cal_vec: np.ndarray) -> "CameraModel":
    """Build a CameraModel from a flat cal_vec [fx, fy, cx, cy, dist...].

    The cam-IMU / CameraCalib form: the distortion coefficients are everything
    after the principal point, canonicalized to the model's n_dist (legacy short
    lists zero-padded, zero-padded extras trimmed, anything else raises).
    """
    cal_vec = np.asarray(cal_vec, float)
    if model == "kb4" and len(cal_vec) != 8:   # no legacy kb4 files, so never guess
        raise ValueError(f"kb4 cal_vec must be 8 long [fx,fy,cx,cy,k1..k4], got {len(cal_vec)}")
    if len(cal_vec) != 4 + n_dist(model):
        cal_vec = np.concatenate([cal_vec[:4], canonical_dist(model, cal_vec[4:])])
    return create_camera_model(model, cal_vec[0], cal_vec[1], cal_vec[2],
                               cal_vec[3], cal_vec[4:])
