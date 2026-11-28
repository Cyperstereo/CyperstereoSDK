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
"""ChArUco board: a chessboard with an ArUco marker in each white square.

The calibration points are the chessboard's inner corners, as on a plain
`Chessboard` (same point ids, same board frame, one point per id), so every
chessboard code path applies. The markers only name the corners, so a ChArUco
board is found when part of it is out of view or hidden, which a plain
chessboard is not.
"""

import logging

import numpy as np

from mvis.detection.aruco_board import _ARUCO_DICT_MAP, aruco_detector
from mvis.detection.board import LazyDetector, scaled_to_original
from mvis.detection.chessboard import Chessboard
from mvis.core.types import Observation

logger = logging.getLogger(__name__)

MIN_CORNERS = 6
UPSCALE_BELOW = 0.85      # share of inner corners under which a 2x pass is tried
SUBPIX_ITERS = 40         # cv2.cornerSubPix stop: iterations ...
SUBPIX_EPS = 0.001        # ... or a step below this (px)
# Bump when the detection method changes: it is in the cache key (board_fingerprint),
# so detections cached by an older method are not reused.
DETECTOR_REV = 1


def dictionary_problem(aruco_dict: str, rows: int, cols: int):
    """Why `aruco_dict` can't name a `rows` x `cols`-square ChArUco board (unknown name, or
    fewer markers than the board's rows*cols//2), or None if it can."""
    if aruco_dict not in _ARUCO_DICT_MAP:
        return f"Unknown ArUco dictionary {aruco_dict!r}"
    need = (rows * cols) // 2
    if need > int(aruco_dict.rsplit("_", 1)[1]):
        return f"{aruco_dict} has fewer than the {need} markers a {rows} x {cols} board needs"
    return None


class CharucoBoard(LazyDetector, Chessboard):
    """A ChArUco board of `rows` x `cols` SQUARES (the count board generators and
    OpenCV give), so `(rows - 1) x (cols - 1)` inner corners.

    Point ids and positions follow OpenCV: id = r * (cols - 1) + c for inner corner
    (r, c), at (c, r) * square_size from the first inner corner (the Chessboard
    frame, shifted by one square from OpenCV's board corner origin).

    `legacy_pattern` is for boards printed with OpenCV before 4.6, whose marker
    layout differs on boards with an even number of rows.

    OpenCV's `CharucoDetector` (4.13) returns corners about 0.5 px down and right of
    the true corner, so each corner is refined again with `cv2.cornerSubPix` in the
    native image (see `_refine`); `refine=False` keeps OpenCV's corners.

    The detector settings are instance scalars so that `board_fingerprint` (the
    detection-cache key) changes with them.
    """

    def __init__(self, rows: int, cols: int, square_size: float, marker_size: float,
                 aruco_dict: str = "DICT_6X6_250", legacy_pattern: bool = False,
                 refine: bool = True):
        if rows < 3 or cols < 3:
            raise ValueError(f"ChArUco board needs at least 3 x 3 squares, got {rows} x {cols}")
        if not 0 < marker_size < square_size:
            raise ValueError(f"ChArUco marker_size {marker_size} must be in (0, square_size "
                             f"{square_size})")
        problem = dictionary_problem(aruco_dict, rows, cols)
        if problem:
            raise ValueError(problem)
        super().__init__(rows - 1, cols - 1, square_size)
        self._squares_rows = rows
        self._squares_cols = cols
        self._charuco_marker = float(marker_size)   # not `_marker_size`: that marks a tag grid
        self._aruco_dict_name = aruco_dict
        self._legacy_pattern = bool(legacy_pattern)
        self._refine_corners = bool(refine)
        self._min_corners = MIN_CORNERS
        self._upscale_below = UPSCALE_BELOW
        self._subpix_iters = SUBPIX_ITERS
        self._subpix_eps = SUBPIX_EPS
        self._detector_rev = DETECTOR_REV
        self._warned = []         # a list, not a bool: kept out of the fingerprint

    def _make_detector(self):
        import cv2
        if not hasattr(cv2.aruco, "CharucoDetector"):
            raise ImportError(f"ChArUco detection needs OpenCV >= 4.7 (found {cv2.__version__})")
        marker_det = aruco_detector(self._aruco_dict_name)
        board = cv2.aruco.CharucoBoard(
            (self._squares_cols, self._squares_rows), self._square_size,
            self._charuco_marker, marker_det.getDictionary())
        board.setLegacyPattern(self._legacy_pattern)
        return cv2.aruco.CharucoDetector(board, cv2.aruco.CharucoParameters(),
                                         marker_det.getDetectorParameters())

    def _detect_raw(self, gray: np.ndarray):
        """(corners (N, 2), ids (N,)) found by OpenCV in `gray`; empty arrays if none."""
        corners, ids, _, marker_ids = self._get_detector().detectBoard(gray)
        if ids is None or len(ids) == 0:
            self._check_layout(marker_ids)
            return np.zeros((0, 2)), np.zeros(0, int)
        return corners.reshape(-1, 2).astype(np.float64), ids.reshape(-1).astype(int)

    def _check_layout(self, marker_ids) -> None:
        """Warn once when markers decode but give no corner: the usual sign of rows/cols
        given as inner corners or swapped, or the wrong `legacy_pattern`."""
        if self._warned or marker_ids is None or len(marker_ids) < 10:   # 10: not a corner of the board
            return
        self._warned.append(True)
        n_markers = (self._squares_rows * self._squares_cols) // 2
        top = int(np.max(marker_ids))
        logger.warning(
            f"ChArUco: {len(marker_ids)} markers decoded (ids up to {top}) but no corner "
            f"found on the {self._squares_rows} x {self._squares_cols} board ({n_markers} "
            "markers). Check rows/cols (they count squares, rows along the image height "
            "of the printed board), the dictionary and charuco_legacy.")

    def detect(self, image: np.ndarray) -> list[Observation]:
        """Inner corners found in `image`, refined to subpixel in the native image by
        `_refine` (not OpenCV's own corners, which are off by ~0.5 px). If fewer than 85 % of the corners are found, a 2x upscaled
        pass is tried and kept only when it finds more. Returns [] for fewer than six
        corners, or corners all on one board row or column (no usable pose)."""
        import cv2
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        corners, ids = self._detect_raw(gray)
        if len(ids) < self._upscale_below * self.num_points():
            up_c, up_ids = self._detect_raw(
                cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
            if len(up_ids) > len(ids):
                logger.debug("ChArUco upscale (2x): %d -> %d corners", len(ids), len(up_ids))
                corners, ids = scaled_to_original(up_c, 2.0), up_ids

        keep = (ids >= 0) & (ids < self.num_points())
        corners, ids = corners[keep], ids[keep]
        if (len(ids) < self._min_corners or len(set(ids // self._cols)) < 2
                or len(set(ids % self._cols)) < 2):
            return []
        if self._refine_corners:
            corners = self._refine(gray, corners, ids)
        order = np.argsort(ids)
        return [Observation(point_id=int(ids[k]), pixel=corners[k], row=int(corners[k][1]))
                for k in order]

    def _refine(self, gray: np.ndarray, corners: np.ndarray, ids: np.ndarray) -> np.ndarray:
        """`cv2.cornerSubPix` on each corner, with a half-window no wider than the
        white gap between the corner and the nearest marker, (square - marker) / 2, so
        marker edges stay out of the window (2 to 11 px). The gap in pixels comes from
        the corner's detected neighbours up to two squares away (the smallest pixels
        per square among them)."""
        import cv2
        gap = (self._square_size - self._charuco_marker) / (2 * self._square_size)
        grid = np.column_stack([ids % self._cols, ids // self._cols]).astype(float)
        d_px = np.linalg.norm(corners[:, None] - corners[None], axis=2)
        d_sq = np.linalg.norm(grid[:, None] - grid[None], axis=2)
        near = (np.abs(grid[:, None] - grid[None]).max(axis=2) <= 2) & (d_sq > 0)
        px_per_sq = np.where(near, d_px / np.where(d_sq > 0, d_sq, 1), np.inf).min(axis=1)
        px_per_sq[~np.isfinite(px_per_sq)] = np.median(px_per_sq[np.isfinite(px_per_sq)]) \
            if np.isfinite(px_per_sq).any() else 0.0
        half = np.clip(np.floor(gap * px_per_sq), 2, 11).astype(int)
        out = corners.astype(np.float32)
        for w in np.unique(half):
            sel = half == w
            out[sel] = cv2.cornerSubPix(gray, out[sel].reshape(-1, 1, 2).copy(),
                                        (int(w), int(w)), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                                         self._subpix_iters, self._subpix_eps)).reshape(-1, 2)
        return out.astype(np.float64)
