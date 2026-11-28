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
"""AprilTag grid calibration board detection.

Supports both standard AprilTag (1-bit border) and Kalibr AprilGrid (2-bit border).
Uses the 'aprilgrid' package which natively handles both border formats.

Tag family naming convention:
  - "t36h11"   = Kalibr format (2-bit black border, 10x10 grid) — used by TUM-VI, Kalibr
  - "t36h11b1" = Standard apriltag3 format (1-bit black border, 8x8 grid)
"""

import logging

import numpy as np

from mvis.detection.board import Board, scaled_to_original  # noqa: F401  (old import path)
from mvis.core.types import Observation

logger = logging.getLogger(__name__)

# Minimum median tag size in pixels for reliable detection.
# Below this, the quad finder fails on oblique/small tags.
MIN_TAG_PIXEL_SIZE = 20.0

# If fewer than this fraction of expected tags are detected, warn and retry.
MIN_DETECTION_RATE = 0.85


def _compute_tag_sizes(detections) -> np.ndarray:
    """Compute the size (median side length in pixels) of each detected tag.

    Uses the **median** of the 4 quad sides rather than the max. This is
    more robust to perspective distortion: an oblique tag might have one
    side at 120 px and the opposite at 30 px. Max would report 120 (making
    the tag look huge even though one dimension is tiny). Median gives a
    representative ~75 px that better reflects the actual module size the
    decoder is working with.

    Returns:
        Array of tag sizes, one per detection.
    """
    sizes = []
    for det in detections:
        corners = np.asarray(det.corners).reshape(-1, 2)
        sides = sorted(
            float(np.linalg.norm(corners[i] - corners[(i + 1) % 4]))
            for i in range(4)
        )
        # Lower-median of 4 values = sides[1] — biases toward the
        # shorter dimension, which is the limiting factor for detection.
        sizes.append(sides[1])
    return np.array(sizes) if sizes else np.array([])



# Two detections of the same tag whose original-pixel quad areas agree within
# this fraction count as a tie (the area then only reflects corner noise).
_AREA_TIE_TOL = 0.05

# Frames whose native tags have a median module size below this (px) keep the
# highest-scale detection of each tag instead of the native one: there the
# native corners are the noisy ones. TUM-VI 512 (2.2-2.8 px modules): native
# vs 8x gives fx +3.3 px and cam-IMU reprojection 0.18 -> 0.60 px. At 3+ px
# (TUM-VI 1024, ELP, a 180-deg wide-FOV rig) native wins (TUM-VI 1024 cam-IMU
# 0.38 -> 0.19 px). Decided per frame, not per tag: mixing native and
# upscaled corners of one frame adds a ~0.3 px inconsistency.
_NATIVE_MIN_MODULE_PX = 3.0


def _quad_area(c: np.ndarray) -> float:
    """Area of a 4-corner quad (shoelace on the diagonals)."""
    return float(abs((c[2, 0] - c[0, 0]) * (c[3, 1] - c[1, 1]) -
                     (c[3, 0] - c[1, 0]) * (c[2, 1] - c[0, 1]))) * 0.5

class AprilTagBoard(Board):
    """AprilTag grid board — each tag has a unique ID, 4 corners per tag.

    Supports Kalibr's AprilGrid format (2-bit border) via the 'aprilgrid' package.

    Adaptive upscaling: If first-pass detection finds tags that are too small
    (median < 20px) or fewer than 85% of expected tags, the image is upscaled
    and detection is retried automatically.

    Requires: pip install aprilgrid
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        tag_size: float,
        tag_spacing: float,
        tag_family: str = "t36h11",
        start_id: int = 0,
        min_tag_px: float = MIN_TAG_PIXEL_SIZE,
        min_detection_rate: float = MIN_DETECTION_RATE,
    ):
        """
        Args:
            rows: Number of tag rows.
            cols: Number of tag columns.
            tag_size: Size of each tag in meters (edge to edge).
            tag_spacing: Spacing between tags in meters. If using Kalibr config
                         where tag_spacing is a ratio, pass tag_size * ratio.
            tag_family: Tag family string for aprilgrid detector:
                        "t36h11"   = Kalibr (2-bit border)
                        "t36h11b1" = standard apriltag3 (1-bit border)
            start_id: ID of the first tag in the grid.
            min_tag_px: Minimum median tag size in pixels. If detected tags are
                        smaller, the image is upscaled and detection retried.
            min_detection_rate: Minimum fraction of expected tags that should be
                                detected. If below this, warn and retry with upscaling.
        """
        self._rows = rows
        self._cols = cols
        self._tag_size = tag_size
        self._tag_spacing = tag_spacing
        self._tag_family = tag_family
        self._start_id = start_id
        self._min_tag_px = min_tag_px
        self._min_detection_rate = min_detection_rate

        self._num_tags = rows * cols
        step = tag_size + tag_spacing

        # Precompute 3D positions of all tag corners in the board frame (z=0 plane)
        # Corner order follows aprilgrid convention:
        #   0=bottom-left, 1=bottom-right, 2=top-right, 3=top-left
        self._points_3d = {}
        half = tag_size / 2.0
        corner_offsets = np.array([
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
        ])

        for r in range(rows):
            for c in range(cols):
                tag_idx = r * cols + c
                center = np.array([c * step, r * step, 0.0])
                for corner_idx in range(4):
                    point_id = tag_idx * 4 + corner_idx
                    self._points_3d[point_id] = center + corner_offsets[corner_idx]

        self._detector = None

    def _get_detector(self):
        """Lazily construct the aprilgrid detector (import deferred so the optional
        dependency is only required when detection is actually used)."""
        if self._detector is None:
            try:
                import aprilgrid
            except ImportError:
                raise ImportError(
                    "AprilTag detection requires 'aprilgrid'. "
                    "Install with: pip install aprilgrid\n"
                    "Note: 'aprilgrid' supports Kalibr's 2-bit border format (t36h11) "
                    "which is NOT compatible with standard pupil-apriltags."
                )
            self._detector = aprilgrid.Detector(self._tag_family)
        return self._detector

    def _detect_at_scale(self, gray: np.ndarray, scale: float = 1.0):
        """Run detection at a given scale factor.

        Returns:
            (detections, scale_used): raw detections and actual scale.
        """
        import cv2

        if scale != 1.0:
            gray_scaled = cv2.resize(
                gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        else:
            gray_scaled = gray

        detector = self._get_detector()
        try:
            return detector.detect(gray_scaled), scale
        except cv2.error:
            # The aprilgrid package runs cv2.cornerSubPix internally;
            # on upscaled images this can assert when a detected corner
            # falls outside the valid rect. Catch and return empty.
            return [], scale

    # AprilGrid t36h11 modules per tag side (8 data bits + 2-bit border).
    _TAG_MODULES_PER_SIDE = 10

    # Decoder comfort zone in module pixels (informed by AprilTag library
    # author guidance + Kalibr empirical tuning).
    _MODULE_PX_MIN = 15.0
    _MODULE_PX_MAX = 40.0
    _MODULE_PX_TARGET = 25.0
    _DOWNSCALE_TRIGGER_MODULE_PX = 50.0   # tags larger than this waste cycles

    # Stop the per-frame scale search when at least this fraction of tags
    # is found and the size distribution is comfortable.
    _EARLY_STOP_DETECTION_RATE = 0.95
    # Moderate coverage threshold: below this we try a 3rd adaptive scale.
    _MODERATE_DETECTION_RATE = 0.70

    # Skip the upscale pass when the native pass already decodes the board:
    # at least `_MODERATE_DETECTION_RATE` of the tags found AND every found
    # tag's module (lower-median quad side / modules per side) >= this many px (the floor above which native corners are
    # trusted over upscaled ones, `_NATIVE_MIN_MODULE_PX`). The tags the native
    # pass misses there are mostly inside the found grid (occluded, blurred),
    # not too small, and an upscale rarely recovers them. CyberStereo
    # 1280x1024, 4-7 px modules, 44 frames/camera: 35-40 % less detector time
    # per frame, 0.02-0.37 fewer tags per frame, identical corners for the
    # tags found both ways. None restores the previous rule (upscale whenever
    # a module is < `_MODULE_PX_MIN` or the rate is < `_EARLY_STOP_DETECTION_RATE`).
    _UPSCALE_SKIP_MODULE_PX: float | None = _NATIVE_MIN_MODULE_PX

    def _auto_max_scale(self, gray: np.ndarray) -> float:
        """Pick max_scale based on image dimensions."""
        max_dim = max(gray.shape[:2])
        if max_dim <= 512:
            return 8.0
        if max_dim <= 1024:
            return 6.0
        return 4.0

    def _hail_mary_scales(self, max_scale: float) -> list[float]:
        """Progressively aggressive scales when nothing detected at 1.0."""
        candidates = [2.0, 4.0, 6.0, 8.0]
        return [s for s in candidates if s <= max_scale]

    def detect(self, image: np.ndarray,
                max_scale: float | None = None,
                min_scale: float = 0.5,
                refine_subpixel: bool = False) -> list[Observation]:
        """Per-frame adaptive multi-scale AprilTag detection.

        Strategy:

        1. Detect at scale=1.0.
        2. If detection rate ≥ 95% and module size in comfort zone,
           return immediately — single pass cost.
        3. Otherwise pick up to 3 adaptive extra scales:
           - Upscale to bring smallest module to ~25 px, unless the native
             pass already decodes the board (>= ``_MODERATE_DETECTION_RATE``
             found and smallest module >= ``_UPSCALE_SKIP_MODULE_PX``).
           - Downscale to bring largest module from >50 px to ~25 px.
           - If still < 70% coverage, add an intermediate scale.
        4. If no tags were detected at scale=1.0, try Hail-Mary scales
           progressively up to ``max_scale``.
        5. If partial detections exist and the board is far away (detected
           tags occupy < 1/3 of image area), try a **regional crop
           upscale**: crop a padded region around detected tags, upscale
           just that crop, re-detect to recover edge tags.
        6. Union across all scales by ``tag_id``; keep the detection with
           the largest quad area per tag.
        7. Optionally refine corners with ``cv2.cornerSubPix`` on the
           original image.

        Detection history is stored in ``self._last_detection_history``
        as a list of ``(scale, n_detected, median_tag_px)`` tuples.
        """
        import cv2

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Auto-select max_scale from image dimensions if not explicit.
        if max_scale is None:
            max_scale = self._auto_max_scale(gray)

        # ----- Pass 1: scale = 1.0 ---------------------------------
        history: list[tuple[float, int, float]] = []
        chosen: dict[int, tuple[np.ndarray, float]] = {}
        chosen_scale: dict[int, float] = {}
        small_frame = False

        def _ingest(detections, scale: float,
                    offset: tuple[float, float] = (0.0, 0.0)) -> None:
            """Merge detections into ``chosen`` keyed by tag_id.

            Quad areas are compared in ORIGINAL-image pixels; a detection
            replaces the kept one only if it is clearly larger (a more
            complete quad), and on a near-tie the one detected closest to
            scale 1.0 wins — upscaled corners carry the resize blur and the
            detector's fixed-window subpixel refinement at the upscaled
            resolution, so they are noisier than native ones. Frames of
            small tags (``small_frame``, see ``_NATIVE_MIN_MODULE_PX``) are
            the exception: there the highest-scale detection always wins.

            ``offset`` is (ox, oy) in original-image pixels — non-zero
            when detections come from a regional crop."""
            for det in detections:
                tag_id = det.tag_id - self._start_id
                if tag_id < 0 or tag_id >= self._num_tags:
                    continue
                corners_scaled = np.asarray(det.corners).reshape(-1, 2)
                corners_orig = scaled_to_original(corners_scaled, scale, offset)
                a = _quad_area(corners_orig)
                prev = chosen.get(tag_id)
                if prev is None:
                    take = True
                elif small_frame:
                    take = scale > chosen_scale[tag_id]
                else:
                    take = a > prev[1] * (1.0 + _AREA_TIE_TOL) or (
                        a >= prev[1] * (1.0 - _AREA_TIE_TOL)
                        and abs(np.log(scale)) < abs(np.log(chosen_scale[tag_id])))
                if take:
                    chosen[tag_id] = (corners_orig, a)
                    chosen_scale[tag_id] = scale

        # Run scale 1.0 first.
        dets_1, _ = self._detect_at_scale(gray, 1.0)
        _ingest(dets_1, 1.0)
        small_frame = not chosen or float(np.median(
            [np.sqrt(v[1]) for v in chosen.values()])
        ) / self._TAG_MODULES_PER_SIDE < _NATIVE_MIN_MODULE_PX
        history.append(self._history_entry(1.0, dets_1, scale_for_orig=1.0))

        # ----- Decide on extra scales ------------------------------
        extra_scales = self._adaptive_extra_scales(
            dets_1, chosen, max_scale=max_scale, min_scale=min_scale,
            image_shape=gray.shape[:2],
        )

        scales_tried = [1.0]
        for s in extra_scales:
            dets_s, _ = self._detect_at_scale(gray, s)
            _ingest(dets_s, s)
            history.append(self._history_entry(s, dets_s, scale_for_orig=s))
            scales_tried.append(s)
            if (len(chosen) >= self._EARLY_STOP_DETECTION_RATE * self._num_tags
                    and self._coverage_is_comfortable(chosen)):
                break

        # ----- Regional crop upscale for far-away boards -----------
        # If we have a partial detection (some tags found, but <70% of
        # expected) and the detected tags occupy < 1/3 of image area,
        # crop a padded region around them, upscale just that crop, and
        # re-detect. This catches edge tags the full-image upscale missed.
        rate = len(chosen) / max(self._num_tags, 1)
        if 0 < rate < self._MODERATE_DETECTION_RATE:
            crop_result = self._regional_crop_detect(
                gray, chosen, max_scale, history)
            if crop_result is not None:
                crop_dets, crop_scale, crop_offset = crop_result
                _ingest(crop_dets, crop_scale, offset=crop_offset)

        self._last_detection_history = history

        # ----- Subpixel refinement on the original image -----------
        if refine_subpixel and chosen:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            for tag_id, (corners_orig, _) in list(chosen.items()):
                # Pick window size from the tag's edge length: half the
                # nearest-neighbour spacing keeps cornerSubPix from crossing
                # into adjacent tag/board structure.
                edge = float(np.linalg.norm(corners_orig[1] - corners_orig[0]))
                win = max(3, int(min(edge / 4.0, 15)))   # clamp to [3, 15]
                pts = corners_orig.reshape(-1, 1, 2).astype(np.float32)
                try:
                    cv2.cornerSubPix(gray, pts, (win, win), (-1, -1), criteria)
                    chosen[tag_id] = (pts.reshape(-1, 2).astype(np.float64),
                                       chosen[tag_id][1])
                except cv2.error:
                    pass  # keep un-refined corners

        if len(chosen) > 0:
            scales_used = sorted({round(s, 2) for s, _, _ in history})
            if len(scales_used) > 1 or scales_used != [1.0]:
                logger.info(
                    f"AprilTag adaptive multi-scale: scales={scales_used} → "
                    f"{len(chosen)}/{self._num_tags} tags"
                )

        # ----- Convert to Observations -----------------------------
        observations: list[Observation] = []
        for tag_id, (corners, _) in chosen.items():
            for corner_idx in range(4):
                point_id = tag_id * 4 + corner_idx
                px = corners[corner_idx]
                observations.append(Observation(
                    point_id=point_id, pixel=px, row=int(px[1]),
                ))
        return observations

    # ------------------------------------------------------------------
    # Adaptive scale-selection helpers
    # ------------------------------------------------------------------

    def _adaptive_extra_scales(self, dets_at_1, chosen, *,
                                max_scale: float, min_scale: float,
                                image_shape: tuple[int, int] = (0, 0),
                                ) -> list[float]:
        """Pick up to 3 extra scales based on what scale=1.0 found.

        Returns a list of scales (execution order). May be empty when
        scale=1.0 gave comfortable coverage, or when the native pass already
        decodes the board (see ``_UPSCALE_SKIP_MODULE_PX``).

        Only detections whose id belongs to this board count towards the
        rate and module sizes (tags of another board in view must not make
        this one look decoded).
        """
        dets_at_1 = [d for d in dets_at_1
                     if 0 <= d.tag_id - self._start_id < self._num_tags]
        if not dets_at_1:
            return self._hail_mary_scales(max_scale)

        sizes_px = _compute_tag_sizes(dets_at_1)
        if sizes_px.size == 0:
            return self._hail_mary_scales(max_scale)

        modules_px = sizes_px / self._TAG_MODULES_PER_SIDE
        smallest = float(np.min(modules_px))
        largest = float(np.max(modules_px))
        median_module = float(np.median(modules_px))
        n_detected = len(dets_at_1)
        rate = n_detected / max(self._num_tags, 1)

        # Early termination: enough tags + comfortable size distribution.
        if (rate >= self._EARLY_STOP_DETECTION_RATE
                and smallest >= self._MODULE_PX_MIN
                and largest <= self._DOWNSCALE_TRIGGER_MODULE_PX):
            return []

        # ----- FoV clipping check -----
        # If detected tags touch the image edge, the board extends beyond
        # the visible frame. Missing tags are clipped, not small — extra
        # scales won't help. Use the average tag size as the edge margin.
        fov_clipped = False
        if image_shape[0] > 0 and image_shape[1] > 0 and dets_at_1:
            img_h, img_w = image_shape
            avg_tag_px = float(np.median(sizes_px))
            margin = max(avg_tag_px, 0.05 * max(img_w, img_h))
            all_corners = np.vstack(
                [np.asarray(d.corners).reshape(-1, 2) for d in dets_at_1])
            u_min, v_min = all_corners.min(axis=0)
            u_max, v_max = all_corners.max(axis=0)
            fov_clipped = (u_min < margin or v_min < margin or
                           u_max > img_w - margin or v_max > img_h - margin)

        # ----- FoV clipping guard -----
        # If the board is clipped by the image edge AND we already found
        # a reasonable fraction of tags (≥30%), the missing ones are
        # outside the frame — not too small. Skip upscale, only try
        # downscale if modules are oversized.
        if fov_clipped and rate >= 0.3:
            extras: list[float] = []
            if largest > self._DOWNSCALE_TRIGGER_MODULE_PX:
                s_down = self._MODULE_PX_TARGET / largest
                s_down = max(s_down, min_scale)
                if s_down < 0.85:
                    extras.append(round(s_down, 2))
            if extras:
                logger.debug(
                    f"FoV-clipped board (median_module={median_module:.0f}px, "
                    f"rate={rate:.0%}) — skipping upscale, downscale only"
                )
            else:
                logger.debug(
                    f"FoV-clipped board (median_module={median_module:.0f}px, "
                    f"rate={rate:.0%}) — no extra scales needed"
                )
            return extras

        # ----- Normal adaptive scale selection -----
        extras: list[float] = []
        native_decodes = (self._UPSCALE_SKIP_MODULE_PX is not None
                          and rate >= self._MODERATE_DETECTION_RATE
                          and smallest >= self._UPSCALE_SKIP_MODULE_PX)
        # Skip upscale: the native pass already decodes the board.
        if native_decodes:
            logger.debug(
                f"native pass decodes the board (rate={rate:.0%}, smallest "
                f"module={smallest:.1f}px) — skipping upscale")
        # Upscale: bring smallest module into comfort zone.
        elif smallest < self._MODULE_PX_MIN or rate < self._EARLY_STOP_DETECTION_RATE:
            s_up = self._MODULE_PX_TARGET / max(smallest, 1.0)
            s_up = min(s_up, max_scale)
            if s_up > 1.2:
                extras.append(round(s_up, 2))
        # Downscale: bring largest module from > 50 px back to 25 px.
        if largest > self._DOWNSCALE_TRIGGER_MODULE_PX:
            s_down = self._MODULE_PX_TARGET / largest
            s_down = max(s_down, min_scale)
            if s_down < 0.85:
                extras.insert(0, round(s_down, 2))

        # 3rd adaptive scale: if the first two still leave us below 70%
        # coverage, add an intermediate scale between the most recent
        # upscale and the max_scale ceiling.
        if len(extras) >= 1 and rate < self._MODERATE_DETECTION_RATE:
            biggest_tried = max(extras)
            if biggest_tried < max_scale * 0.85:
                s_mid = round((biggest_tried + max_scale) / 2.0, 2)
                if s_mid not in extras:
                    extras.append(s_mid)
        return extras

    def _coverage_is_comfortable(self, chosen: dict) -> bool:
        """True when every chosen tag's smallest module (tag side / modules-per-side)
        is at least `_MODULE_PX_MIN` px in the original image — i.e. the adaptive
        upscale loop can stop trying larger scales."""
        if not chosen:
            return False
        # All chosen corners are in original-image pixels; estimate tag
        # edge length from corner 0 → corner 1 of each.
        edges = [float(np.linalg.norm(c[1] - c[0])) for c, _ in chosen.values()]
        if not edges:
            return False
        smallest_module = min(edges) / self._TAG_MODULES_PER_SIDE
        return smallest_module >= self._MODULE_PX_MIN

    def _regional_crop_detect(
        self, gray: np.ndarray, chosen: dict,
        max_scale: float,
        history: list,
    ):
        """Crop around detected tags and upscale just the crop to find
        more tags in the board neighbourhood.

        Only fires when:
        - Some tags were detected (partial detection)
        - The detected bounding box occupies < 1/3 of image area (far away)

        Returns (detections, scale, (ox, oy)) or None if not applicable.
        """
        import cv2

        if not chosen:
            return None

        # Bounding box of detected tag corners in original pixels.
        all_corners = np.vstack([c for c, _ in chosen.values()])
        u_min, v_min = all_corners.min(axis=0)
        u_max, v_max = all_corners.max(axis=0)
        bbox_w = u_max - u_min
        bbox_h = v_max - v_min
        img_h, img_w = gray.shape[:2]
        bbox_area = bbox_w * bbox_h
        img_area = img_w * img_h

        # Only crop if the detection region is < 1/3 of the image.
        if bbox_area >= img_area / 3.0:
            return None

        # Pad the bbox by 50% on each side to capture edge tags.
        pad_u = bbox_w * 0.5
        pad_v = bbox_h * 0.5
        x0 = max(0, int(u_min - pad_u))
        y0 = max(0, int(v_min - pad_v))
        x1 = min(img_w, int(u_max + pad_u))
        y1 = min(img_h, int(v_max + pad_v))

        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            return None

        # Upscale the crop so median tag module lands at ~25 px.
        crop_max_dim = max(crop.shape[:2])
        # Target: the crop upscaled should be ~800 px on the long side.
        target_crop_dim = 800.0
        crop_scale = min(target_crop_dim / max(crop_max_dim, 1),
                         max_scale)
        if crop_scale <= 1.2:
            return None

        crop_up = cv2.resize(crop, None, fx=crop_scale, fy=crop_scale,
                              interpolation=cv2.INTER_CUBIC)
        try:
            dets = self._get_detector().detect(crop_up)
        except cv2.error:
            return None

        n_new = sum(1 for d in dets
                    if (d.tag_id - self._start_id) not in chosen
                    and 0 <= d.tag_id - self._start_id < self._num_tags)
        if n_new > 0:
            history.append((round(crop_scale, 2), len(dets),
                            0.0))  # median_tag_px not meaningful for crop
            logger.debug(
                f"Regional crop ({x0},{y0})-({x1},{y1}) scale={crop_scale:.1f}x → "
                f"{n_new} new tags"
            )
        return dets, crop_scale, (float(x0), float(y0))

    @staticmethod
    def _history_entry(scale: float, dets,
                        scale_for_orig: float) -> tuple[float, int, float]:
        """One `(scale, n_detections, median_tag_px_in_original)` row for the
        detection-attempt history used in diagnostics."""
        n = len(dets)
        if n > 0:
            sizes = _compute_tag_sizes(dets)
            median_orig_px = float(np.median(sizes)) / max(scale_for_orig, 1e-6)
        else:
            median_orig_px = 0.0
        return (round(scale, 2), n, round(median_orig_px, 1))

    def point_3d(self, point_id: int) -> np.ndarray:
        """Board-frame 3D position of corner `point_id` (Board interface)."""
        return self._points_3d[point_id].copy()

    def num_points(self) -> int:
        """Total corner count: 4 per tag (Board interface)."""
        return self._num_tags * 4

    def min_points_for_pose(self) -> int:
        """Minimum detections for a usable PnP pose — one full tag (Board interface)."""
        return 4

    @classmethod
    def from_kalibr_yaml(
        cls,
        yaml_path: str,
        min_tag_px: float = MIN_TAG_PIXEL_SIZE,
        min_detection_rate: float = MIN_DETECTION_RATE,
    ) -> "AprilTagBoard":
        """Create an AprilTagBoard from a Kalibr target YAML file.

        Kalibr YAML format:
            target_type: 'aprilgrid'
            tagCols: 6
            tagRows: 6
            tagSize: 0.088       # meters
            tagSpacing: 0.3      # ratio of space to tagSize

        Args:
            yaml_path: Path to Kalibr target YAML file.
            min_tag_px: Minimum median tag pixel size for adaptive upscaling.
            min_detection_rate: Minimum detection rate before triggering upscaling.

        Returns:
            AprilTagBoard configured for the Kalibr target.
        """
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        rows = data["tagRows"]
        cols = data["tagCols"]
        tag_size = data["tagSize"]
        spacing_ratio = data["tagSpacing"]
        tag_spacing = tag_size * spacing_ratio

        return cls(
            rows=rows,
            cols=cols,
            tag_size=tag_size,
            tag_spacing=tag_spacing,
            tag_family="t36h11",
            min_tag_px=min_tag_px,
            min_detection_rate=min_detection_rate,
        )
