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
"""Rerun visualization for camera calibration.

Displays:
  1. Calibration board as 3D points (static)
  2. Reference camera pose history (rig moves through frames)
  3. Multi-camera rig via extrinsics (child transforms of ref camera)
  4. Per-camera detection thumbnails (child of pinhole entity)

Entity hierarchy (per Rerun docs: Transform3D + Pinhole on SAME entity,
Image as child):

  world/
  ├── board                      # Points3D: calibration target (static)
  ├── trajectory                 # Points3D + LineStrips3D: ref camera path
  └── camera/                    # Transform3D + Pinhole: ref camera
      ├── image                  # Image: ref camera thumbnail
      └── cam{id}/               # Transform3D: extrinsic offset
          └── (inherits parent pinhole for frustum)
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import rerun as rr
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False


class RerunVisualizer:
    """Stream calibration data to Rerun viewer."""

    def __init__(self, name: str = "mvis", spawn: bool = True):
        if not HAS_RERUN:
            logger.warning("rerun-sdk not installed.")
            self._enabled = False
            return
        self._enabled = True
        rr.init(name, spawn=spawn)
        logger.info(f"Rerun: '{name}' (spawn={spawn})")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log_board_points(self, board, entity: str = "world/board"):
        """Log 3D board points (static)."""
        if not self._enabled or board is None:
            return
        pts = board.all_points_3d()
        rr.log(entity, rr.Points3D(pts, colors=[200, 50, 50], radii=0.005))

    def log_detection(self, cam_id, timestamp, image, observations,
                      num_tags=36):
        """Log a single detection frame (detection phase only)."""
        if not self._enabled:
            return
        rr.set_time("timestamp", timestamp=timestamp)
        rr.log(f"detection/cam{cam_id}/image", rr.Image(image))
        if observations:
            corners = np.array([obs.pixel for obs in observations])
            tag_ids = np.array([obs.point_id // 4 for obs in observations])
            colors = np.zeros((len(corners), 3), dtype=np.uint8)
            for i, tid in enumerate(tag_ids):
                colors[i] = [(tid * 67) % 256, (tid * 137 + 80) % 256,
                             (tid * 43 + 160) % 256]
            rr.log(f"detection/cam{cam_id}/corners",
                    rr.Points2D(corners, colors=colors, radii=3.0))

    def log_calibration_result(self, result, all_frames, board,
                               image_sizes=None, detection_images=None):
        """Log post-optimization results.

        Args:
            result: CalibrationResult
            all_frames: camera_id -> list of FrameObservations (can be empty)
            board: Board object with all_points_3d()
            image_sizes: camera_id -> (width, height)
            detection_images: camera_id -> list of images
        """
        if not self._enabled:
            return

        import cv2

        image_sizes = image_sizes or {}
        detection_images = detection_images or {}
        cam_params = {c.camera_id: c for c in result.cameras}
        ref_cam = result.reference_camera

        # --- Display transform: [x,y,z] → [z,x,y] ---
        # Same as the HTML report's plot_3d.py so the board stands upright
        # on the YZ plane (board normal = +X), matching the interactive 3D.
        from mvis.analysis.display_frame import (
            to_display_points as _disp_pts, to_display_pose as _disp_pose)

        # --- 1. Board points (static, large and visible) ---
        if board is not None:
            pts = board.all_points_3d()
            pts_d = _disp_pts(pts)
            rr.log("world/board", rr.Points3D(
                pts_d, colors=[200, 50, 50], radii=0.015))

        # --- 2. Collect valid poses ---
        valid_poses = []
        for i, bp in enumerate(result.board_poses):
            if bp is None:
                continue
            bp_arr = np.asarray(bp)
            if bp_arr.shape != (4, 4) or np.allclose(bp_arr, np.eye(4)):
                continue
            t = result.board_pose_times[i] if i < len(result.board_pose_times) else float(i)
            valid_poses.append((i, bp_arr, t))

        if not valid_poses:
            logger.warning("No valid board poses to visualize")
            return

        logger.info(f"Rerun: logging {len(valid_poses)} poses, "
                    f"{sum(len(v) for v in detection_images.values())} images")

        # --- 3. Trajectory (static) ---
        # T_board_cam[:3, 3] = camera position in board frame (no inversion)
        positions = []
        for _, bp, _ in valid_poses:
            T_d = _disp_pose(bp)
            positions.append(T_d[:3, 3])
        positions_arr = np.array(positions)

        n = len(positions_arr)
        traj_colors = np.zeros((n, 3), dtype=np.uint8)
        for i in range(n):
            frac = i / max(n - 1, 1)
            traj_colors[i] = [30, int(120 + 135 * frac), 255]

        rr.log("world/trajectory",
               rr.Points3D(positions_arr, colors=traj_colors, radii=0.006))
        if n >= 2:
            rr.log("world/trajectory_line",
                   rr.LineStrips3D([positions_arr], colors=[[0, 150, 255]],
                                   radii=0.002))

        # --- 4. Build per-camera info (intrinsics scaled to thumbnail) ---
        ref_params = cam_params.get(ref_cam)
        if ref_params is None:
            logger.warning(f"No params for ref camera {ref_cam}")
            return

        # Image plane distance: ~5% of median camera-to-board distance
        dists = [np.linalg.norm(pos) for pos in positions]
        median_dist = float(np.median(dists)) if dists else 1.0
        img_plane_dist = max(0.02, median_dist * 0.05)

        def _thumb_pinhole(cam, cam_id):
            """Build scaled Pinhole params matching thumbnail size."""
            imgs = detection_images.get(cam_id, [])
            img_sz = image_sizes.get(cam_id, cam.image_size)
            if img_sz == (0, 0):
                img_sz = (640, 480)
            tw, th = img_sz
            for im in imgs:
                if im is not None:
                    th, tw = im.shape[:2]
                    break
            sx = tw / max(img_sz[0], 1)
            sy = th / max(img_sz[1], 1)
            return {
                "focal_length": [cam.fx * sx, cam.fy * sy],
                "principal_point": [cam.cx * sx, cam.cy * sy],
                "resolution": [tw, th],
                "image_plane_distance": img_plane_dist,
            }

        # Build extrinsic map: cam_id → T_ref_cam (inv of T_to_from)
        ext_map = {}
        for ext in result.extrinsics:
            # T_to_from = T_ref_cam → inv gives T_cam_ref
            # T_board_cam_i = T_board_cam_ref @ T_cam_ref_to_cam_i
            # T_cam_ref_to_cam_i = inv(T_ref_cam) = inv(T_to_from)...
            # Actually: plot_3d uses T_board_cam_ref @ inv(T_cam_ref)
            # where T_cam_ref is what we store.
            # Let's match plot_3d.py exactly:
            ext_map[ext.from_camera] = np.linalg.inv(np.asarray(ext.T_to_from))

        all_cam_ids = [ref_cam] + [cid for cid in sorted(ext_map.keys())]

        # --- Build timestamp → image index lookup per camera ---
        # Board poses are indexed by ref camera frames. For non-ref cameras,
        # find the image whose timestamp is closest to the board pose time.
        cam_ts_to_img_idx = {}
        for cam_id in all_cam_ids:
            cam_imgs = detection_images.get(cam_id, [])
            frames = all_frames.get(cam_id, [])
            if frames:
                # Use frame timestamps from detection cache
                timestamps = [f.timestamp for f in frames]
                cam_ts_to_img_idx[cam_id] = timestamps
            else:
                # No detection cache — assume images are 1:1 with ref poses
                cam_ts_to_img_idx[cam_id] = None

        def _find_closest_img(cam_id, target_ts):
            """Find the image index closest to target_ts for a camera."""
            ts_list = cam_ts_to_img_idx.get(cam_id)
            if ts_list is None:
                return None  # no timestamp info
            if not ts_list:
                return None
            # Binary search for closest timestamp
            idx = int(np.searchsorted(ts_list, target_ts))
            if idx >= len(ts_list):
                idx = len(ts_list) - 1
            if idx > 0 and abs(ts_list[idx - 1] - target_ts) < abs(ts_list[idx] - target_ts):
                idx = idx - 1
            # Only match if within 0.1s
            if abs(ts_list[idx] - target_ts) < 0.1:
                return idx
            return None

        # --- 5. Per-frame: all cameras with poses + pinhole + images ---
        for frame_idx, (pose_idx, bp, t) in enumerate(valid_poses):
            rr.set_time("frame", sequence=frame_idx)
            rr.set_time("timestamp", timestamp=t)

            # Ref camera pose in display frame
            T_ref_d = _disp_pose(bp)

            for cam_id in all_cam_ids:
                if cam_id == ref_cam:
                    T_d = T_ref_d
                    entity = "world/ref_camera"
                else:
                    T_board_cam_i = bp @ ext_map[cam_id]
                    T_d = _disp_pose(T_board_cam_i)
                    entity = f"world/cam{cam_id}"

                R = T_d[:3, :3]
                pos = T_d[:3, 3]
                cam = cam_params.get(cam_id)
                if cam is None:
                    continue

                ph = _thumb_pinhole(cam, cam_id)
                rr.log(entity, rr.Transform3D(translation=pos, mat3x3=R))
                rr.log(entity, rr.Pinhole(**ph))

                # Image: find closest by timestamp, not by pose index
                cam_imgs = detection_images.get(cam_id, [])
                if cam_id == ref_cam:
                    # Ref camera: pose_idx maps directly to frame index
                    img_idx = pose_idx
                else:
                    # Non-ref: find closest timestamp match
                    img_idx = _find_closest_img(cam_id, t)
                    if img_idx is None:
                        img_idx = pose_idx  # fallback

                if img_idx is not None and img_idx < len(cam_imgs) and cam_imgs[img_idx] is not None:
                    img = cam_imgs[img_idx]
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    rr.log(f"{entity}/image", rr.Image(img))

        # --- 6. Summary text ---
        for cam in result.cameras:
            rr.log("log", rr.TextLog(
                f"cam{cam.camera_id} ({cam.distortion_model}): "
                f"fx={cam.fx:.2f} fy={cam.fy:.2f}"))
        for ext in result.extrinsics:
            bl = float(np.linalg.norm(ext.translation)) * 1000
            rr.log("log", rr.TextLog(
                f"Extrinsic {ext.from_camera}->{ext.to_camera}: "
                f"baseline={bl:.1f}mm"))

    def log_text(self, text, level="INFO"):
        if not self._enabled:
            return
        rr.log("log", rr.TextLog(text, level=level))
