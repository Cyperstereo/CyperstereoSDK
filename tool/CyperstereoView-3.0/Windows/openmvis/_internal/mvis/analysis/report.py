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
"""Generate calibration reports."""

from mvis.core.camera_models import KB_THETA_MODELS, dist_names
from mvis.core.types import CalibrationResult


def _distortion_lines(model: str, distortion) -> list[str]:
    """Per-coefficient lines of the 'Distortion (<model>):' block: registry names and
    significant digits for a named model, the historical 'd{i} = {:.6f}' lines otherwise."""
    names = dist_names(model)
    if names is not None and len(names) == len(distortion):
        return [f"    {n} = {c:.6g}" for n, c in zip(names, distortion)]
    return [f"    d{i} = {c:.6f}" for i, c in enumerate(distortion)]


def print_report(result: CalibrationResult) -> str:
    """Generate a text report of calibration results.

    Args:
        result: Calibration result.

    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("MVIS — Calibration Report")
    lines.append("=" * 60)

    for cam in result.cameras:
        lines.append(f"\ncam{cam.camera_id}:")
        lines.append(f"  Model: pinhole + {cam.distortion_model}")
        lines.append(f"  Image size: {cam.image_size[0]} x {cam.image_size[1]}")
        lines.append(f"  Intrinsics:")
        lines.append(f"    fx = {cam.fx:.4f}")
        lines.append(f"    fy = {cam.fy:.4f}")
        lines.append(f"    cx = {cam.cx:.4f}")
        lines.append(f"    cy = {cam.cy:.4f}")
        lines.append(f"  Distortion ({cam.distortion_model}):")
        lines.extend(_distortion_lines(cam.distortion_model, cam.distortion))
        if cam.distortion_model in KB_THETA_MODELS:
            from mvis.camera.kb_domain import (format_kb_valid_domain, kb_domain_tag,
                                               kb_valid_domain)
            info = kb_valid_domain(list(cam.intrinsics) + list(cam.distortion),
                                    cam.image_size)
            tag = kb_domain_tag(info)
            tag = f"  [{tag}]" if tag else ""
            lines.append(f"  {format_kb_valid_domain(info, cam.distortion_model)}{tag}")
        acc = (getattr(result, "corner_accounting", None) or {}).get(cam.camera_id)
        if acc:
            from mvis.calibration.corner_accounting import accounting_text
            lines.append(f"  {accounting_text(acc)}")
        if cam.rolling_shutter_readout > 0:
            lines.append(f"  Rolling shutter readout: {cam.rolling_shutter_readout:.6f} s")

    if result.extrinsics:
        lines.append(f"\nExtrinsics (reference camera: {result.reference_camera}):")
        for ext in result.extrinsics:
            lines.append(f"  cam{ext.from_camera} -> cam{ext.to_camera}:")
            lines.append(f"    Rotation (SO3): [{ext.so3[0]:.6f}, {ext.so3[1]:.6f}, {ext.so3[2]:.6f}]")
            lines.append(f"    Translation: [{ext.translation[0]:.6f}, {ext.translation[1]:.6f}, {ext.translation[2]:.6f}]")

    if result.time_offsets:
        lines.append(f"\nTime offsets (ref camera: {result.reference_camera}):")
        for cam_id, dt in result.time_offsets.items():
            lines.append(f"  cam{cam_id}: dt = {dt:.6f} s")

    lines.append(f"\nReprojection Errors:")
    for err in result.reprojection_errors:
        lines.append(f"  cam{err.camera_id}:")
        lines.append(f"    Mean:   {err.mean:.4f} px")
        lines.append(f"    Median: {err.median:.4f} px")
        lines.append(f"    Std:    {err.std:.4f} px")
        lines.append(f"    P90:    {err.p90:.4f} px")
        lines.append(f"    P95:    {err.p95:.4f} px")
        lines.append(f"    Max:    {err.max:.4f} px")
        lines.append(f"    Observations: {err.num_observations}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    return report
