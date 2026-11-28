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
"""Valid image domain of a theta-direct Kannala-Brandt fit (`KB_THETA_MODELS`).

theta_d(theta) is invertible only up to its first fold (d theta_d / d theta = 0;
pi when monotonic), so pixels beyond radius f * theta_d(fold) from (cx, cy) have
no ray. The polynomial is fit only inside the observed field, so a fold inside
the image is possible on a fit that is fine where it saw corners: it is
metadata and a warning, never a guardrail, and it does not stop a solve.
"""

import logging
import math

from mvis.calibration.factors.custom_projections import _kb_fold, _kb_poly

logger = logging.getLogger(__name__)


def kb_valid_domain(cal_vec, image_size=None) -> dict:
    """Valid image domain of a KB-N fit. Returns

      fold_theta_rad   first fold angle, None when monotonic on (0, pi)
      max_theta_rad    edge of the valid domain (fold, else pi)
      max_radius_px    min(fx, fy) * theta_d(max_theta) (min f: conservative)
      corner_radius_px farthest image corner from (cx, cy); None without a size
      covers_image     max_radius_px >= corner_radius_px (None without a size)
    """
    v = [float(x) for x in cal_vec]
    fx, fy, cx, cy = v[:4]
    ks = tuple(v[4:])
    fold = _kb_fold(ks)
    folds = fold < math.pi
    p, _dp = _kb_poly(fold, ks)
    r_max = min(abs(fx), abs(fy)) * fold * p
    corner = None
    if image_size is not None and min(int(image_size[0]), int(image_size[1])) > 0:
        w, h = float(image_size[0]), float(image_size[1])
        corner = max(math.hypot(u - cx, vv - cy)
                     for u in (0.0, w) for vv in (0.0, h))
    return {
        "fold_theta_rad": float(fold) if folds else None,
        "max_theta_rad": float(fold),
        "max_radius_px": float(r_max),
        "corner_radius_px": None if corner is None else float(corner),
        "covers_image": None if corner is None else bool(r_max >= corner),
    }


def format_kb_valid_domain(info: dict, model: str) -> str:
    """One-line text form of `kb_valid_domain` for logs and text reports."""
    th = info["fold_theta_rad"]
    edge = (f"first fold at theta {math.degrees(th):.1f} deg" if th is not None
            else "theta_d monotonic up to theta 180 deg")
    s = f"{model} valid domain: {edge} -> radius {info['max_radius_px']:.1f} px"
    if info["corner_radius_px"] is not None:
        s += f" (image corner {info['corner_radius_px']:.1f} px)"
    return s


def kb_domain_tag(info: dict) -> str:
    """Short report tag for `kb_valid_domain`: '' when the domain covers the
    image (or no image size), else whether a real fold ends it inside the
    image or theta_d is monotonic and only theta = 180 deg falls short."""
    if info["covers_image"] is not False:
        return ""
    if info["fold_theta_rad"] is not None:
        return "FOLDS INSIDE IMAGE"
    return "VALID DOMAIN ENDS INSIDE IMAGE"


def fold_level(info: dict, n_past_fold: int = 0) -> str:
    """'alarm': observed corners past the fold (report alarm, the solve still
    finishes). 'warn': the valid domain ends inside the image but past the data.
    'none': it covers the image, or there is no image size to judge by."""
    if n_past_fold > 0:
        return "alarm"
    return "warn" if info["covers_image"] is False else "none"


def check_kb_valid_domain(cal_vec, image_size=None, model: str = "kb4",
                          label: str | None = None,
                          log: logging.Logger | None = None) -> dict:
    """Compute + log `kb_valid_domain` after a solve; a WARNING (not a failure)
    when the valid domain ends inside the image (pixels past the fold radius
    cannot be unprojected)."""
    log = log or logger
    info = kb_valid_domain(cal_vec, image_size)
    msg = f"{label or model}: {format_kb_valid_domain(info, model)}"
    if info["covers_image"] is False and info["fold_theta_rad"] is not None:
        log.warning(f"{msg} — the fold lies INSIDE the image: pixels beyond "
                    f"{info['max_radius_px']:.1f} px from (cx, cy) have no ray "
                    f"(polynomial extrapolated past the observed field)")
    elif info["covers_image"] is False:
        log.warning(f"{msg} — no fold, but the valid domain ends at theta = "
                    f"180 deg INSIDE the image: pixels beyond "
                    f"{info['max_radius_px']:.1f} px from (cx, cy) have no ray")
    else:
        log.info(msg)
    return info
