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
"""Camera-intrinsics input adapter.

Read a prior camera calibration as input to the cam-IMU pipeline, in BOTH formats:
  * MVIS `*_calib.yaml`  — top-level `cameras: [{id,model,intrinsics{fx..},
    distortion{model,coeffs},image_size,rolling_shutter_readout}]` (+ extrinsics)
  * Kalibr `camchain[-imucam].yaml` — `cam0: {camera_model,distortion_model,
    distortion_coeffs,intrinsics,resolution,rostopic[,T_cam_imu,T_cn_cnm1]}`

Returns a normalized list (cam order) of dicts the cam-IMU pipeline consumes:
  {rostopic, intrinsics[fx,fy,cx,cy], distortion_model, distortion_coeffs,
   resolution[w,h], is_fisheye, rolling_shutter_readout}
"""
from __future__ import annotations
import re

import yaml

from mvis.core.camera_models import (FISHEYE_ALIASES, KB4_ALIASES, is_removed_model,
                                     removed_model_message)

# MVIS `model` / Kalibr `camera_model`+`distortion_model` -> internal id.
# The wide-FOV models are matched FIRST (their names embed substrings like
# "radtan"/"sphere"), then the generic pinhole distortions.
_EQUIDISTANT = {"equidistant"}            # only its own name; generic fisheye names are kb4
_RADTAN = {"radtan", "plumb_bob", "radial-tangential"}
# ucm as a whole token (not inside 'eucm'), plus the MVIS-side omni-none spellings;
# a Kalibr omni + none PAIR is resolved in `_from_kalibr`, not here
_UCM = re.compile(r"(^|[^a-z])(ucm|unified)([^a-z]|$)|omni[_-]none")


def _norm_model(name: str) -> str:
    n = (name or "").lower()
    if is_removed_model(n):                           # first: 'kannala_brandt8' embeds
        raise ValueError(removed_model_message(name)) # 'kannala_brandt', 'omni_kb8' 'omni'
    if _UCM.search(n):                                # BEFORE omni: 'omni_none'
        return "ucm"
    if "omni" in n:                                   # MEI + radtan (Kalibr `omni`)
        return "omni_radtan"
    if "eucm" in n:
        return "eucm"
    if "double_sphere" in n or "double sphere" in n or "ds" in n.split():
        return "double_sphere"
    if "fov" in n:
        return "fov"
    if any(k in n for k in KB4_ALIASES):              # BEFORE the generic names: 'kannala_brandt4'
        return "kb4"                                  # embeds 'kannala_brandt'
    if any(k in n for k in _EQUIDISTANT):         # BEFORE the generic names: 'fisheye equidistant'
        return "equidistant"
    if any(k in n for k in FISHEYE_ALIASES):
        return "kb4"
    if any(k in n for k in _RADTAN):
        return "radtan"
    raise ValueError(f"unsupported/unknown distortion model {name!r} (cam-IMU "
                     "supports radtan / equidistant / fov / double_sphere / eucm "
                     "/ ucm / omni_radtan / kb4)")


def _dist_len(model: str) -> int:
    """Number of distortion coefficients for `model` (the model is authoritative —
    cal_vec length must be 4 + this, so the cam-IMU CameraCalib/factor dims agree).
    From the `core.camera_models` registry: radtan/equidistant = 4,
    omni_radtan = 5, kb4 = 4, double_sphere/eucm = 2, fov/ucm = 1."""
    from mvis.core.camera_models import n_dist
    return n_dist(model)


def _fit_coeffs(model: str, coeffs: list[float]) -> list[float]:
    """Trim/zero-pad distortion coeffs to the model's exact count. MVIS
    `_calib.yaml` zero-pads ds/eucm/fov coeffs to 4 (real params lead), and a legacy
    omni file has 4 ([xi,k1,k2,p1] -> p2 = 0 on load); fitting to
    the model length keeps cal_vec == n_intr so J_intr (model) matches H_calib.
    Non-zero extras raise (`core.camera_models.fit_dist`; radtan's OpenCV k3
    tail is still cut, with a warning)."""
    from mvis.core.camera_models import fit_dist
    return fit_dist(model, coeffs)


def load_camera_intrinsics(path: str, topic_template: str = "/{name}/image_raw") -> list[dict]:
    """Detect format and return normalized per-camera intrinsics (cam order)."""
    doc = yaml.safe_load(open(path))
    if isinstance(doc, dict) and "cameras" in doc:        # MVIS *_calib.yaml
        return _from_mvis(doc, topic_template)
    if isinstance(doc, dict) and any(k.startswith("cam") for k in doc):  # Kalibr camchain
        return _from_kalibr(doc)
    raise ValueError(f"unrecognized camera-calibration format: {path}")


def _from_mvis(doc, topic_template) -> list[dict]:
    out = []
    for cam in sorted(doc["cameras"], key=lambda c: c.get("id", 0)):
        intr = cam["intrinsics"]; dist = cam["distortion"]
        name = cam.get("name") or f"cam{cam.get('id', len(out))}"
        model = _norm_model(dist.get("model") or cam.get("model", ""))
        out.append(dict(
            rostopic=topic_template.format(name=name),
            intrinsics=[float(intr[k]) for k in ("fx", "fy", "cx", "cy")],
            distortion_model=model,
            distortion_coeffs=_fit_coeffs(model, dist["coeffs"]),
            resolution=[int(x) for x in cam["image_size"]],
            is_fisheye=model != "radtan",   # fov/ds/eucm/omni are fisheye too
            rolling_shutter_readout=float(cam.get("rolling_shutter_readout", 0.0)),
        ))
    return out


def kalibr_cam_keys(doc) -> list[str]:
    """The camera keys of a Kalibr camchain in camera order: `cam0, cam1, ..., cam10` by
    number (a plain string sort put cam10 before cam2), then any other `cam*` key by name."""
    def order(k):
        m = re.fullmatch(r"cam(\d+)", k)
        return (0, int(m.group(1)), "") if m else (1, 0, k)
    return sorted((k for k in doc if isinstance(k, str) and k.startswith("cam")), key=order)


def _from_kalibr(doc) -> list[dict]:
    """Kalibr's model is split across `camera_model` (pinhole/omni/ds/eucm) and
    `distortion_model` (radtan/equidistant/fov/none), and the wide-FOV models pack
    their extra params differently than our flat cal_vec [fx,fy,cx,cy, dist...]:
      * pinhole+{radtan,equidistant,fov}: intrinsics=[fx,fy,cx,cy], dist=coeffs
      * ds:   intrinsics=[xi,alpha,fx,fy,cx,cy] (projection params FIRST), no distortion
      * eucm: intrinsics=[alpha,beta,fx,fy,cx,cy] (projection params FIRST), no distortion
      * omni: intrinsics=[xi,fx,fy,cx,cy] (xi FIRST) + radtan coeffs[k1,k2,p1,p2]
              -> our omni_radtan = [fx,fy,cx,cy, xi,k1,k2,p1,p2] (a legacy
              3-coeff list loads with p2 = 0)
      * omni + none: intrinsics=[xi,gamma_u,gamma_v,cx,cy] is the UCM in xi form
              -> our ucm = [gamma_u/(1+xi), gamma_v/(1+xi), cx, cy, xi/(1+xi)]
      * kb4 exports as pinhole + equidistant (same k1..k4), so it reads back as
        equidistant; a hand-made pinhole/kb4 camchain is read leniently as
        [fx,fy,cx,cy] + coeffs
    """
    out = []
    for k in kalibr_cam_keys(doc):
        cc = doc[k]
        cam_m = (cc.get("camera_model") or "").lower()
        dist_m = (cc.get("distortion_model") or "").lower()
        intr = [float(x) for x in cc["intrinsics"]]
        coeffs = [float(x) for x in cc.get("distortion_coeffs", [])]
        if cam_m == "omni" and dist_m in ("none", ""):
            model = "ucm"
        else:
            model = _norm_model(f"{cam_m} {dist_m}")
        if model == "ucm" and cam_m == "omni":
            if any(coeffs):
                raise ValueError(f"{k}: omni + none with non-zero distortion_coeffs {coeffs}")
            xi = intr[0]
            fxfycxcy = [intr[1] / (1 + xi), intr[2] / (1 + xi), intr[3], intr[4]]
            extra = [xi / (1 + xi)]
        elif model in ("double_sphere", "eucm"):
            fxfycxcy, extra = intr[2:6], intr[:2]
        elif model == "omni_radtan":
            fxfycxcy = intr[1:5]
            extra = [intr[0]] + coeffs[:4]            # xi, k1, k2, p1, p2
        else:                                          # pinhole + radtan/equi/fov (/kb4)
            fxfycxcy, extra = intr[:4], coeffs
        out.append(dict(
            rostopic=cc["rostopic"],
            intrinsics=fxfycxcy,
            distortion_model=model,
            distortion_coeffs=_fit_coeffs(model, extra),
            resolution=[int(x) for x in cc["resolution"]],
            is_fisheye=model != "radtan",   # fov/ds/eucm/omni are fisheye too
            rolling_shutter_readout=float(cc.get("line_delay", 0.0)) * int(cc["resolution"][1]),
        ))
    return out
