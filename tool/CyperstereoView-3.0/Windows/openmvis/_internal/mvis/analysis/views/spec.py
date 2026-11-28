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
"""Data for the report's canvas views: the calibration process and the solved rig.

`scene_spec` and `rig_spec` turn a cam-IMU result into plain JSON-able dicts that
`views.js` draws. `frames=False` leaves every camera image out when the spec is
built, so the copy written beside the report (`<report>_views.json`) can go to
the public sensor pages without frames ever being in it.

Spec version 2. The player refuses specs it doesn't know (`v`).
"""
from __future__ import annotations

import base64

import numpy as np

from mvis.analysis.views.geometry import fov_deg, project, rim
from mvis.calibration.factors.camera_projection import k_cal_vec

VERSION = 2
COLORS = ["#2563EB", "#DC2626", "#16A34A", "#9333EA", "#EA580C", "#0891B2", "#DB2777", "#65A30D"]
MAX_LANDMARKS = 3000


def _r(a, n=5):
    return np.round(np.asarray(a, float), n).tolist()


def board_outline(points_B: np.ndarray) -> np.ndarray:
    """The board's outer edge in its own frame (4 corners, z = 0): the bounding box
    of the points grown by one point spacing (the square for a chessboard's
    inner corners; about a gap for a grid of tags)."""
    p = np.asarray(points_B, float)[:, :2]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    m = float(np.median(d.min(axis=1))) if len(p) > 1 else 0.0
    lo, hi = p.min(axis=0) - m, p.max(axis=0) + m
    return np.array([[lo[0], lo[1], 0], [hi[0], lo[1], 0], [hi[0], hi[1], 0], [lo[0], hi[1], 0]])


def _pose_list(values):
    import gtsam
    X = gtsam.symbol_shorthand.X
    out, k = [], 0
    while values.exists(X(k)):
        out.append(values.atPose3(X(k)).matrix())
        k += 1
    return np.array(out) if out else np.zeros((0, 4, 4))


def _cal(e):
    cal = e["calib"]
    intr = np.asarray(cal.intr, float)
    return cal, k_cal_vec(cal.model, intr[:4], intr[4:])


def scene_spec(result, board=None, views=None, state_idx=None, landmarks=None,
               image_sizes=None, frames: bool = True) -> dict | None:
    """The calibration process: IMU poses over time, the board (or landmarks), and
    per camera its mounting, intrinsics, detections, coverage and - with
    `frames` - the panel images that also texture the frustum.

    `views` and `state_idx` are `cam_imu_report._cam_views` output (one dict per
    camera; per-step data aligned with `state_idx`). None without a trajectory."""
    values = getattr(result, "values", None)
    if values is None:
        return None
    T = _pose_list(values)
    if len(T) < 2:
        return None
    idx = np.asarray(state_idx if state_idx else range(len(T)), int)
    t_states = getattr(result, "t_states", None)
    t = (np.asarray(t_states, float)[idx] - float(t_states[idx[0]])
         if t_states is not None and len(t_states) == len(T) else None)
    spec = {"v": VERSION, "t": _r(t, 3) if t is not None else None,
            "unit": "s" if t is not None else "step",
            "xyz": _r(T[idx, :3, 3]), "R": _r(T[idx, :3, :3].reshape(len(idx), 9)), "cams": []}
    T_BinW = getattr(result, "T_BinW", None)
    outline_W = None
    if board is not None and T_BinW is not None:
        TB = T_BinW.matrix() if hasattr(T_BinW, "matrix") else np.asarray(T_BinW, float)
        bp = np.asarray(board.all_points_3d(), float)
        outline_B = board_outline(bp)
        outline_W = (TB[:3, :3] @ outline_B.T).T + TB[:3, 3]
        n = TB[:3, 2]
        spec["board"] = {"pts": _r((TB[:3, :3] @ bp.T).T + TB[:3, 3]), "outline": _r(outline_W),
                         "normal": _r(n), "center": _r(outline_W.mean(axis=0))}
    else:
        pts = np.asarray(landmarks, float) if landmarks is not None else np.zeros((0, 3))
        if len(pts) > MAX_LANDMARKS:
            pts = pts[np.linspace(0, len(pts) - 1, MAX_LANDMARKS).round().astype(int)]
        spec["landmarks"] = _r(pts, 4)
    cams = getattr(result, "cameras", None) or []
    for c, e in enumerate(cams):
        cal, cv = _cal(e)
        Tc = cal.T.matrix()
        size = list((image_sizes or {}).get(c) or e.get("image_size") or (640, 480))
        cam = {"name": f"cam{c}", "color": COLORS[c % len(COLORS)], "R": _r(Tc[:3, :3].reshape(9), 6),
               "p": _r(Tc[:3, 3], 6), "k": _r(cv[:4], 4), "size": [int(size[0]), int(size[1])]}
        v = (views or [None] * len(cams))[c] if c < len(views or []) else None
        panel = (v or {}).get("panel")
        if panel is not None and len(panel.get("uv", ())) == len(idx):
            uv = [np.asarray(u, np.int16) for u in panel["uv"]]
            off = np.concatenate([[0], np.cumsum([len(u) for u in uv])]).astype(int)
            cam["uvb"] = base64.b64encode(np.concatenate(uv or [np.zeros(0, np.int16)]).astype("<i2")
                                          .tobytes()).decode("ascii")
            cam["uvo"] = off.tolist()
            cam["cov"] = _r(panel.get("cov", []), 1)
            cam["n"] = [int(x) for x in panel.get("n", [])]
            if frames and panel.get("jpg"):
                cam["jpg"], cam["jpg_idx"] = list(panel["jpg"]), [int(x) for x in panel["jpg_idx"]]
                cam["jpg_size"] = [int(panel.get("w", 0)), int(panel.get("h", 0))]
        ids = (v or {}).get("ids")
        if ids is not None and len(ids) == len(idx):
            cam["ids"] = [list(map(int, x)) for x in ids]
        if outline_W is not None:   # the board edge as the camera sees it at each step: the drawn view
            out = []
            for k in idx:
                Twc = T[k] @ Tc
                Xc = (Twc[:3, :3].T @ (outline_W - Twc[:3, 3]).T).T
                q = [project(cal.model, cv, x) if x[2] > 1e-3 else None for x in Xc]
                out.append(None if any(p is None for p in q) else _r(np.array(q), 1))
            cam["outline"] = out
        spec["cams"].append(cam)
    return spec


def _domain(model: str) -> float | None:
    """The widest angle off axis (deg) the model projects in MVIS, when it has a limit."""
    from mvis.analysis.views.geometry import KB_DOMAIN_DEG, KB_MODELS
    return KB_DOMAIN_DEG if model in KB_MODELS else None


def _reach(e) -> float | None:
    """The widest angle off the optical axis (deg) at which the solve used a corner: past it the rim is past the fit."""
    ca = e.get("corner_accounting") or {}
    v = ca.get("max_theta_used_deg", ca.get("max_theta_deg"))
    return round(float(v), 2) if v is not None else None


def _sigmas(e) -> dict:
    """Lever 1-sigma (mm, IMU axes) and rotation 1-sigma (rad) from the marginals, when the solve has them."""
    sig = e.get("calib_sigma")
    if sig is None:
        return {}
    sig = np.asarray(sig, float)
    if sig.size < 6 or not np.all(np.isfinite(sig[:6])):
        return {}
    return {"sig_p": _r(sig[3:6] * 1e3, 4), "sig_r": _r(sig[0:3], 6)}


def _by_cam(d: dict | None, i: int):
    """A per-camera value keyed by index, its string, or "cam<i>"."""
    from mvis.analysis.reference import camera_value
    return camera_value(d, i)


def _rotvec_R(v) -> np.ndarray:
    from mvis.calibration.factors.lie_groups import exp_so3
    return np.asarray(exp_so3(np.asarray(v, float)), float)


def reference_rig(reference: dict | None, cams_R: list, n: int) -> dict | None:
    """The reference rig (factory or vendor values) for the ghost overlay, in MVIS's IMU frame, mm.

    `reference` keys (all optional): `label`; per camera `lever_mm` (camera origin in the IMU frame) and `rotvec`
    (log of R_CtoI, rad); or `camchain` (a Kalibr camchain-imucam YAML path; `T_cam_imu` per camera) with `order`
    (reference camera index for each MVIS camera); `align: cam0_rotation` rotates the whole reference about the IMU so
    its cam0 matches MVIS's cam0 rotation, for references whose IMU frame differs from MVIS's. `cams_R` are MVIS's
    R_CtoI per camera (3x3), for the alignment. None when nothing usable is given."""
    if not reference:
        return None
    ref = []
    if reference.get("camchain"):
        import yaml
        from mvis.io.kalibr import read_camchain_imucam
        doc = yaml.safe_load(open(str(reference["camchain"])).read())
        chain = read_camchain_imucam(doc)
        order = reference.get("order") or list(range(len(chain)))
        for i in range(n):
            j = order[i] if i < len(order) else None
            if j is None or j >= len(chain) or "T_cam_imu" not in chain[j]:
                ref.append(None)
                continue
            T = np.asarray(chain[j]["T_cam_imu"], float)
            R = T[:3, :3].T
            ref.append({"p": -R @ T[:3, 3] * 1e3, "R": R})
    else:
        for i in range(n):
            lever, rv = _by_cam(reference.get("lever_mm"), i), _by_cam(reference.get("rotvec"), i)
            if lever is None or np.ndim(lever) == 0:
                ref.append(None)
                continue
            ref.append({"p": np.asarray(lever, float), "R": _rotvec_R(rv) if rv is not None else None})
    if not any(ref):
        return None
    align = reference.get("align")
    if align == "cam0_rotation" and ref[0] is not None and ref[0]["R"] is not None and cams_R:
        A = np.asarray(cams_R[0], float) @ ref[0]["R"].T
        ref = [None if r is None else {"p": A @ r["p"], "R": None if r["R"] is None else A @ r["R"]} for r in ref]
    return {"label": str(reference.get("label", "reference")), "align": align,
            "cams": [None if r is None else {"p": _r(r["p"], 3), "R": None if r["R"] is None else _r(r["R"].reshape(9), 6)}
                     for r in ref]}


def rig_spec(result, image_sizes=None, imu=None, reference=None) -> dict | None:
    """The solved rig: each camera's mounting, intrinsics, timing and
    field-of-view rim; aux IMUs; base-IMU scale and t_ga (`imu`, from
    cam_imu_report._imu_block)."""
    cams = getattr(result, "cameras", None) or []
    if not cams:
        return None
    out = {"v": VERSION, "cams": [], "imus": [], "imu": {}}
    for c, e in enumerate(cams):
        cal, cv = _cal(e)
        Tc = cal.T.matrix()
        size = list((image_sizes or {}).get(c) or e.get("image_size") or (640, 480))
        out["cams"].append({
            "name": f"cam{c}", "color": COLORS[c % len(COLORS)], "model": cal.model,
            "p": _r(Tc[:3, 3] * 1e3, 3), "R": _r(Tc[:3, :3].reshape(9), 6),
            "intr": _r(np.asarray(cal.intr, float), 6), "size": [int(size[0]), int(size[1])],
            "td": round(float(cal.td) * 1e3, 3), "tr": round(float(cal.tr) * 1e3, 3),
            "reproj": (round(float(e["reproj_rms_px"]), 3) if e.get("reproj_rms_px") is not None else None),
            "reproj_med": _r3(e.get("reproj_median_px")), "reproj_p95": _r3(e.get("reproj_p95_px")),
            "rim": rim(cal.model, cv, size, per_side=12), "fov": fov_deg(cal.model, cv, size),
            "reach": _reach(e), "domain": _domain(cal.model), **_sigmas(e)})
    for m in getattr(result, "imus", None) or []:
        T = m.get("T_IstoI")
        if T is None:
            continue
        M = T.matrix() if hasattr(T, "matrix") else np.asarray(T, float)
        out["imus"].append({"name": m.get("name", "aux"), "p": _r(M[:3, 3] * 1e3, 3), "R": _r(M[:3, :3].reshape(9), 6),
                            "td": round(float(m.get("td", 0.0)) * 1e3, 3)})
    try:
        ref = reference_rig(reference, [np.asarray(e["calib"].T.matrix(), float)[:3, :3] for e in cams], len(cams))
    except Exception:   # a bad reference never breaks the report
        ref = None
    if ref:
        out["ref"] = ref
    if imu:
        out["imu"] = {"Dw": imu.get("Dw_diag"), "Da": imu.get("Da_diag"),
                      "tga": (imu["t_ga"] * 1e3 if imu.get("t_ga") is not None else None)}
    return out


FRAME_KEYS = ("jpg", "jpg_idx", "jpg_size")


def without_frames(spec: dict | None) -> dict | None:
    """A scene spec with every camera image removed: what may leave the machine.
    Raises if anything that looks like image data is left."""
    if spec is None:
        return None
    out = dict(spec)
    out["cams"] = [{k: v for k, v in c.items() if k not in FRAME_KEYS} for c in spec.get("cams", [])]
    if any(_looks_like_image(x) for x in _strings(out)):
        raise ValueError("view spec still holds image data")
    return out


def _strings(o):
    if isinstance(o, dict):
        for x in o.values():
            yield from _strings(x)
    elif isinstance(o, list):
        for x in o:
            yield from _strings(x)
    elif isinstance(o, str):
        yield o


def _looks_like_image(s: str) -> bool:
    """A JPEG or PNG in base64, or a data URL. Only at the start of a string: packed number arrays
    (`uvb`) can hold these letters by chance."""
    return s.startswith(("/9j/", "iVBOR", "data:image/"))


# ---------------------------------------------------------------- camera-only reports
# The cam-only world is the board frame (board on z = 0). Shown with the board stood up like a wall, the same display
# rotation the Plotly cam-only plots used (plot_3d._T_DISP): display x, y, z = board z, x, y.
_D = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def _cam_cal(cam):
    return k_cal_vec(cam.distortion_model, np.asarray(cam.intrinsics, float)[:4], np.asarray(cam.distortion, float))


def cam_only_scene_spec(poses, times, board, cameras, ext_by_id, ref_id, views=None) -> dict | None:
    """The calibration process of a camera-only solve: the reference camera's path around the board.

    `poses` are the reference camera's T_board_cam, `times` their stamps (or None); `ext_by_id[c]` is camera c's
    T_to_from (c -> reference), so c sits at inv(ext) in the reference camera's frame. `views` is
    `pose_images.camera_views` output (reference camera first): panels, detected ids and - in the report - frames."""
    poses = [np.asarray(T, float) for T in poses]
    if len(poses) < 2:
        return None
    t = None
    if times is not None and len(times) == len(poses):
        t = np.asarray(times, float) - float(times[0])
    R = np.array([_D @ T[:3, :3] for T in poses])
    xyz = np.array([_D @ T[:3, 3] for T in poses])
    bp = np.asarray(board.all_points_3d(), float)
    outline_B = board_outline(bp)
    outline = (_D @ outline_B.T).T
    spec = {"v": VERSION, "t": _r(t, 3) if t is not None else None, "unit": "s" if t is not None else "frame",
            "xyz": _r(xyz), "R": _r(R.reshape(len(R), 9)), "origin": "camera",
            "board": {"pts": _r((_D @ bp.T).T), "outline": _r(outline), "normal": _r(_D @ np.array([0.0, 0.0, 1.0])),
                      "center": _r(outline.mean(axis=0))}, "cams": []}
    by_id = {v.get("camera_id"): v for v in (views or [])}
    order = sorted(cameras, key=lambda c: c.camera_id != ref_id)
    for k, cam in enumerate(order):
        cid = cam.camera_id
        if cid == ref_id:
            Tc = np.eye(4)
        elif cid in (ext_by_id or {}):
            Tc = np.linalg.inv(np.asarray(ext_by_id[cid], float))
        else:
            continue
        cv = _cam_cal(cam)
        size = [int(cam.image_size[0]), int(cam.image_size[1])]
        c = {"name": f"cam{cid}", "color": COLORS[k % len(COLORS)], "R": _r(Tc[:3, :3].reshape(9), 6), "p": _r(Tc[:3, 3], 6),
             "k": _r(cv[:4], 4), "size": size}
        v = by_id.get(cid) or {}
        panel = v.get("panel")
        if panel is not None and len(panel.get("uv", ())) == len(poses):
            uv = [np.asarray(u, np.int16) for u in panel["uv"]]
            c["uvb"] = base64.b64encode(np.concatenate(uv or [np.zeros(0, np.int16)]).astype("<i2").tobytes()).decode("ascii")
            c["uvo"] = np.concatenate([[0], np.cumsum([len(u) for u in uv])]).astype(int).tolist()
            c["cov"], c["n"] = _r(panel.get("cov", []), 1), [int(x) for x in panel.get("n", [])]
            if panel.get("jpg"):
                c["jpg"], c["jpg_idx"] = list(panel["jpg"]), [int(x) for x in panel["jpg_idx"]]
                c["jpg_size"] = [int(panel.get("w", 0)), int(panel.get("h", 0))]
        if v.get("ids") is not None and len(v["ids"]) == len(poses):
            c["ids"] = [list(map(int, x)) for x in v["ids"]]
        out = []
        for T in poses:   # board edge in camera c at each step: the drawn view
            Tbc = T @ Tc
            Xc = (Tbc[:3, :3].T @ (outline_B - Tbc[:3, 3]).T).T
            q = [project(cam.distortion_model, cv, x) if x[2] > 1e-3 else None for x in Xc]
            out.append(None if any(u is None for u in q) else _r(np.array(q), 1))
        c["outline"] = out
        spec["cams"].append(c)
    return spec


def _r3(v):
    return None if v is None else round(float(v), 3)


def cam_only_rig_spec(cameras, ext_by_id, ref_id, reproj=None) -> dict | None:
    """The solved rig of a multi-camera camera-only solve, in the reference camera's frame (mm). None for one camera.
    reproj: camera id -> (median, P95) px, for the rig table."""
    order = sorted(cameras, key=lambda c: c.camera_id != ref_id)
    if len(order) < 2:
        return None
    out = {"v": VERSION, "origin": f"cam{ref_id}", "cams": [], "imus": [], "imu": {}}
    for k, cam in enumerate(order):
        cid = cam.camera_id
        if cid == ref_id:
            Tc = np.eye(4)
        elif cid in (ext_by_id or {}):
            Tc = np.linalg.inv(np.asarray(ext_by_id[cid], float))
        else:
            continue
        cv = _cam_cal(cam)
        size = [int(cam.image_size[0]), int(cam.image_size[1])]
        out["cams"].append({
            "name": f"cam{cid}", "color": COLORS[k % len(COLORS)], "model": cam.distortion_model,
            "p": _r(Tc[:3, 3] * 1e3, 3), "R": _r(Tc[:3, :3].reshape(9), 6),
            "intr": _r(np.concatenate([np.asarray(cam.intrinsics, float)[:4], np.asarray(cam.distortion, float)]), 6),
            "size": size, "td": 0.0, "tr": round(float(getattr(cam, "rolling_shutter_readout", 0.0)) * 1e3, 3),
            "reproj": None,     # camera-only rows carry no RMS here; the table shows median / P95
            "reproj_med": _r3((reproj or {}).get(cid, (None, None))[0]),
            "reproj_p95": _r3((reproj or {}).get(cid, (None, None))[1]),
            "rim": rim(cam.distortion_model, cv, size, per_side=12), "fov": fov_deg(cam.distortion_model, cv, size), "reach": None,
            "domain": _domain(cam.distortion_model)})
    return out
