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
"""Digital twin of a real cam-IMU recording, kept small enough to ship as a test fixture.

A real solve, moved off by a seeded perturbation (`perturb`), is the truth: its
calibration, board pose and shape, biases and IMU-filled trajectory. The perturbation keeps real
device calibrations out of shipped twins and makes a solve recover values it was not given. The
real frame times, per-frame corner ids, IMU stamps, camera prior and solve settings are kept.
`simulate` regenerates corners and IMU from that truth, including rolling shutter and the
gyro-to-accel offset t_ga, and `solve` runs the real pipeline on them and reports the errors
against the truth.

    twin = Twin.load("tests/data/twins/twin_gs.npz")
    report = solve(twin, vision_pose="imu", seed=0)

`capture` builds a twin from a config (through `capture_solve`); `Twin.window` cuts it down
to a fixture-sized window (maintainer tooling exports the shipped twins this way, not part of
the package). The simulated noise is Gaussian and there are no lens or
timing model errors, so results are a check of the estimator, not of a real recording.
"""
from __future__ import annotations

import dataclasses
import json
import os
import zlib
from dataclasses import dataclass, field

import gtsam
import numpy as np

from mvis.calibration import pipeline_cam_imu as P
from mvis.core.gravity import of_result
from mvis.calibration.factors.imu_segment import ImuSegment
from mvis.calibration.factors.lie_groups import exp_so3, log_so3
from mvis.imu import create_imu_model, parse_imu_model
from mvis.imu.imu_model import ImuModel
from mvis.simulation.camera_measurements import project_points_rolling
from mvis.simulation.from_result import cal_from_vec, capture_gt, fit_bspline_from_states
from mvis.simulation.imu_measurements import simulate_imu

KNOT_DT = 0.005            # spline knots of the truth trajectory (s)
EDGE_S = 0.2               # keep simulated samples this far inside the trajectory (s)
_CAM_META = ("cal_vec", "is_fisheye", "model", "image_size", "pin_intrinsics",
             "estimate_readout", "rolling_mode", "tr_hint")
_CAM_OPTIONAL = ("factor_hz", "vision_pose")    # per-camera solve settings, only when the row sets them
# `perturb`: the truth moves off the real solve by uniform draws within these half-widths
PERTURB_SEED = 3
PILOT_SEED = 100          # noise draw of the pilot solve that sets a captured twin's IMU noise
PERTURB = dict(focal_rel=0.01,      # fx, fy: up to 1 %
               center_px=4.0,       # cx, cy
               dist_rel=0.05,       # each further intrinsic coefficient: up to 5 % of its value
               rot_rad=0.01,        # camera-to-IMU (and aux-to-IMU) rotation, rotvec per axis
               lever_m=0.005,       # their lever, per axis
               td_s=0.001,          # camera and aux td
               t_ga_s=0.0005)       # base and aux t_ga


def _cam_meta(cam: dict) -> dict:
    return {k: cam[k] for k in _CAM_META} | {k: cam[k] for k in _CAM_OPTIONAL if k in cam}


@dataclass
class Twin:
    """Everything a simulation and solve need; no images or real pixels."""
    name: str
    truth: dict                # cams[intr8, T_CtoI, td, tr, ...], xi24, imu_model, b_g, b_a,
                               # T_BinW, g_inW, t_ga, heights_m, and aux (when there are aux
                               # IMUs): [{name, xi24, T_IstoI, td, t_ga, b_g, b_a, gyro_only}]
    cams: list                 # t_cam, ids_list, plus the _CAM_META fields
    board_points: np.ndarray   # flat board corners as the solver sees them (N, 3)
    t_imu: np.ndarray
    t_traj: np.ndarray
    T_traj: np.ndarray         # (M, 4, 4) IMU-filled trajectory
    solve_config: dict         # CamImuCalibConfig fields that differ from the defaults
    sigma_px: float | list = 0.0   # noise that reproduces the real residual; a list is per camera
    imu_noise: dict = field(default_factory=dict)
    aux_t_imu: list = field(default_factory=list)   # real stamps of each aux IMU
    knot_dt: float = KNOT_DT   # spline knots of the truth trajectory (s); wider is smoother

    def window(self, t0: float, seconds: float) -> "Twin":
        """The same twin over [t0, t0 + seconds], in the recording's time."""
        t1 = t0 + seconds
        m = (self.t_traj >= t0) & (self.t_traj <= t1)
        assert m.sum() > 10, f"no trajectory in [{t0}, {t1}]"
        out = dataclasses.replace(self, t_traj=self.t_traj[m], T_traj=self.T_traj[m],
                                  t_imu=self.t_imu[(self.t_imu >= t0) & (self.t_imu <= t1)],
                                  aux_t_imu=[t[(t >= t0) & (t <= t1)] for t in self.aux_t_imu])
        cams = []
        for cam in self.cams:
            keep = np.nonzero((cam["t_cam"] >= t0) & (cam["t_cam"] <= t1))[0]
            cams.append(dict(cam, t_cam=cam["t_cam"][keep],
                             ids_list=[cam["ids_list"][i] for i in keep]))
        out.cams = cams
        return out

    def save(self, path: str) -> None:
        cams_meta, arrays = [], {}
        for c, cam in enumerate(self.cams):
            meta = _cam_meta(cam)
            meta["cal_vec"] = [float(v) for v in cam["cal_vec"]]
            meta["image_size"] = [int(v) for v in cam["image_size"]]
            cams_meta.append(meta)
            arrays[f"t_cam_{c}"] = np.asarray(cam["t_cam"], float)
            arrays[f"ids_{c}"] = np.concatenate(cam["ids_list"]).astype(np.int16)
            arrays[f"ids_len_{c}"] = np.array([len(i) for i in cam["ids_list"]], np.int16)
        for i, t in enumerate(self.aux_t_imu):
            arrays[f"aux_t_{i}"] = np.asarray(t, float)
        meta = dict(name=self.name, truth=self.truth, cams=cams_meta, solve_config=self.solve_config,
                    sigma_px=self.sigma_px, imu_noise=self.imu_noise, knot_dt=self.knot_dt)
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), board_points=self.board_points,
                            t_imu=self.t_imu, t_traj=self.t_traj,
                            T_traj=self.T_traj[:, :3, :4], **arrays)

    @classmethod
    def load(cls, path: str) -> "Twin":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            cams = []
            for c, cm in enumerate(meta["cams"]):
                ids = np.split(z[f"ids_{c}"].astype(int), np.cumsum(z[f"ids_len_{c}"])[:-1])
                cams.append(dict(cm, cal_vec=np.asarray(cm["cal_vec"], float),
                                 image_size=tuple(cm["image_size"]),
                                 t_cam=z[f"t_cam_{c}"], ids_list=ids))
            T = np.zeros((len(z["T_traj"]), 4, 4))
            T[:, :3, :4] = z["T_traj"]
            T[:, 3, 3] = 1.0
            return cls(name=meta["name"], truth=meta["truth"], cams=cams,
                       board_points=z["board_points"], t_imu=z["t_imu"], t_traj=z["t_traj"],
                       T_traj=T, solve_config=meta["solve_config"],
                       sigma_px=meta["sigma_px"], imu_noise=meta["imu_noise"],
                       aux_t_imu=[z[f"aux_t_{i}"] for i in range(len(meta["truth"].get("aux", [])))],
                       knot_dt=meta.get("knot_dt", KNOT_DT))


def _plain(v):
    """Numpy scalars and arrays to JSON types."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def dense_trajectory(result, t_imu, w_m, a_m, model):
    """IMU-filled trajectory of a solve: every IMU stamp inside each state interval, the pose a
    blend of the forward propagation from X(k) and the backward one from X(k+1)."""
    X, V = gtsam.symbol_shorthand.X, P._Vk
    vals = result.values
    ts = np.asarray(result.t_states, float)
    seg = ImuSegment(t_imu, w_m, a_m, model)
    intr = create_imu_model(model).retract(np.asarray(result.imu_intr_xi, float))
    g = np.array([0.0, 0.0, -of_result(result)])
    t_imu = np.asarray(t_imu, float)

    def state(k):
        T = vals.atPose3(X(k))
        vb = np.asarray(vals.atVector(V(k)), float)
        return T.rotation().matrix(), np.asarray(T.translation(), float), vb

    out_t, out_T = [], []
    for k in range(len(ts) - 1):
        t0, t1 = ts[k], ts[k + 1]
        R0, p0, vb0 = state(k)
        R1, p1, vb1 = state(k + 1)
        out_t.append(t0)
        out_T.append(np.asarray(vals.atPose3(X(k)).matrix(), float))
        inner = t_imu[(t_imu > t0 + 1e-6) & (t_imu < t1 - 1e-6)]
        if len(inner) == 0 or t1 - t0 > 0.3:          # long gap: geodesic only
            continue
        fwd = seg.propagate(t0, inner - t0, vb0[3:6], vb0[6:9], intr, jac=False)
        bwd = seg.propagate(t1, inner - t1, vb1[3:6], vb1[6:9], intr, jac=False)
        for t, F, B in zip(inner, fwd, bwd):
            dF, dB = t - t0, t - t1
            RF = R0 @ F.D_R
            pF = p0 + vb0[0:3] * dF + 0.5 * g * dF * dF + R0 @ F.D_p
            RB = R1 @ B.D_R
            pB = p1 + vb1[0:3] * dB + 0.5 * g * dB * dB + R1 @ B.D_p
            lam = (t - t0) / (t1 - t0)
            M = np.eye(4)
            M[:3, :3] = RF @ exp_so3(lam * log_so3(RF.T @ RB))
            M[:3, 3] = (1.0 - lam) * pF + lam * pB
            out_t.append(t)
            out_T.append(M)
    out_t.append(ts[-1])
    out_T.append(np.asarray(vals.atPose3(X(len(ts) - 1)).matrix(), float))
    return np.asarray(out_t), np.asarray(out_T)


def noise_from_solve(result, config) -> tuple[list, dict]:
    """(sigma_px per camera, imu_noise at the solver's sigmas) of a real solve. The camera RMS is of
    the 2D corner error, so one axis has RMS / sqrt(2). `capture` then scales the IMU noise with
    `imu_noise_scale`, so the twin's IMU fit matches the real one."""
    return ([float(e["reproj_rms_px"]) / np.sqrt(2.0) for e in result.cameras],
            dict(sigma_w=config.sigma_g, sigma_a=config.sigma_a,
                 sigma_bg=config.sigma_bg, sigma_ba=config.sigma_ba))


def imu_noise_scale(real_chi2: float, pilot_chi2: float) -> float:
    """Factor for the IMU noise so a twin's IMU chi2/dof matches the real solve's. Free states absorb
    most of the residual, so chi2/dof is far below 1 even with noise at the sigmas and can't be read
    as a noise level. It grows with the noise squared, so one pilot solve with noise at the sigmas
    gives the factor."""
    return float(np.sqrt(real_chi2 / pilot_chi2))


def _aux_truth(m: dict, aux: dict) -> dict:
    return dict(name=m["name"], xi24=[float(v) for v in m["intr_xi24"]],
                T_IstoI=np.asarray(m["T_IstoI"].matrix(), float).tolist(), td=float(m["td"]),
                t_ga=float(m.get("t_ga") or 0.0), b_g=[float(v) for v in m["b_g_s"]],
                b_a=[float(v) for v in m["b_a_s"]], gyro_only=bool(aux.get("gyro_only", False)))


def from_solve(name, result, cams, board_points, t_imu, w_m, a_m, config, sigma_px=0.0, imu_noise=None,
               aux_imus=None) -> Twin:
    """Twin from a finished `run_cam_imu_calibration_multi`: `result` is the truth."""
    aux_imus = aux_imus or []
    gt = capture_gt(result)
    model = parse_imu_model(gt.imu_model)
    t_d, T_d = dense_trajectory(result, t_imu, w_m, a_m, model)
    heights = result.diagnostics.get("board_heights") or {}
    truth = dict(cams=[dict(gt.cams[c], tr=float(e["calib"].tr)) for c, e in enumerate(result.cameras)],
                 xi24=gt.imus[0]["xi24"], imu_model=gt.imu_model, b_g=gt.b_g, b_a=gt.b_a,
                 T_BinW=gt.T_BinW, g_inW=gt.g_inW, t_ga=float(result.diagnostics.get("t_ga") or 0.0),
                 heights_m=[float(h) for h in heights.get("heights_m", np.asarray(board_points, float)[:, 2])])
    if aux_imus:
        truth["aux"] = [_aux_truth(m, a) for m, a in zip(result.imus, aux_imus)]
    default = type(config)()
    overrides = {f.name: _plain(getattr(config, f.name)) for f in dataclasses.fields(config)
                 if repr(getattr(config, f.name)) != repr(getattr(default, f.name))}
    keep = [_cam_meta(cam) | dict(t_cam=np.asarray(cam["t_cam"], float),
                                  ids_list=[np.asarray(i, int) for i in cam["ids_list"]])
            for cam in cams]
    return Twin(name=name, truth=truth, cams=keep, board_points=np.asarray(board_points, float),
                t_imu=np.asarray(t_imu, float), t_traj=t_d, T_traj=T_d, solve_config=overrides,
                sigma_px=sigma_px, imu_noise=dict(imu_noise or {}),
                aux_t_imu=[np.asarray(a["t_imu"], float) for a in aux_imus])


def capture_solve(cfg) -> dict:
    """Run `cli.calibrate_cam_imu(cfg)` and capture its `run_cam_imu_calibration_multi` call.

    Returns a dict with the solve result `r`, its inputs `cams`, `board_points`, `t_imu`, `w_m`,
    `a_m`, the solve `config` and `aux_imus` (as passed, not copied), the CLI `summary`, and the
    config's `cam_topics`, `src_path` and `src_format` (for camera labels). The solver is restored
    even when the run fails."""
    import mvis.cli.main as cli

    grab = dict(cam_topics=list(cfg.input.cam_topics or []), src_path=cfg.input.path,
                src_format=cfg.input.format)
    orig = P.run_cam_imu_calibration_multi

    def spy(cams, board_points, t_imu, w_m, a_m, **k):
        r = orig(cams, board_points, t_imu, w_m, a_m, **k)
        grab.update(r=r, cams=cams, board_points=board_points, t_imu=t_imu, w_m=w_m, a_m=a_m,
                    config=k.get("config"), aux_imus=k.get("aux_imus"))
        return r
    P.run_cam_imu_calibration_multi = spy
    try:
        grab["summary"] = cli.calibrate_cam_imu(cfg)
    finally:
        P.run_cam_imu_calibration_multi = orig
    return grab


def capture(config_path, name, sigma_px=0.0, imu_noise=None, work_dir="results/twin_capture",
            noise_from_residuals=False, vision_pose=None, knot_dt=KNOT_DT,
            perturb_seed: int | None = PERTURB_SEED) -> Twin:
    """Run the config's real solve (it needs the recording) and take it, through
    `perturb(twin, perturb_seed)`, as the truth of a new twin; `perturb_seed=None` keeps the real
    solve as it is.
    The CLI's outputs go under `work_dir`. `sigma_px` and `imu_noise` are the noise that reproduces
    the real residuals; `noise_from_residuals` measures them from this solve instead, with one extra
    pilot solve of the twin for the IMU (`imu_noise_scale`). `vision_pose` replaces the config's solve
    setting."""
    from mvis.core.config import CamImuRunConfig

    cfg = CamImuRunConfig.from_yaml(config_path)
    cfg.output.path = os.path.join(work_dir, "cam_imu.json")
    os.makedirs(os.path.dirname(cfg.output.path), exist_ok=True)
    with open(config_path) as f:
        cfg._source_yaml = f.read()
    if vision_pose is not None:
        cfg.solve = dataclasses.replace(cfg.solve, vision_pose=vision_pose)
    grab = capture_solve(cfg)
    if noise_from_residuals:
        sigma_px, imu_noise = noise_from_solve(grab["r"], grab["config"])
    twin = from_solve(name, grab["r"], grab["cams"], grab["board_points"], grab["t_imu"],
                      grab["w_m"], grab["a_m"], grab["config"], sigma_px=sigma_px,
                      imu_noise=imu_noise, aux_imus=grab["aux_imus"])
    if knot_dt != KNOT_DT:
        twin = dataclasses.replace(twin, knot_dt=knot_dt)
    if perturb_seed is not None:
        twin = perturb(twin, perturb_seed)
    if noise_from_residuals:
        pilot = solve(twin, vision_pose=None, seed=PILOT_SEED)
        s = imu_noise_scale(grab["r"].diagnostics["imu_chi2_dof"], pilot["imu_chi2_dof"])
        twin = dataclasses.replace(twin, imu_noise={k: v * s for k, v in twin.imu_noise.items()})
    return twin


def _jitter(rng, n, half_width) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, n) * half_width


def _move_pose(T, rng, scale) -> list:
    """T (4x4) turned by up to rot_rad per rotvec axis and shifted by up to lever_m per axis."""
    T = np.array(T, float)
    T[:3, :3] = T[:3, :3] @ exp_so3(_jitter(rng, 3, scale["rot_rad"]))
    T[:3, 3] += _jitter(rng, 3, scale["lever_m"])
    return T.tolist()


def perturb_rng_seed(seed: int, name: str) -> int:
    """The seed and a stable hash of the twin name (crc32, not the salted `hash`), as one int."""
    return int(seed) * 2 ** 32 + zlib.crc32(name.encode())


def perturb(twin: Twin, seed: int = PERTURB_SEED) -> Twin:
    """The twin with its truth moved off the real solve it came from, so no real
    device calibration ships and a solve must find values nobody gave it.

    Each draw is uniform in [-1, 1] times its half-width in `PERTURB`, from
    `np.random.default_rng(perturb_rng_seed(seed, twin.name))`, so each twin name gets its own
    draws: per camera in order, then base, then per aux IMU:
    intrinsics (focal relative, principal point px, every further coefficient relative), the
    camera-to-IMU rotation (rotvec rad per axis) and lever (m per axis), the camera td, the IMU
    t_ga, and each aux IMU's pose, td and t_ga. The camera prior `cal_vec` moves by the same
    intrinsics step, so the real gap between prior and truth is kept. Readout, IMU intrinsics,
    biases, gravity, the board and the trajectory are kept. `truth["perturbation"]` records the
    seed, the name, the effective rng seed and the scale; perturbing a twin twice raises."""
    if "perturbation" in twin.truth:
        raise ValueError(f"{twin.name} is already perturbed: {twin.truth['perturbation']}")
    scale = PERTURB
    rng_seed = perturb_rng_seed(seed, twin.name)
    rng = np.random.default_rng(rng_seed)
    truth = json.loads(json.dumps(twin.truth))      # deep copy
    cams = []
    for cam, cg in zip(twin.cams, truth["cams"]):
        intr = np.asarray(cg["intr8"], float)
        step = np.concatenate([intr[:2] * _jitter(rng, 2, scale["focal_rel"]),
                               _jitter(rng, 2, scale["center_px"]),
                               intr[4:] * _jitter(rng, len(intr) - 4, scale["dist_rel"])])
        cg["intr8"] = (intr + step).tolist()
        cg["T_CtoI"] = _move_pose(cg["T_CtoI"], rng, scale)
        cg["td"] = float(cg["td"]) + float(_jitter(rng, 1, scale["td_s"])[0])
        cams.append(dict(cam, cal_vec=np.asarray(cam["cal_vec"], float) + step))
    truth["t_ga"] = float(truth["t_ga"]) + float(_jitter(rng, 1, scale["t_ga_s"])[0])
    for at in truth.get("aux", []):
        at["T_IstoI"] = _move_pose(at["T_IstoI"], rng, scale)
        at["td"] = float(at["td"]) + float(_jitter(rng, 1, scale["td_s"])[0])
        at["t_ga"] = float(at["t_ga"]) + float(_jitter(rng, 1, scale["t_ga_s"])[0])
    truth["perturbation"] = dict(seed=int(seed), name=twin.name, rng_seed=rng_seed, **scale)
    return dataclasses.replace(twin, truth=truth, cams=cams)


def _truth_spline_and_window(twin: Twin):
    spline, _ = fit_bspline_from_states(twin.t_traj, list(twin.T_traj), twin.knot_dt)
    return spline, float(twin.t_traj[0]) + EDGE_S, float(twin.t_traj[-1]) - EDGE_S


def simulate_aux(twin: Twin, seed: int = 0, noise: bool = True) -> list:
    """The twin's aux IMUs for the solver, at their real stamps. An aux sample stamped t was
    measured at t + td (gyro) and t + td - t_ga (accel)."""
    aux_truth = twin.truth.get("aux", [])
    if not aux_truth:
        return []
    spline, t_lo, t_hi = _truth_spline_and_window(twin)
    g = np.asarray(twin.truth["g_inW"], float)
    model = parse_imu_model(twin.truth["imu_model"])
    nk = dict(twin.imu_noise) if noise else {}
    out = []
    for i, (at, t_real) in enumerate(zip(aux_truth, twin.aux_t_imu)):
        T = np.asarray(at["T_IstoI"], float)
        t = t_real[(t_real + at["td"] > t_lo) & (t_real + at["td"] < t_hi)]
        intr = ImuModel.from_tangent(np.asarray(at["xi24"], float), model)
        kw = dict(g_inG=g, R_IstoI=T[:3, :3], p_IsinI=T[:3, 3], b_g0=at["b_g"], b_a0=at["b_a"],
                  seed=seed + 200 + i, **nk)
        gyro = simulate_imu(spline, t, intr, td=at["td"], **kw)
        accel = simulate_imu(spline, t, intr, td=at["td"] - at["t_ga"], **kw)
        out.append(dict(t_imu=gyro.t, w_m=gyro.w_m, a_m=accel.a_m, name=at["name"],
                        gyro_only=at.get("gyro_only", False)))
    return out


def simulate(twin: Twin, seed: int = 0, noise: bool = True, true_t_ga: float | None = None):
    """(cams, t_imu, w_m, a_m) for the solver. `noise=False` gives exact measurements, and
    `true_t_ga` (s) replaces the truth's offset in the simulated IMU."""
    gt = twin.truth
    spline, t_lo, t_hi = _truth_spline_and_window(twin)
    Tb = np.asarray(gt["T_BinW"], float)
    pts_true = twin.board_points.copy()
    pts_true[:, 2] = np.asarray(gt["heights_m"], float)    # the true heights, not an offset from board_points
    pts_world = (Tb[:3, :3] @ pts_true.T).T + Tb[:3, 3]
    rng = np.random.default_rng(1000 + seed)
    sigmas = twin.sigma_px if isinstance(twin.sigma_px, list) else [twin.sigma_px] * len(twin.cams)
    cams = []
    for cam, cg, sigma_cam in zip(twin.cams, gt["cams"], sigmas):
        sigma = sigma_cam if noise else 0.0
        cal = cal_from_vec(cg["intr8"], cg["is_fisheye"], cg["model"])
        T_CtoI = gtsam.Pose3(np.asarray(cg["T_CtoI"], float))
        size, td, tr = tuple(cg["image_size"]), cg["td"], cg["tr"]
        t_keep, ids_out, uv_out = [], [], []
        for t, real_ids in zip(cam["t_cam"], cam["ids_list"]):
            t0 = float(t) + td
            if min(t0, t0 + tr) <= t_lo or max(t0, t0 + tr) >= t_hi:
                continue
            ids, uv, _ = project_points_rolling(spline, t0, cal, T_CtoI, pts_world[real_ids], size, tr, 0)
            if len(ids):
                t_keep.append(float(t))
                ids_out.append(real_ids[ids])
                uv_out.append(uv + rng.standard_normal(uv.shape) * sigma)
        cams.append(_cam_meta(cam) | dict(
            t_cam=np.asarray(t_keep), ids_list=ids_out, uv_list=uv_out, frames=None,
            retime_sensor_clock=False))
    model = parse_imu_model(gt["imu_model"])
    intr = ImuModel.from_tangent(np.asarray(gt["xi24"], float), model)
    g = np.asarray(gt["g_inW"], float)
    nk = dict(twin.imu_noise) if noise else {}
    t_ga = gt["t_ga"] if true_t_ga is None else float(true_t_ga)
    t_imu = twin.t_imu[(twin.t_imu > t_lo) & (twin.t_imu < t_hi)]
    gyro = simulate_imu(spline, t_imu, intr, g_inG=g, b_g0=gt["b_g"], b_a0=gt["b_a"], seed=seed + 100, **nk)
    # the accel sample stamped t was measured at t - t_ga
    accel = simulate_imu(spline, t_imu, intr, g_inG=g, td=-t_ga, b_g0=gt["b_g"], b_a0=gt["b_a"],
                         seed=seed + 100, **nk)
    return cams, gyro.t, gyro.w_m, accel.a_m


def config_for(twin: Twin, **overrides):
    """The solve config of the recording, with `overrides` on top."""
    return P.CamImuCalibConfig(**{**twin.solve_config, **overrides})


def errors(twin: Twin, result, true_t_ga: float | None = None) -> dict:
    """Estimate minus truth for one solve; per-camera lists are in camera order."""
    gt, d = twin.truth, result.diagnostics
    model = parse_imu_model(gt["imu_model"])
    intr = ImuModel.from_tangent(np.asarray(gt["xi24"], float), model)
    est = create_imu_model(model).retract(np.asarray(result.imu_intr_xi, float))
    t_ga = gt["t_ga"] if true_t_ga is None else float(true_t_ga)
    out = dict(ok=bool(result.ok), iterations=int(d["iterations"]), rms_px=float(d["reproj_rms_px"]),
               cam_chi2_dof=float(d["cam_chi2_dof"]), imu_chi2_dof=float(d["imu_chi2_dof"]),
               t_ga_ms=(float(d["t_ga"]) - t_ga) * 1e3, t_ga_sigma_ms=_ms(d.get("t_ga_sigma")),
               gyro_scale_pct=((np.diag(est.Dw()) - np.diag(intr.Dw())) * 100).tolist(),
               accel_scale_pct=((np.diag(est.Da()) - np.diag(intr.Da())) * 100).tolist(), cams=[])
    for cg, e in zip(gt["cams"], result.cameras):
        cal = e["calib"]
        T = np.asarray(cg["T_CtoI"], float)
        out["cams"].append(dict(
            lever_mm=((np.asarray(cal.T.translation(), float) - T[:3, 3]) * 1e3).tolist(),
            rotation_mrad=(log_so3(T[:3, :3].T @ cal.T.rotation().matrix()) * 1e3).tolist(),
            td_ms=(cal.td - cg["td"]) * 1e3, tr_ms=(cal.tr - cg["tr"]) * 1e3,
            focal_px=(np.asarray(cal.intr[:2], float) - np.asarray(cg["intr8"][:2], float)).tolist(),
            **_pose_sigmas(e.get("calib_sigma"))))
    out["aux"] = [_aux_errors(at, m, model) for at, m in zip(gt.get("aux", []), result.imus or [])]
    heights = d.get("board_heights")
    if heights:
        dh = np.asarray(heights["heights_m"], float) - np.asarray(gt["heights_m"], float)
        out["board_height_rms_mm"] = float(np.sqrt(np.mean(dh ** 2)) * 1e3)
    return out


def _ms(seconds):
    return None if seconds is None else float(seconds) * 1e3


def _pose_sigmas(sigma) -> dict:
    """Size of the marginal sigma of a pose block [rotation 3, translation 3]; empty if there is none."""
    if sigma is None or not np.all(np.isfinite(sigma[:6])):
        return {}
    sigma = np.asarray(sigma, float)
    return dict(rotation_sigma_mrad=float(np.linalg.norm(sigma[:3]) * 1e3),
                lever_sigma_mm=float(np.linalg.norm(sigma[3:6]) * 1e3))


def _aux_errors(at: dict, m: dict, model) -> dict:
    T = np.asarray(at["T_IstoI"], float)
    T_est = np.asarray(m["T_IstoI"].matrix(), float)
    true_i = ImuModel.from_tangent(np.asarray(at["xi24"], float), model)
    est_i = ImuModel.from_tangent(np.asarray(m["intr_xi24"], float), model)
    return dict(name=at["name"], lever_mm=((T_est[:3, 3] - T[:3, 3]) * 1e3).tolist(),
                rotation_mrad=(log_so3(T[:3, :3].T @ T_est[:3, :3]) * 1e3).tolist(),
                td_ms=(float(m["td"]) - at["td"]) * 1e3,
                t_ga_ms=(float(m.get("t_ga") or 0.0) - at["t_ga"]) * 1e3,
                t_ga_sigma_ms=_ms(m.get("t_ga_sigma")),
                gyro_scale_pct=((np.diag(est_i.Dw()) - np.diag(true_i.Dw())) * 100).tolist(),
                accel_scale_pct=((np.diag(est_i.Da()) - np.diag(true_i.Da())) * 100).tolist(),
                **_pose_sigmas(m.get("calib_sigma")))


def solve(twin: Twin, vision_pose: str | None = "imu", seed: int = 0, noise: bool = True,
          true_t_ga: float | None = None, **overrides) -> dict:
    """Simulate, run the real pipeline, report the errors against the truth. `vision_pose=None`
    keeps the recording's setting."""
    cams, t, w, a = simulate(twin, seed=seed, noise=noise, true_t_ga=true_t_ga)
    aux = simulate_aux(twin, seed=seed, noise=noise)
    if vision_pose is not None:
        overrides["vision_pose"] = vision_pose
    res = P.run_cam_imu_calibration_multi(cams, twin.board_points, t, w, a, aux_imus=aux or None,
                                          config=config_for(twin, **overrides))
    return errors(twin, res, true_t_ga)
