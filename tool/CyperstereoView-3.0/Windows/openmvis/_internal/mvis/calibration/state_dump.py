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
"""Full-precision state dump of a joint cam-IMU solve.

`cam_imu.json` rounds every number to 3-5 decimals, so it cannot tell a
bitwise-identical solve from one that moved by a few ulps. The speed-up steps
are gated on the npz this module writes instead: every key of the final
`gtsam.Values` (named by its gtsam symbol, Pose3 as a 4x4 matrix), the
retracted camera calibs, the marginal sigmas, the cost before/after, the
per-frame / per-corner reprojection norms and the stage timings.

Debug environment variables, read by `_run_cam_imu_calibration_multi_once`:

* ``MVIS_DUMP_STATE=<path.npz>`` -- write the dump after the solve. When the
  pipeline runs more than once in a process (sensor-clock two-pass, batch
  runners) the second and later dumps go to ``<stem>_pass2.npz`` etc.
* ``MVIS_PERTURB_INIT=<rel>`` -- perturb the initial Values by
  ``rel * (|x| + 1) * u``, u ~ U(-1, 1), before the solve (noise-floor
  measurement; ``MVIS_PERTURB_SEED`` picks the RNG seed, default 0).

`compare_states` holds the comparison rules (bitwise, or the per-quantity
tolerances of the plan) so that the maintainer comparison script and the
tests share one implementation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import gtsam

from mvis.core.rotation import rotation_angle

logger = logging.getLogger(__name__)

_PASS_COUNTER = {"n": 0}

# npz keys that are reported but never gated
_UNGATED_PREFIXES = ("timings/", "meta/")


# ---------------------------------------------------------------------------
# Values <-> arrays
# ---------------------------------------------------------------------------
def symbol_name(key: int) -> str:
    """'x12', 'c0', 'H0', ... for a gtsam symbol key."""
    return gtsam.Symbol(int(key)).string()


def values_to_arrays(values: gtsam.Values) -> dict:
    """Every key of `values` as ``values/<symbol>``: Pose3 -> (4, 4) matrix,
    vectors as float arrays. Unsupported types are listed under
    ``meta/unsupported_keys``."""
    out, unsupported = {}, []
    for k in values.keys():
        name = symbol_name(k)
        try:
            out[f"values/{name}"] = np.asarray(values.atPose3(k).matrix(), float)
            continue
        except RuntimeError:
            pass
        try:
            out[f"values/{name}"] = np.asarray(values.atVector(k), float).reshape(-1)
            continue
        except RuntimeError:
            pass
        try:
            out[f"values/{name}"] = np.asarray(values.atRot3(k).matrix(), float)
            continue
        except RuntimeError:
            unsupported.append(name)
    if unsupported:
        out["meta/unsupported_keys"] = np.array(unsupported)
    return out


def perturb_values(values: gtsam.Values, rel: float, seed: int = 0) -> int:
    """Perturb every key of `values` IN PLACE by ``rel * (|x| + 1) * u`` with
    u ~ U(-1, 1) (Pose3: retract of such a 6-vector, translation scaled by its
    own magnitude, rotation by 1). Returns the number of keys touched."""
    rng = np.random.default_rng(int(seed))
    n = 0
    for k in values.keys():
        try:
            T = values.atPose3(k)
        except RuntimeError:
            T = None
        if T is not None:
            t = np.asarray(T.translation(), float)
            scale = np.concatenate([np.ones(3), np.abs(t) + 1.0])
            d = rel * scale * rng.uniform(-1.0, 1.0, 6)
            values.update(k, T.retract(d))
            n += 1
            continue
        try:
            x = np.asarray(values.atVector(k), float).reshape(-1)
        except RuntimeError:
            continue
        d = rel * (np.abs(x) + 1.0) * rng.uniform(-1.0, 1.0, x.shape)
        values.update(k, x + d)
        n += 1
    return n


def maybe_perturb_initial_values(values: gtsam.Values) -> float | None:
    """Apply ``MVIS_PERTURB_INIT`` (if set) to `values` in place; returns the
    relative magnitude used, or None."""
    rel = os.environ.get("MVIS_PERTURB_INIT")
    if not rel:
        return None
    rel = float(rel)
    seed = int(os.environ.get("MVIS_PERTURB_SEED", "0"))
    n = perturb_values(values, rel, seed)
    logger.warning("MVIS_PERTURB_INIT: initial Values perturbed by %.1e relative "
                   "(seed %d, %d keys) -- noise-floor run, not a production result",
                   rel, seed, n)
    return rel


# ---------------------------------------------------------------------------
# dump
# ---------------------------------------------------------------------------
def _flatten(prefix: str, obj, out: dict) -> None:
    """Nested dicts of numbers / arrays -> flat ``prefix/key`` entries."""
    if obj is None:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}/{k}", v, out)
        return
    if isinstance(obj, gtsam.Pose3):
        out[prefix] = np.asarray(obj.matrix(), float)
        return
    arr = np.asarray(obj)
    if arr.dtype.kind in "biufc":
        out[prefix] = arr.astype(float) if arr.dtype.kind in "fc" else arr
    elif arr.dtype.kind in "US":
        out[prefix] = arr


def build_state_dump(values: gtsam.Values, *, calibs=None, sigmas=None,
                     scalars=None, arrays=None, timings=None) -> dict:
    """Assemble the flat npz dict (see module docstring for the layout).

    `calibs` : {name: CameraCalib} -> ``calib/<name>/{T,td,tr,intr}``
    `sigmas` : {symbol: vector | None} -> ``sigma/<symbol>`` (None skipped)
    `scalars`: {name: float} -> top-level ``<name>``
    `arrays` : {name: array} -> top-level ``<name>``
    `timings`: {name: float} -> ``timings/<name>`` (never gated)"""
    out = values_to_arrays(values)
    for name, cal in (calibs or {}).items():
        out[f"calib/{name}/T"] = np.asarray(cal.T.matrix(), float)
        out[f"calib/{name}/td"] = np.array(float(cal.td))
        out[f"calib/{name}/tr"] = np.array(float(cal.tr))
        out[f"calib/{name}/intr"] = np.asarray(cal.intr, float).reshape(-1)
    for name, s in (sigmas or {}).items():
        if s is not None:
            out[f"sigma/{name}"] = np.asarray(s, float).reshape(-1)
    for name, v in (scalars or {}).items():
        if v is not None:
            out[name] = np.array(v)
    for name, v in (arrays or {}).items():
        _flatten(name, v, out)
    for name, v in (timings or {}).items():
        out[f"timings/{name}"] = np.array(float(v))
    return out


def next_dump_path(path: str) -> str:
    """First call in a process -> `path`; later calls -> ``<stem>_pass<n>.npz``."""
    n = _PASS_COUNTER["n"]
    _PASS_COUNTER["n"] = n + 1
    if n == 0:
        return path
    stem, ext = os.path.splitext(path)
    return f"{stem}_pass{n + 1}{ext or '.npz'}"


def write_state_dump(path: str, dump: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez(path, **dump)


def maybe_dump_state(values: gtsam.Values, **kw) -> str | None:
    """Write the dump when ``MVIS_DUMP_STATE`` is set; returns the path."""
    path = os.environ.get("MVIS_DUMP_STATE")
    if not path:
        return None
    path = next_dump_path(path)
    write_state_dump(path, build_state_dump(values, **kw))
    logger.info("full-precision state dumped to %s", path)
    return path


def load_state_dump(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
@dataclass
class Tolerances:
    """Per-quantity tolerances of the plan (cam-imu-speedup, "ROUNDING steps").
    `scale` multiplies every absolute / relative tolerance (noise-floor
    calibration); `sigma_mult` is the sigma-normalised rule
    |d| <= sigma_mult * sigma where a marginal sigma exists."""
    rot: float = 1e-8          # rad, |log(R_a^T R_b)|
    trans: float = 1e-8        # m
    time: float = 1e-9         # s (td, tr, t_ga, per-frame dt)
    intr_rel: float = 1e-9     # camera / IMU intrinsics, relative
    state_abs: float = 1e-8    # vb / aux state vectors (no plan rule), absolute
    heights: float = 1e-8      # board heights, m
    reproj_px: float = 1e-6    # rms / per-corner norms
    cost_rel: float = 1e-9     # err_init / err_final / imu interval costs
    sigma_rel: float = 1e-6    # marginal sigmas, relative
    sigma_mult: float = 1e-4   # sigma-normalised rule
    scale: float = 1.0

    def with_overrides(self, overrides: dict) -> "Tolerances":
        d = dict(self.__dict__)
        for k, v in overrides.items():
            if k not in d:
                raise KeyError(f"unknown tolerance group {k!r}; "
                               f"known: {sorted(d)}")
            d[k] = float(v)
        return Tolerances(**d)


@dataclass
class BlockResult:
    key: str
    n: int
    max_abs: float          # max |a - b| (rotations: geodesic angle)
    max_ratio: float        # max |a - b| / tol   (nan when not gated)
    max_sigma: float        # max |a - b| / sigma (nan when no sigma)
    tol: float              # the (smallest) tolerance applied to the block
    ok: bool
    note: str = ""


def _sym_of(key: str) -> str | None:
    """'values/x12' -> 'x12'; None for other keys."""
    return key.split("/", 1)[1] if key.startswith("values/") else None


def _gate_block(key: str, a: np.ndarray, b: np.ndarray, tol: Tolerances,
                sig: np.ndarray | None, calib_intr: np.ndarray | None) -> BlockResult:
    """Tolerance rule for one npz entry (tolerance mode)."""
    s = tol.scale
    n = int(a.size)
    sym = _sym_of(key)

    # ---- Pose3 matrices (states, board poses, calib T) ------------------
    if a.shape == (4, 4) and (sym is not None and sym[0] in "xb" or key.endswith("/T")):
        ang = rotation_angle(a[:3, :3].T @ b[:3, :3])
        dt = float(np.max(np.abs(a[:3, 3] - b[:3, 3])))
        r = max(ang / (tol.rot * s), dt / (tol.trans * s))
        return BlockResult(key, 2, max(ang, dt), r, np.nan, min(tol.rot, tol.trans) * s,
                           r <= 1.0, f"rot {ang:.3e} rad, trans {dt:.3e} m")

    if a.dtype.kind in "iub" or key in ("iterations", "fac_cam", "corner_cam"):
        eq = bool(np.array_equal(a, b))
        d = np.abs(a.astype(float) - b.astype(float))
        return BlockResult(key, n, float(d.max()) if n else 0.0, 0.0 if eq else np.inf,
                           np.nan, 0.0, eq, "exact")

    d = np.abs(a - b)
    # NaN patterns must agree; NaN-vs-NaN counts as equal
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if np.any(nan_a != nan_b):
        return BlockResult(key, n, np.inf, np.inf, np.nan, 0.0, False, "NaN pattern differs")
    d = np.where(nan_a, 0.0, d)

    # ---- per-entry absolute tolerance vector -------------------------------
    if sym is not None and sym[0] == "c":                   # camera calib xi
        t = np.full(a.shape, tol.state_abs)
        t[0:3] = tol.rot
        t[3:6] = tol.trans
        t[6:8] = tol.time
        if calib_intr is not None and a.size - 8 == calib_intr.size:
            t[8:] = tol.intr_rel * np.maximum(np.abs(calib_intr), 1.0)
        else:
            t[8:] = tol.intr_rel * np.maximum(np.abs(a[8:]), 1.0)
    elif sym is not None and sym[0] == "A":                 # aux IMU xi31
        t = np.full(a.shape, tol.state_abs)
        t[0:3] = tol.rot
        t[3:6] = tol.trans
        t[6] = tol.time
        t[7:] = tol.intr_rel * np.maximum(np.abs(a[7:]), 1.0)
    elif sym is not None and sym[0] == "I":                 # IMU intrinsics xi24
        t = tol.intr_rel * np.maximum(np.abs(a), 1.0)
    elif sym is not None and sym[0] in "of":                # t_ga, per-frame dt
        t = np.full(a.shape, tol.time)
    elif sym is not None and sym[0] in "Hl":                # heights, landmarks
        t = np.full(a.shape, tol.heights)
    elif sym is not None:                                   # v, u, g, ...
        t = np.full(a.shape, tol.state_abs)
    elif key.endswith("/td") or key.endswith("/tr"):
        t = np.full(a.shape, tol.time)
    elif key.endswith("/intr"):
        t = tol.intr_rel * np.maximum(np.abs(a), 1.0)
    elif key.startswith("sigma/"):
        t = tol.sigma_rel * np.maximum(np.abs(a), 1e-300)
    elif key in ("rms_pre", "rms_post", "corner_px"):
        t = np.full(a.shape, tol.reproj_px)
    elif key in ("err_init", "err_final") or key.startswith("imu_interval_cost") \
            or key.startswith("imu_cost_split"):
        t = tol.cost_rel * np.maximum(np.abs(a), 1e-300)
    else:
        t = np.full(a.shape, tol.state_abs)
    t = t * s
    # sigma-normalised rule where a marginal sigma exists
    sig_ratio = np.nan
    if sig is not None and sig.shape == a.shape and np.all(np.isfinite(sig)):
        t = np.maximum(t, tol.sigma_mult * sig)
        with np.errstate(divide="ignore", invalid="ignore"):
            sig_ratio = float(np.max(np.where(sig > 0, d / sig, 0.0))) if n else 0.0
    ratio = float(np.max(d / t)) if n else 0.0
    return BlockResult(key, n, float(d.max()) if n else 0.0, ratio, sig_ratio,
                       float(t.min()) if n else 0.0, ratio <= 1.0)


def compare_states(A: dict, B: dict, *, exact: bool = False,
                   tol: Tolerances | None = None) -> tuple[bool, list[BlockResult]]:
    """Compare two dumps. `exact`: every gated entry bitwise (NaN == NaN).
    Otherwise the plan's per-quantity tolerances. Timings / meta are reported
    with max |d| but never gated. Missing keys fail."""
    tol = tol or Tolerances()
    rows: list[BlockResult] = []
    ok = True
    keys = sorted(set(A) | set(B))
    for key in keys:
        if key not in A or key not in B:
            rows.append(BlockResult(key, 0, np.inf, np.inf, np.nan, 0.0, False,
                                    "missing in " + ("A" if key not in A else "B")))
            ok = False
            continue
        a, b = A[key], B[key]
        if a.dtype.kind in "US" or b.dtype.kind in "US":
            eq = bool(np.array_equal(a, b))
            rows.append(BlockResult(key, int(a.size), 0.0 if eq else np.inf,
                                    0.0 if eq else np.inf, np.nan, 0.0, eq, "string"))
            ok &= eq
            continue
        a = np.asarray(a, float) if a.dtype.kind == "f" else a
        b = np.asarray(b, float) if b.dtype.kind == "f" else b
        if a.shape != b.shape:
            rows.append(BlockResult(key, int(a.size), np.inf, np.inf, np.nan, 0.0, False,
                                    f"shape {a.shape} vs {b.shape}"))
            ok = False
            continue
        if key.startswith(_UNGATED_PREFIXES):
            mx = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
            rows.append(BlockResult(key, int(a.size), mx, np.nan, np.nan, np.nan, True,
                                    "not gated"))
            continue
        if exact:
            eq = bool(np.array_equal(a, b, equal_nan=True))
            with np.errstate(invalid="ignore"):
                dd = np.abs(a.astype(float) - b.astype(float))
            mx = float(np.nanmax(dd)) if a.size and not np.all(np.isnan(dd)) else 0.0
            rows.append(BlockResult(key, int(a.size), mx, 0.0 if eq else np.inf,
                                    np.nan, 0.0, eq, "bitwise"))
            ok &= eq
            continue
        sym = _sym_of(key)
        sig = A.get(f"sigma/{sym}") if sym is not None else None
        calib_intr = None
        if sym is not None and sym[0] == "c":
            calib_intr = A.get(f"calib/{sym}/intr")
        r = _gate_block(key, a, b, tol, sig, calib_intr)
        rows.append(r)
        ok &= r.ok
    return ok, rows


def group_of(key: str) -> str:
    """Report group: 'values/x', 'values/c', 'calib/T', 'sigma', 'rms_post', ..."""
    if key.startswith("values/"):
        return "values/" + key[7]
    if key.startswith("calib/"):
        return "calib/" + key.rsplit("/", 1)[1]
    if key.startswith("sigma/"):
        return "sigma/" + key[6]
    return key.split("/", 1)[0]


def summarize(rows: list[BlockResult]) -> list[dict]:
    """Per-group max over the block rows (for the report table)."""
    groups: dict[str, dict] = {}
    for r in rows:
        g = groups.setdefault(group_of(r.key), dict(
            group=group_of(r.key), blocks=0, n=0, max_abs=0.0, max_ratio=0.0,
            max_sigma=np.nan, tol=np.inf, ok=True, worst=""))
        g["blocks"] += 1
        g["n"] += r.n
        if r.max_abs > g["max_abs"] or (np.isinf(r.max_abs) and not np.isinf(g["max_abs"])):
            g["max_abs"] = r.max_abs
            g["worst"] = r.key
        if not np.isnan(r.max_ratio):
            g["max_ratio"] = max(g["max_ratio"], r.max_ratio)
        if not np.isnan(r.max_sigma):
            g["max_sigma"] = (r.max_sigma if np.isnan(g["max_sigma"])
                              else max(g["max_sigma"], r.max_sigma))
        if not np.isnan(r.tol):
            g["tol"] = min(g["tol"], r.tol)
        g["ok"] &= r.ok
    return sorted(groups.values(), key=lambda g: g["group"])


def format_summary(groups: list[dict]) -> str:
    lines = [f"{'group':<14} {'blocks':>6} {'n':>7} {'max|d|':>10} {'d/tol':>9} "
             f"{'d/sigma':>9} {'tol':>9}  ok  worst block"]
    for g in groups:
        lines.append(
            f"{g['group']:<14} {g['blocks']:>6} {g['n']:>7} {g['max_abs']:>10.3e} "
            f"{g['max_ratio']:>9.3g} {g['max_sigma']:>9.3g} {g['tol']:>9.2e}  "
            f"{'ok ' if g['ok'] else 'FAIL'} {g['worst']}")
    return "\n".join(lines)
