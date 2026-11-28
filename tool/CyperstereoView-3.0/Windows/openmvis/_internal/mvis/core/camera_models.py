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
"""Camera-model registry: the single source of truth for model names, their
distortion-coefficient counts, parameter names and defaults.

Pure python (no numpy / gtsam / cv2) so core, io and mvisdb can import it
cheaply; the per-model math lives in `mvis.camera` / `custom_projections`.
cal_vec layout everywhere is [fx, fy, cx, cy, dist...] with len(dist) = n_dist.
"""

import re
import warnings

MODELS = ("radtan", "equidistant", "fov", "double_sphere", "eucm", "ucm",
          "omni_radtan", "kb4")

N_DIST = {
    "radtan": 4, "equidistant": 4, "fov": 1, "double_sphere": 2, "eucm": 2,
    "ucm": 1, "omni_radtan": 5, "kb4": 4,
}

DIST_NAMES = {
    "radtan": ("k1", "k2", "p1", "p2"),
    "equidistant": ("k1", "k2", "k3", "k4"),
    "fov": ("omega",),
    "double_sphere": ("xi", "alpha"),
    "eucm": ("alpha", "beta"),
    "ucm": ("alpha",),                              # alpha form (Basalt); EUCM beta=1
    "omni_radtan": ("xi", "k1", "k2", "p1", "p2"),  # Kalibr omni-radtan
    "kb4": ("k1", "k2", "k3", "k4"),
}

# Models whose reports label each coefficient by its DIST_NAMES name (omni's first coefficient
# is xi); the others keep their historical unnamed layout.
NAMED_DIST_MODELS = ("omni_radtan", "kb4", "ucm")


def dist_names(model):
    """The coefficient names a report prints for ``model``, or None for an unnamed model."""
    return DIST_NAMES.get(model) if model in NAMED_DIST_MODELS else None

_DEFAULT_DIST = {
    "radtan": (0.0, 0.0, 0.0, 0.0),
    "equidistant": (0.0, 0.0, 0.0, 0.0),
    "fov": (1.0,),                                  # omega
    "double_sphere": (-0.2, 0.6),                   # Usenko 2018 / Basalt
    "eucm": (0.6, 1.0),
    "ucm": (0.6,),
    "omni_radtan": (1.0, 0.0, 0.0, 0.0, 0.0),       # xi=1 per Kalibr
    "kb4": (0.0,) * 4,
}

# Accepted input spellings of kb4. Name parsers must match these BEFORE the
# equidistant aliases ('kannala_brandt4' embeds 'kannala_brandt').
KB4_ALIASES = ("kb4", "kb_4", "kannala_brandt4", "kannala-brandt4",
               "kannala_brandt_4", "kannala-brandt-4")
# Generic fisheye names: they mean kb4, the polynomial with no 90 deg limit. Below 90 deg
# it gives the same pixels as equidistant, which only its own name (or a Kalibr
# pinhole + equidistant camchain) selects.
FISHEYE_ALIASES = ("fisheye", "kannala_brandt", "kannala-brandt")

# Spellings of the removed KB8 model (Kannala-Brandt with 8 radial terms, e.g.
# kb8, kb_8, kannala_brandt8, kannala-brandt-8). They are refused with
# `removed_model_message`, never treated as a typo: name parsers must match this
# BEFORE the equidistant aliases ('kannala_brandt8' embeds 'kannala_brandt', and
# would otherwise silently become equidistant). tests/camera/test_removed_kb8.py
# holds the spellings as its test vector.
_REMOVED_KB8 = re.compile(r"kb[-_ ]?8|kannala[-_ ]?brandt[-_ ]?8")

# Accepted input spellings of ucm. Never 'omni' / 'mei': those mean omni_radtan.
UCM_ALIASES = ("ucm", "unified", "omni_none", "omni-none")

GTSAM = frozenset({"radtan", "equidistant"})        # gtsam Cal3DS2 / Cal3Fisheye
CUSTOM = frozenset(m for m in MODELS if m not in GTSAM)   # custom_projections kernels
# theta from atan2, valid past 90 deg up to the first fold (`camera.kb_domain`)
KB_THETA_MODELS = frozenset({"kb4"})
# Every model has a Kalibr camchain form. kb4 is written as pinhole-equi with the same k1..k4: the
# same polynomial, exact for points in front of the camera; pinhole-equi divides by z, so Kalibr
# cannot project the rays at or past 90 deg (`kb4_kalibr_warning`)
KALIBR_EXPORTABLE = frozenset(MODELS)


def is_removed_model(name) -> bool:
    """True when `name` spells a model MVIS no longer has (case-insensitive, with any
    separator, also inside a longer name such as 'pinhole_kb8')."""
    return isinstance(name, str) and _REMOVED_KB8.search(name.lower()) is not None


def removed_model_message(name) -> str:
    """The error text for a removed model: what happened and what to use."""
    return (f"camera model {name!r} (Kannala-Brandt with 8 radial coefficients) "
            f"was removed from MVIS; use 'kb4' past 90 degrees, 'equidistant' below")


def check_model(model: str) -> str:
    """Return `model` if registered, else raise ValueError."""
    if is_removed_model(model):
        raise ValueError(removed_model_message(model))
    if model not in N_DIST:
        raise ValueError(f"Unknown distortion model: {model!r} "
                         f"(supported: {', '.join(MODELS)})")
    return model


def resolve_model_alias(name):
    """Canonical id for an accepted kb4 / ucm spelling, a generic fisheye name
    included (`FISHEYE_ALIASES`, kb4), case-insensitive; any
    other name is returned unchanged, so validation still sees (and rejects) typos.
    A spelling of a removed model raises ValueError with what to use instead."""
    if is_removed_model(name):
        raise ValueError(removed_model_message(name.strip()))
    if isinstance(name, str) and name.strip().lower() in KB4_ALIASES + FISHEYE_ALIASES:
        return "kb4"
    if isinstance(name, str) and name.strip().lower() in UCM_ALIASES:
        return "ucm"
    return name


def legacy_model(is_fisheye, cal_vec=None) -> str:
    """Model of a spec that gives only the legacy `is_fisheye` flag:
    equidistant / radtan. Both are 8-long, so a cal_vec of any other length is
    a custom model that must be named explicitly — raise, never guess radtan.
    kb4 is 8-long too: length cannot catch an unnamed one, so every path carrying
    a kb4 cal_vec must also carry its name (tests/camera/test_kb4.py)."""
    if cal_vec is not None and len(cal_vec) != 8:
        raise ValueError(f"{len(cal_vec)}-long cal_vec with no distortion model; "
                         f"only radtan/equidistant (8) may omit it")
    return "equidistant" if is_fisheye else "radtan"


def n_dist(model: str) -> int:
    """Number of distortion coefficients of `model` (n_intr = 4 + n_dist)."""
    return N_DIST[check_model(model)]


def is_custom(model: str) -> bool:
    """True for models projected by a custom_projections kernel (not gtsam)."""
    return check_model(model) in CUSTOM


def default_dist(model: str) -> list:
    """Default distortion coefficients of `model` (length n_dist)."""
    return list(_DEFAULT_DIST[check_model(model)])


def canonical_dist(model: str, coeffs) -> list:
    """Coefficients of exactly n_dist(model): short lists are zero-padded
    (legacy files), longer ones trimmed only when the extras are all zero
    (e.g. ds/eucm/fov zero-padded to 4); otherwise raise."""
    n = n_dist(model)
    c = [float(x) for x in (coeffs if coeffs is not None else [])]
    if len(c) > n:
        if any(x != 0.0 for x in c[n:]):
            raise ValueError(f"{model}: expected {n} distortion coefficients, "
                             f"got {len(c)} with non-zero extras {c[n:]}")
        return c[:n]
    return c + [0.0] * (n - len(c))


def fit_dist(model: str, coeffs) -> list:
    """`canonical_dist`, except a radtan list longer than 4 (OpenCV's
    k1,k2,p1,p2,k3[,...] tail) keeps its legacy cut to 4 with a warning instead
    of raising — the radtan model has no k3. Every other model is strict."""
    c = list(coeffs) if coeffs is not None else []
    if check_model(model) == "radtan" and len(c) > 4:
        if any(float(x) != 0.0 for x in c[4:]):
            warnings.warn(f"radtan: dropping non-zero extra coefficients {c[4:]} "
                          f"(only k1, k2, p1, p2 are modelled)", stacklevel=2)
        c = c[:4]
    return canonical_dist(model, c)


def canonical_cal_vec(model: str, cal_vec) -> list:
    """[fx, fy, cx, cy] + canonical_dist(model, cal_vec[4:])."""
    v = [float(x) for x in cal_vec]
    if len(v) < 4:
        raise ValueError(f"cal_vec needs at least fx, fy, cx, cy; got {len(v)}")
    return v[:4] + canonical_dist(model, v[4:])


# Distortion sanity rule shared by the cam-only pass-2 guardrail
# (`pipeline_cam._pass2_is_pathological`) and the regression health check
# (`mvis.regression.cam_stats`), so the two stay aligned.
OMNI_XI_RANGE = (0.0, 5.0)   # omni_radtan: accepted xi (wide lenses fit xi > 2)
UCM_ALPHA_RANGE = (0.0, 1.0)  # ucm: alpha in [0, 1); 1 is xi = inf (no Kalibr camchain)


def dist_sanity_issue(model, coeffs, max_abs: float = 2.0):
    """First pathology of a distortion vector, or None.

    Returns (i, reason): `reason` is None for the plain |coeffs[i]| > max_abs
    bound (callers keep their historical message), else a description.
    Rules, per model:
      * default (radtan / equidistant / fov / ds / eucm / unknown): every
        |c| <= max_abs.
      * ucm: alpha in UCM_ALPHA_RANGE, [0, 1) (alpha = 1 is xi = infinity; a
        negative alpha is not a unified-model fit).
      * omni_radtan: xi in OMNI_XI_RANGE ([0, 5]: very wide lenses fit xi
        above 2, and a negative xi is not a valid unified-model fit); the
        radtan part k1, k2, p1, p2 keeps |c| <= max_abs.
    """
    if is_removed_model(model):
        raise ValueError(removed_model_message(model))
    c = [float(x) for x in coeffs]
    for i, v in enumerate(c):
        if model == "ucm" and i == 0:
            lo, hi = UCM_ALPHA_RANGE
            if not lo <= v < hi:       # also catches nan
                return 0, f"ucm alpha={v:.3f} outside [{lo:g}, {hi:g})"
        elif model == "omni_radtan" and i == 0:
            lo, hi = OMNI_XI_RANGE
            if not lo <= v <= hi:      # also catches nan
                return 0, f"omni xi={v:.3f} outside [{lo:g}, {hi:g}]"
        elif abs(v) > max_abs:
            return i, None
    return None
