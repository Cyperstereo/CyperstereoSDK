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
"""The gravity magnitude a camera + IMU solve holds fixed.

Gravity dominates what an accelerometer measures, so the accelerometer scale a solve
estimates absorbs g_true / g_used: holding 9.81 where g is 9.788 (Shenzhen) reads the scale
0.23 % high. A config gives the local value as `gravity:` (m/s^2) or as
`site: {latitude_deg, altitude_m}` (WGS84 normal gravity). Neither: MVIS's historical 9.81,
so existing configs give unchanged results. See doc/local_gravity.md.
"""
from __future__ import annotations

import math

STANDARD_MVIS = 9.81          # what every solve used before `gravity` / `site` existed
G_RANGE = (9.70, 9.90)        # anything outside is a unit or typing mistake
AGREE_TOL = 0.01              # m/s^2: `gravity` and `site` both given must agree this well
DEFAULT_HINT = ("no `gravity:` or `site:` in the config, so |g| = 9.81 m/s^2; the accel scale absorbs "
                "g_local / 9.81 (about -0.23 % to +0.3 % by latitude and altitude). Set the local value, e.g. "
                "`gravity: 9.8013   # m/s², Newark, Delaware`, or `site: {latitude_deg, altitude_m}` "
                "(doc/local_gravity.md lists common places)")


def normal_gravity(latitude_deg: float, altitude_m: float = 0.0) -> float:
    """WGS84 normal gravity (Somigliana) with the free-air correction, m/s^2. Within about
    0.05 % of a measured value; altitude lowers it 0.3 mm/s^2 per 100 m."""
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError(f"site.latitude_deg {latitude_deg} must be in [-90, 90]")
    s2 = math.sin(math.radians(latitude_deg)) ** 2
    g0 = 9.7803253359 * (1 + 0.00193185265241 * s2) / math.sqrt(1 - 0.00669437999013 * s2)
    return g0 - 3.086e-6 * float(altitude_m)


def of_result(result) -> float:
    """|g| a solve held fixed, from its diagnostics; 9.81 for results without one (made before
    `gravity` existed, or stand-ins in tests)."""
    return float((getattr(result, "diagnostics", None) or {}).get("gravity", STANDARD_MVIS))


def resolve(gravity: float | None = None, site: dict | None = None) -> tuple[float, str]:
    """(g, source) from a config's `gravity` and `site`; source is "config", "site" or "default"."""
    g_site = None
    if site:
        unknown = set(site) - {"latitude_deg", "longitude_deg", "altitude_m", "name"}
        if unknown or "latitude_deg" not in site:
            raise ValueError("site needs latitude_deg (and optionally longitude_deg, altitude_m, name); "
                             f"got {sorted(site)}")
        if not -180.0 <= float(site.get("longitude_deg", 0.0)) <= 180.0:   # display only: g ignores it
            raise ValueError(f"site.longitude_deg {site['longitude_deg']} must be in [-180, 180]")
        g_site = normal_gravity(float(site["latitude_deg"]), float(site.get("altitude_m", 0.0)))
    if gravity is not None:
        g = float(gravity)
        if not G_RANGE[0] < g < G_RANGE[1]:
            raise ValueError(f"gravity {g} m/s^2 is outside {G_RANGE}: give it in m/s^2")
        if g_site is not None and abs(g - g_site) > AGREE_TOL:
            raise ValueError(f"gravity {g} and site ({g_site:.4f} m/s^2) disagree by more than {AGREE_TOL}")
        return g, "config"
    if g_site is not None:
        return g_site, "site"
    return STANDARD_MVIS, "default"


def site_record(site: dict | None) -> dict | None:
    """The config `site` as a solve keeps it for the report: the given keys plus g_site, its normal gravity."""
    if not site:
        return None
    return {**site, "g_site": normal_gravity(float(site["latitude_deg"]), float(site.get("altitude_m", 0.0)))}


def _place(site: dict) -> str:
    lat, lon = float(site["latitude_deg"]), site.get("longitude_deg")
    parts = [f"{abs(lat):.2f}° {'N' if lat >= 0 else 'S'}"]
    if lon is not None:
        parts.append(f"{abs(float(lon)):.2f}° {'E' if float(lon) >= 0 else 'W'}")
    if site.get("altitude_m") is not None:
        parts.append(f"{float(site['altitude_m']):.0f} m")
    return f"{site.get('name') or 'site'} ({', '.join(parts)})"


def describe(g: float, source: str, site: dict | None = None, place: str | None = None) -> str:
    """One line on the |g| a solve held and where it came from, for the reports. `place`, from the
    comment after the config's `gravity:` value, names a config value that has no site."""
    vs = f"vs {STANDARD_MVIS}: {(g / STANDARD_MVIS - 1) * 100:+.2f} %"
    if source == "site":           # results made before gravity_site existed have no site record
        return f"{g:.4f} m/s² — WGS84 normal gravity at {_place(site) if site else 'the config site'}; {vs}"
    if source == "config" and site:
        gs = site["g_site"]
        return (f"{g:.4f} m/s² — set in config; {_place(site)} gives {gs:.4f} "
                f"({(gs / STANDARD_MVIS - 1) * 100:+.2f} % vs {STANDARD_MVIS})")
    if source == "config":
        return f"{g:.4f} m/s² — set in config{f' for {place}' if place else ''}; {vs}"
    return f"{g:.4f} m/s² — MVIS default, no site given"
