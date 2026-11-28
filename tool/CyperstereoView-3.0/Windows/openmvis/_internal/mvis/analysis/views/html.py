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
"""HTML for the canvas views: one container per view, spec inline as JSON.

The CSS and JS (`assets/`) go into a page once (`assets_html`); each view is a
`<div class="mvv" data-mvv=...>` holding its spec, booted by `views.js`.
"""
from __future__ import annotations

import json
from importlib import resources

SCENE_LEAD = ("The solved path of the rig around the board. Press Play or drag the bar to follow it, use the buttons "
              "to change the view, and drag to rotate. Click a camera on the right to put it on the frustum.")
RIG_LEAD = ("The solved rig to scale, in its own frame: forward along the cameras, right along the baseline. "
            "Tap a sensor for its values. Field-of-view rims are traced through each lens model from the image edges.")


def asset(name: str) -> str:
    """Text of a file in mvis/analysis/views/assets (views.js, views.css)."""
    return resources.files("mvis.analysis.views").joinpath("assets", name).read_text(encoding="utf-8")


def assets_html() -> str:
    """The views' CSS and JS, inline. Put once per page, before or after the views."""
    js = asset("views.js").replace("</script", "<\\/script")   # an inline script ends at the first </script
    return f"<style>{asset('views.css')}</style><script>{js}</script>"


def _json(spec: dict) -> str:
    # `</` can't appear inside a <script> element
    return json.dumps(spec, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")


def scene_html(spec: dict, mode: str = "report", lead: str | None = SCENE_LEAD) -> str:
    """The calibration process view. mode: 'report' (metre axes on) or 'site'."""
    p = f'<p class="mv-lead">{lead}</p>' if lead else ""
    return (f'<div class="mvv" data-mvv="scene" data-mode="{mode}">{p}'
            f'<script type="application/json">{_json(spec)}</script></div>')


def rig_html(spec: dict, lead: str | None = RIG_LEAD) -> str:
    """The solved rig view: 3D map, to-scale drawings from above and the front, rig table."""
    p = f'<p class="mv-lead">{lead}</p>' if lead else ""
    return f'<div class="mvv" data-mvv="rig">{p}<script type="application/json">{_json(spec)}</script></div>'
