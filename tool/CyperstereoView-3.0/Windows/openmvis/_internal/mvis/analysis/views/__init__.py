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
"""Canvas views for the reports and the sensor pages: the calibration process and the solved rig.

`spec` builds the data, `html` the markup, `assets/views.js` draws it. No Plotly or WebGL, so the views
print and load fast. One copy serves the MVIS reports and openmvis.com (`<report>_views.json`, no frames).
"""
from mvis.analysis.views.html import assets_html, rig_html, scene_html
from mvis.analysis.views.spec import VERSION, rig_spec, scene_spec

__all__ = ["VERSION", "assets_html", "rig_html", "rig_spec", "scene_html", "scene_spec"]
