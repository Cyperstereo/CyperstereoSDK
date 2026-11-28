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
"""Camera models wrapping GTSAM calibration types and custom projections."""

from mvis.camera.camera_model import CameraModel, create_camera_model
from mvis.camera.pinhole_radtan import PinholeRadtan
from mvis.camera.pinhole_equidistant import PinholeEquidistant
from mvis.camera.fov_model import FOVModel
from mvis.camera.double_sphere import DoubleSphere
from mvis.camera.eucm import EUCM
from mvis.camera.ucm import UCM
from mvis.camera.omni_radtan import OmniRadtan
from mvis.camera.kb4 import KannalaBrandt4
