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
"""Map raw-bag topics to calibrated sensors + their base-clock time offsets."""
from __future__ import annotations
from dataclasses import dataclass
from mvis.recalib.calib_spec import RigCalib, ImuCalib, MVIS

@dataclass
class TopicPlan:
    kind: str               # "imu" | "cam" | "passthrough"
    imu: ImuCalib | None = None
    cam: MVIS | None = None
    td: float = 0.0

def derive_mapping(rig: RigCalib) -> tuple[str, dict]:
    """Build (base_imu_topic, mapping{topic: sensor_name}) from the calib's stored
    per-sensor topics. Raises ValueError naming the offending sensor(s) if any
    sensor lacks a `topic` (old calib) — the caller should then pass an explicit map."""
    all_sensors = [rig.base_imu, *rig.aux_imus, *rig.cams]
    missing = [s.name for s in all_sensors if s.topic is None]
    if missing:
        names = ", ".join(repr(n) for n in missing)
        raise ValueError(
            f"Sensor(s) {names} have no stored topic; "
            "re-export the calib with the updated writer or pass an explicit --map."
        )
    mapping: dict[str, str] = {}
    for sensor in all_sensors:
        if sensor.topic in mapping:
            raise ValueError(
                f"Topic {sensor.topic!r} is shared by sensors "
                f"{mapping[sensor.topic]!r} and {sensor.name!r}."
            )
        mapping[sensor.topic] = sensor.name
    base_imu_topic = rig.base_imu.topic
    return base_imu_topic, mapping


class MappingDeriveError(ValueError):
    """`resolve_mapping` fell through to `derive_mapping` and that failed (same message). A
    ValueError subclass, so code catching ValueError still catches it."""


def resolve_mapping(rig: RigCalib, map_path: str | None = None,
                    config: dict | None = None) -> tuple[str, dict]:
    """(base_imu_topic, mapping{topic: sensor_name}) for applying `rig` to a bag, first match wins:
    a topic-map YAML at `map_path` (keys `base_imu_topic`, `mapping`), then a recalib `config`
    dict that sets both keys inline, then `derive_mapping(rig)`. Only a `derive_mapping` failure
    is raised as `MappingDeriveError`, so each caller words its own 'cannot auto-derive' exit
    message for that case alone; a bad --map file raises as it did before (yaml, KeyError, ...)."""
    if map_path:
        import yaml
        with open(map_path) as f:
            m = yaml.safe_load(f)
        return m["base_imu_topic"], m["mapping"]
    if config and "mapping" in config and "base_imu_topic" in config:
        return config["base_imu_topic"], config["mapping"]
    try:
        return derive_mapping(rig)
    except ValueError as e:
        raise MappingDeriveError(str(e)) from e


def build_topic_plan(rig: RigCalib, mapping: dict, base_imu_topic: str) -> dict:
    """Resolve `mapping` (topic -> sensor name) against the rig into a per-topic correction plan (the calib + td each topic uses), validating that base_imu_topic maps to the base IMU."""
    by_name_imu = {i.name: i for i in [rig.base_imu, *rig.aux_imus]}
    by_name_cam = {c.name: c for c in rig.cams}
    if base_imu_topic not in mapping or mapping[base_imu_topic] != rig.base_imu.name:
        raise ValueError(
            f"base_imu_topic {base_imu_topic!r} must map to the base IMU "
            f"{rig.base_imu.name!r}; got {mapping.get(base_imu_topic)!r}")
    plan: dict[str, TopicPlan] = {}
    for topic, name in mapping.items():
        if name in by_name_imu:
            imu = by_name_imu[name]
            plan[topic] = TopicPlan("imu", imu=imu, td=imu.td)
        elif name in by_name_cam:
            cam = by_name_cam[name]
            plan[topic] = TopicPlan("cam", cam=cam, td=cam.td)
        else:
            raise ValueError(f"mapping target {name!r} not found in calib")
    return plan
