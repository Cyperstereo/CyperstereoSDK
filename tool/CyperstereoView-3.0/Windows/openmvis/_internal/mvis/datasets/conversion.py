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
"""Run a dataset's conversion chain: produce ROS1 bags (+ optional mcap twins)
with provenance sidecars and a rosbags topic check.

Output layout: ``<catalog_root()>/<dataset dir>/bags/<sequence>.bag`` plus
``<sequence>.provenance.json``. Already-rosbag datasets need no conversion —
only the optional mcap twin applies (written next to the source bag).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .catalog import CalibrationDataset, DataSource
from .converters import (CONVERTER_REGISTRY, convert_euroc_folder_to_rosbag,
                         convert_image_folder_to_rosbag,
                         convert_penncosyvio_sequence_to_rosbag,
                         convert_rosbag_to_mcap,
                         convert_tumvie_vidata_to_euroc_folder,
                         extract_archive)


def _topic_summary(bag_path: Path) -> dict:
    from rosbags.rosbag1 import Reader
    with Reader(bag_path) as reader:
        return {c.topic: c.msgcount for c in reader.connections}


def _write_provenance(bag_path: Path, dataset: CalibrationDataset,
                      source: DataSource, extra: dict) -> None:
    sidecar = bag_path.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps({
        "dataset": dataset.key,
        "sequence": source.sequence,
        "source_url": source.url,
        "source_sha256": source.sha256,
        "license_bucket": dataset.license_bucket,
        "converted": str(date.today()),
        "conversion_chain": [s.converter for s in dataset.conversion_chain],
        "topics": _topic_summary(bag_path),
        **extra,
    }, indent=1) + "\n")


def convert_sequence_to_rosbag(dataset: CalibrationDataset,
                               source: DataSource, *,
                               also_mcap: bool = False) -> Path | None:
    """Convert one sequence per its dataset's native format.

    Returns the bag path (the source itself for already-rosbag datasets),
    or None when the sequence's data is not local.
    """
    local = source.local_path
    if not local.exists():
        return None
    bags_dir = local.parent / "bags"

    if dataset.native_format == "rosbag":
        bag_path = local                      # nothing to convert
    elif dataset.native_format == "zip_archive":
        extraction = extract_archive(local)
        sequence_dir = extraction / local.stem
        bag_path = bags_dir / f"{source.sequence}.bag"
        stats = convert_euroc_folder_to_rosbag(sequence_dir, bag_path)
        _write_provenance(bag_path, dataset, source,
                          {"timestamps": "original", **stats})
    elif dataset.native_format == "tumvie_vidata":
        extraction_root = local.parent.parent   # tars live in <root>/dl/
        sequence_dir = extraction_root / f"{source.sequence}-vi_data"
        if not sequence_dir.is_dir():
            extract_archive(local, sequence_dir)
        euroc_dir = sequence_dir / "mav0"
        if not euroc_dir.is_dir():
            convert_tumvie_vidata_to_euroc_folder(sequence_dir)
        bag_path = extraction_root / "bags" / f"{source.sequence}.bag"
        stats = convert_euroc_folder_to_rosbag(euroc_dir, bag_path)
        _write_provenance(bag_path, dataset, source,
                          {"timestamps": "original", **stats})
    elif dataset.native_format == "tar_archive":
        extraction = extract_archive(local)
        sequence_dir = extraction / source.sequence
        recordings = sorted(d for d in sequence_dir.iterdir() if d.is_dir()
                            and list(d.glob("*.mp4"))) \
            if sequence_dir.is_dir() else []
        if recordings:                       # PennCOSYVIO: as/af/bs/bf subdirs
            bag_path = None
            for recording in recordings:
                bag_path = bags_dir / f"{source.sequence}_{recording.name}.bag"
                stats = convert_penncosyvio_sequence_to_rosbag(recording,
                                                               bag_path)
                _write_provenance(bag_path, dataset, source,
                                  {"timestamps": "original",
                                   "recording": recording.name, **stats})
        elif sequence_dir.is_dir() and any(
                p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                for sub in sequence_dir.iterdir() if sub.is_dir()
                for p in sub.iterdir()):     # intrinsic stills: c1/c2/... dirs
            bag_path = None
            for camera_dir in sorted(d for d in sequence_dir.iterdir()
                                     if d.is_dir()):
                bag_path = bags_dir / f"{source.sequence}_{camera_dir.name}.bag"
                stats = convert_image_folder_to_rosbag(camera_dir, bag_path)
                _write_provenance(bag_path, dataset, source, stats)
        else:
            bag_path = bags_dir / f"{source.sequence}.bag"
            stats = convert_penncosyvio_sequence_to_rosbag(sequence_dir,
                                                           bag_path)
            _write_provenance(bag_path, dataset, source,
                              {"timestamps": "original", **stats})
    elif dataset.native_format == "image_folder":
        bag_path = bags_dir / f"{source.sequence}.bag"
        stats = convert_image_folder_to_rosbag(local, bag_path)
        _write_provenance(bag_path, dataset, source, stats)
    else:
        raise NotImplementedError(
            f"{dataset.key}: no conversion route for native format "
            f"{dataset.native_format!r}")

    if also_mcap:
        convert_rosbag_to_mcap(bag_path)
    return bag_path


def convert_dataset_to_rosbag(dataset: CalibrationDataset, *,
                              sequences: list[str] | None = None,
                              also_mcap: bool = False) -> dict:
    chosen = ([dataset.get_sequence(name) for name in sequences]
              if sequences else dataset.sources)
    results = {}
    for source in chosen:
        bag = convert_sequence_to_rosbag(dataset, source, also_mcap=also_mcap)
        results[source.sequence] = str(bag) if bag else "not_local"
    return results
