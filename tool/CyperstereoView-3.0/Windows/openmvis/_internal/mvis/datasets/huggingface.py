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
"""Upload converted bags + dataset cards to the calibench Hugging Face repo.

Only datasets whose ``hosting_policy`` allows hosting are uploadable:
``host_both_formats`` (ours: .bag + .mcap side by side) and ``host_rosbag``
(CC BY third-party conversions and partner recordings hosted with
permission, ROS1 first). ``recipe_only`` datasets are refused — they are
reproduced via the exported recipes instead (see recipe_export.py).

Uploads shell out to the ``hf`` CLI (huggingface_hub is not a dependency of
the mvis env); pass ``runner`` to intercept in tests. Repo layout:

    converted/<dataset_key>/<sequence>[_<part>].bag|.mcap
    converted/<dataset_key>/<...>.provenance.json
    converted/<dataset_key>/README.md          # the dataset card
    recipes/SOURCES.md                         # attribution, regenerated per upload
"""
from __future__ import annotations

import datetime as _dt
import re
import json
import subprocess
import tempfile
from pathlib import Path

from .catalog import CalibrationDataset, DataSource
from .recipe_export import attribution_lines, build_sources_markdown

HF_DATASET_REPO = "calibench-vi/datasets"
CONVERTED_PREFIX = "converted"
SOURCES_PATH = "recipes/SOURCES.md"

UPLOADABLE_POLICIES = {"host_both_formats", "host_rosbag"}

_LICENSE_NOTE = {
    "own": "ours — Calibench original recording; commercial use included; access is gated",
    "partner": "partner recording, hosted with permission; commercial use included; access is gated",
    "cc_by": "inherited from the source dataset",
}


def converted_bag_dirs(source: DataSource) -> list[Path]:
    """Candidate ``bags/`` directories conversion.py writes for a source."""
    local = source.local_path
    return [local.parent / "bags", local.parent.parent / "bags"]


def converted_files_for(dataset: CalibrationDataset, source: DataSource,
                        *, include_mcap: bool) -> list[Path]:
    """Existing converted outputs for one sequence (bags, sidecars, mcaps).

    A rosbag-native dataset's bags are already hosted in their own HF
    folders, so its only uploadable artifact is the local .mcap twin
    (a ROS2 folder bag next to the source .bag).
    """
    if dataset.native_format == "rosbag":
        twin = source.local_path.with_suffix(".mcap")
        return [twin] if twin.exists() else []
    suffixes = [".bag", ".provenance.json"] + ([".mcap"] if include_mcap else [])
    found: list[Path] = []
    for bags_dir in converted_bag_dirs(source):
        if not bags_dir.is_dir():
            continue
        for path in sorted(bags_dir.iterdir()):
            stem_ok = (path.name.startswith(source.sequence + "_")
                       or path.name.split(".")[0] == source.sequence)
            if stem_ok and any(path.name.endswith(s) for s in suffixes):
                found.append(path)
    return found


def build_dataset_card(dataset: CalibrationDataset,
                       files: list[Path], *, version_stamp: str) -> str:
    """Markdown card for converted/<key>/README.md."""
    lines = [
        f"# {dataset.title} — converted ROS1 bags",
        "",
        f"- **Dataset key**: `{dataset.key}`",
        *attribution_lines(
            dataset, license_note=_LICENSE_NOTE.get(dataset.license_bucket)),
        f"- **Version**: {version_stamp}",
        f"- **Conversion chain**: "
        + (" → ".join(f"`{step.converter}`" for step in dataset.conversion_chain)
           or "already ROS1 bags"),
        "",
        "Each `.bag` has a `.provenance.json` sidecar recording the source "
        "URL, source sha256, license bucket, conversion chain, and topic "
        "counts. Timestamps are original unless the sidecar says "
        "`\"timestamps\": \"synthetic\"` (still-image sets — usable for "
        "cam-only intrinsics, never for time-offset / cam-IMU work).",
        "",
        "| File | Size | sha256 (first 16) |",
        "|---|---|---|",
    ]
    for path in files:
        if path.suffix == ".json":
            continue
        digest = ""
        sidecar = path.with_suffix(".provenance.json")
        if sidecar.exists():
            digest = json.loads(sidecar.read_text()).get("source_sha256") or ""
        lines.append(f"| `{path.name}` | {path.stat().st_size/1e9:.2f} GB "
                     f"| {digest[:16]} |")
    if dataset.notes:
        lines += ["", dataset.notes]
    lines += ["", f"Sources and attribution of every dataset: `{SOURCES_PATH}` "
                  "at the repo root. If you use this data, cite the original "
                  "dataset's paper."]
    return "\n".join(lines) + "\n"


def plan_upload(dataset: CalibrationDataset, *,
                sequences: list[str] | None = None) -> dict:
    """Resolve what an upload would push: {sequence: [files]}; no side effects."""
    if dataset.hosting_policy not in UPLOADABLE_POLICIES:
        raise ValueError(
            f"{dataset.key}: hosting_policy {dataset.hosting_policy!r} "
            f"forbids uploading — use the exported recipes instead")
    include_mcap = dataset.hosting_policy == "host_both_formats"
    chosen = ([dataset.get_sequence(name) for name in sequences]
              if sequences else dataset.sources)
    return {source.sequence:
            converted_files_for(dataset, source, include_mcap=include_mcap)
            for source in chosen}


def upload_sources_markdown(*, repo: str = HF_DATASET_REPO,
                            runner=subprocess.run) -> str:
    """Regenerate ``recipes/SOURCES.md`` from the catalog and push it: the
    attribution file every dataset card points at. Returns the remote path."""
    with tempfile.TemporaryDirectory() as tmp:
        sources = Path(tmp) / "SOURCES.md"
        sources.write_text(build_sources_markdown())
        runner(["hf", "upload", repo, str(sources), SOURCES_PATH,
                "--repo-type", "dataset"],
               check=True)
    return SOURCES_PATH


def upload_dataset_to_huggingface(dataset: CalibrationDataset, *,
                                  sequences: list[str] | None = None,
                                  repo: str = HF_DATASET_REPO,
                                  runner=subprocess.run,
                                  dry_run: bool = False,
                                  with_sources: bool = True) -> dict:
    """Upload one dataset's converted bags + card. Returns the plan.

    Sequences whose conversion output is missing are reported as
    ``"not_converted"`` and skipped, never silently uploaded partial.
    ``with_sources`` also refreshes ``recipes/SOURCES.md``; the CLI passes
    False and pushes it once after its loop over datasets.
    """
    plan = plan_upload(dataset, sequences=sequences)
    version_stamp = _dt.date.today().isoformat()
    files = [path for paths in plan.values() for path in paths]
    report: dict[str, str] = {
        seq: ("ok" if paths else "not_converted")
        for seq, paths in plan.items()}
    if dry_run or not files:
        return report

    by_source = {source.sequence: source for source in dataset.sources}
    for seq, paths in plan.items():
        for path in paths:
            runner(["hf", "upload", repo, str(path),
                    _remote_location(dataset, by_source[seq], path),
                    "--repo-type", "dataset"],
                   check=True)
    if dataset.native_format != "rosbag":
        # rosbag-native folders keep their existing handwritten cards.
        # The card describes the whole folder, so build it from every
        # converted sequence -- not just the pushed subset, or a
        # --select upload would drop the other sequences' rows.
        card_files = [path for paths in plan_upload(dataset).values()
                      for path in paths]
        with tempfile.TemporaryDirectory() as tmp:
            card = Path(tmp) / "README.md"
            card.write_text(build_dataset_card(dataset, card_files,
                                               version_stamp=version_stamp))
            runner(["hf", "upload", repo, str(card),
                    f"{CONVERTED_PREFIX}/{dataset.key}/README.md",
                    "--repo-type", "dataset"],
                   check=True)
    if with_sources:
        upload_sources_markdown(repo=repo, runner=runner)
    return report


def _remote_location(dataset: CalibrationDataset, source: DataSource,
                     path: Path) -> str:
    """Remote path: mcap twins land beside their already-hosted bag (folder
    taken from the source URL); everything else under converted/<key>/."""
    m = re.search(r"/resolve/[^/]+/(.+)$", source.url or "")
    if dataset.native_format == "rosbag" and m:     # any revision: main or a pinned sha
        folder = m.group(1).rsplit("/", 1)[0]
        return f"{folder}/{path.name}"
    return f"{CONVERTED_PREFIX}/{dataset.key}/{path.name}"
