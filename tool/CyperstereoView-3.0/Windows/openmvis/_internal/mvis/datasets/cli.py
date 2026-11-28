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
"""CLI for the dataset catalog: ``mvis datasets <action>``.

Thin by design — all logic lives in catalog.py / download.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from .catalog import DatasetCatalog


def add_datasets_subparser(subparsers) -> None:
    p = subparsers.add_parser(
        "datasets", help="Download / verify the calibration dataset catalog")
    p.add_argument("action", choices=["list", "download", "verify", "convert",
                                      "upload", "export-recipes"])
    p.add_argument("--mcap", action="store_true",
                   help="also emit a ROS2/mcap twin per produced bag")
    p.add_argument("--dry-run", action="store_true",
                   help="upload: print the plan without pushing anything")
    p.add_argument("--allow-unverified", action="store_true",
                   help="download: also fetch sources the catalog has no "
                        "sha256 for (no integrity check; skipped otherwise)")
    p.add_argument("--out", default="recipes",
                   help="export-recipes: output directory (default: recipes)")
    p.add_argument("--suite", choices=["camera-only", "camera-imu"],
                   help="restrict to one test-suite's sequences")
    p.add_argument("--dataset", help="one dataset key")
    p.add_argument("--select", action="append", default=[],
                   metavar="DATASET[/SEQUENCE]",
                   help="repeatable sequence-level selection")
    p.add_argument("--selection",
                   help="built-in selection name (e.g. 'smoke') or a JSON file "
                        "mapping dataset -> [sequences]")
    p.set_defaults(func=run_datasets_command)


def _resolve_selection(catalog, args):
    """Return list[(dataset, sequences|None)] from the CLI arguments."""
    if args.selection:
        spec = (args.selection if args.selection in
                DatasetCatalog.BUILTIN_SELECTIONS
                else json.loads(Path(args.selection).read_text()))
        return [(catalog.get(i.dataset_key), list(i.sequences) or None)
                for i in catalog.select(spec)]
    if args.select:
        return [(catalog.get(i.dataset_key), list(i.sequences) or None)
                for i in catalog.select(args.select)]
    suite = args.suite.replace("-", "_") if args.suite else None
    datasets = catalog.filter(suite=suite) if suite else list(catalog)
    if args.dataset:
        datasets = [catalog.get(args.dataset)]
    return [(d, None) for d in datasets]


def run_datasets_command(args) -> int:
    catalog = DatasetCatalog.load()
    suite = args.suite.replace("-", "_") if args.suite else None
    if args.action == "export-recipes":
        from .recipe_export import export_standalone_recipes
        for path in export_standalone_recipes(args.out):
            print(f"wrote {path}")
        return 0
    failed = 0
    uploaded = 0
    for dataset, sequences in _resolve_selection(catalog, args):
        chosen = sequences or [s.sequence
                               for s in dataset.sequences(suite=suite)]
        if args.action == "list":
            print(f"{dataset.key}  [{dataset.license_bucket}] {dataset.title}"
                  f"  — {dataset.license}")
            for s in dataset.sequences(suite=suite):
                if sequences and s.sequence not in sequences:
                    continue
                gb = f"{s.size_bytes/1e9:6.2f} GB" if s.size_bytes else "   ?    "
                url = "no public url" if s.url is None else s.url
                print(f"    {s.sequence:24s} {gb}  {url}")
            continue
        if args.action == "verify":
            report = dataset.verify_local_files(sequences=chosen)
        elif args.action == "convert":
            report = dataset.convert_to_rosbag(sequences=chosen,
                                               also_mcap=args.mcap)
        elif args.action == "upload":
            from .huggingface import upload_dataset_to_huggingface
            report = upload_dataset_to_huggingface(
                dataset, sequences=chosen, dry_run=args.dry_run,
                with_sources=False)       # SOURCES.md once, after the loop
            uploaded += sum(v == "ok" for v in report.values())
        else:
            report = dataset.download(sequences=chosen,
                                      allow_unverified=args.allow_unverified)
        bad = {k: v for k, v in report.items()
               if v in ("missing", "size_mismatch", "checksum_mismatch",
                        "size_mismatch_partial", "not_converted",
                        "unverified_skipped")}
        failed += len(bad)
        status = "OK" if not bad else f"{len(bad)} problem(s): {bad}"
        print(f"{dataset.key}: {len(report)} sequence(s) — {status}")
    if args.action == "upload" and uploaded and not args.dry_run:
        from .huggingface import upload_sources_markdown
        print(f"refreshed {upload_sources_markdown()}")
    return 1 if failed else 0
