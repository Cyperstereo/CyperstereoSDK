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
"""Export self-contained conversion recipes for the calibench HF repo.

Restricted-license datasets (NC-SA, GPL, citation-only, no licence) are never
re-hosted; instead the HF repo's ``recipes/`` folder carries standalone
scripts so anyone can rebuild the exact same bags locally. The recipes are
standalone so Hub users need no MVIS install: the exported ``convert.py``
embeds the converter module sources verbatim — the lib stays the single
source of truth, and ``tests/datasets/test_datasets_recipes.py`` asserts the
embedded code matches the lib byte-for-byte. ``SOURCES.md`` (creator,
licence, origin and citation of every catalog dataset) is generated from the
catalog's attribution blocks next to it.

Dependencies of the exported script: stdlib + ``rosbags`` + ``numpy`` +
``Pillow`` (+ ``av`` only for the PennCOSYVIO video route).
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from importlib import resources
from pathlib import Path

from .catalog import CalibrationDataset, DatasetCatalog

# Concatenation order matters: euroc_folder_to_rosbag defines the shared
# typestore/message helpers the later modules import from it.
EMBEDDED_MODULES = [
    "archive",
    "euroc_folder_to_rosbag",
    "tumvie_to_euroc_folder",
    "penncosyvio_to_rosbag",
    "image_folder_to_rosbag",
    "rosbag_mono16_to_mono8",
    "rosbag_to_mcap",
]

_HEADER = '''\
#!/usr/bin/env python3
"""Calibench — standalone dataset-to-rosbag converters (generated {stamp}).

GENERATED from mvis.datasets.converters by recipe_export.py — do not edit
here; the mvis library is the source of truth. Needs: numpy, Pillow,
rosbags (and av for the PennCOSYVIO video route).

    python convert.py euroc_folder <mav0 dir> <out.bag>
    python convert.py tumvie <seq-vi_data dir>            # emits mav0/
    python convert.py penncosyvio <seq dir> <out.bag>
    python convert.py image_folder <dir> <out.bag> [rate_hz]
    python convert.py to_mcap <in.bag>

Licenses stay with the source datasets — see SOURCES.md. Bags built this
way are byte-identical to the ones on calibench (same code path).
"""
'''

_RELATIVE_IMPORT = re.compile(
    r"^from \.\S+ import (?:\([^)]*\)|[^\n]+)\n", re.M)

_MAIN = '''

if __name__ == "__main__":
    import sys
    actions = {
        "euroc_folder": lambda a: convert_euroc_folder_to_rosbag(a[0], a[1]),
        "tumvie": lambda a: convert_tumvie_vidata_to_euroc_folder(a[0]),
        "penncosyvio":
            lambda a: convert_penncosyvio_sequence_to_rosbag(a[0], a[1]),
        "image_folder": lambda a: convert_image_folder_to_rosbag(
            a[0], a[1], frame_rate_hz=float(a[2]) if len(a) > 2 else 1.0),
        "to_mcap": lambda a: convert_rosbag_to_mcap(a[0]),
    }
    if len(sys.argv) < 2 or sys.argv[1] not in actions:
        raise SystemExit(f"usage: convert.py {{{'|'.join(actions)}}} ...")
    print(actions[sys.argv[1]](sys.argv[2:]))
'''


def _module_source(name: str) -> str:
    return (resources.files("mvis.datasets.converters")
            .joinpath(f"{name}.py").read_text())


def embedded_module_body(name: str) -> str:
    """A converter module's source with future/relative imports stripped
    (all modules share one namespace in the exported script)."""
    source = _module_source(name)
    source = source.replace("from __future__ import annotations\n", "")
    return _RELATIVE_IMPORT.sub("", source)


def build_standalone_convert_script() -> str:
    stamp = _dt.date.today().isoformat()
    parts = [_HEADER.format(stamp=stamp),
             "from __future__ import annotations\n"]
    for name in EMBEDDED_MODULES:
        parts.append(f"\n# ===== mvis.datasets.converters.{name} =====\n")
        parts.append(embedded_module_body(name))
    parts.append(_MAIN)
    return "".join(parts)


def build_conversion_manifest() -> str:
    """conversion_manifest.json: which converter chain rebuilds each dataset."""
    entries = {}
    for dataset in DatasetCatalog.load():
        entries[dataset.key] = {
            "title": dataset.title,
            "license_bucket": dataset.license_bucket,
            "license": dataset.license,
            "hosted_converted": dataset.hosted_on_hub,
            "native_format": dataset.native_format,
            "conversion_chain": [step.converter
                                 for step in dataset.conversion_chain],
        }
    return json.dumps({"generated": _dt.date.today().isoformat(),
                       "datasets": entries}, indent=1) + "\n"


def attribution_lines(dataset: CalibrationDataset, *,
                      license_note: str | None = None) -> list[str]:
    """Markdown bullets with a dataset's creator, licence, origin and
    citation — shared by SOURCES.md and the Hub dataset cards.
    ``license_note`` is appended to the licence bullet in parentheses."""
    a = dataset.attribution
    note = f" ({license_note})" if license_note else ""
    if a is None:
        return [f"- **License**: {dataset.license}{note}"]
    licence = a.license + (f" — <{a.license_url}>" if a.license_url else "")
    lines = [f"- **Creator**: {a.creator}",
             f"- **License**: {licence}{note}"]
    if a.source_url:
        lines.append(f"- **Source**: <{a.source_url}>")
    cite = ", ".join(part for part in (a.citation, f"doi:{a.doi}" if a.doi else None)
                     if part)
    if cite:
        lines.append(f"- **Cite**: {cite}")
    return lines


def _hosting_text(dataset: CalibrationDataset) -> str:
    """Where a dataset's files live, for SOURCES.md (same rule as the
    manifest's ``hosted_converted``)."""
    if dataset.hosted_on_hub:
        return "yes, with attribution"
    if dataset.hosting_policy == "recipe_only":
        return "recipe only (fetched from the original host)"
    return "original host (ROS1 bags need no conversion)"


def build_sources_markdown(catalog: DatasetCatalog | None = None) -> str:
    """recipes/SOURCES.md: creator, licence, origin and citation per dataset.

    This is the attribution the CC BY conversions under ``converted/`` carry,
    and the reference the dataset cards point at.
    """
    catalog = catalog or DatasetCatalog.load()
    hosted = [d for d in catalog if d.hosting_policy != "recipe_only"]
    recipe_only = [d for d in catalog if d.hosting_policy == "recipe_only"]
    # No date in the body: the content changes only when the catalog does,
    # so a re-upload of unchanged attribution is a no-op on the Hub.
    lines = [
        "# Calibench dataset sources — creators, licences, citations",
        "",
        "Generated from the MVIS dataset catalog (`mvis datasets "
        "export-recipes`, refreshed by `mvis datasets upload`).",
        "Every dataset stays under its original licence; if you use one, cite",
        "its paper. Datasets marked *recipe only* are never re-hosted:",
        "`convert.py` rebuilds their bags from the original host per",
        "`conversion_manifest.json`.",
        "",
        "| Key | Dataset | License | Hosted here |",
        "|---|---|---|---|",
    ]
    for d in catalog:
        lines.append(f"| `{d.key}` | {d.title} | {d.license} | "
                     f"{_hosting_text(d)} |")
    for heading, group in (("## Hosted and re-hostable datasets", hosted),
                           ("## Recipe-only datasets", recipe_only)):
        lines += ["", heading]
        for d in group:
            lines += ["", f"### {d.title} — `{d.key}`"]
            lines += attribution_lines(d)
            if d.notes:
                lines.append(f"- **Notes**: {d.notes}")
    return "\n".join(lines) + "\n"


def export_standalone_recipes(output_dir: str | Path) -> list[Path]:
    """Write convert.py, conversion_manifest.json and SOURCES.md into
    ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script = output_dir / "convert.py"
    script.write_text(build_standalone_convert_script())
    manifest = output_dir / "conversion_manifest.json"
    manifest.write_text(build_conversion_manifest())
    sources = output_dir / "SOURCES.md"
    sources.write_text(build_sources_markdown())
    return [script, manifest, sources]
