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
"""Dataset catalog: the class model over the packaged ``dataset_catalog.json``.

One ``CalibrationDataset`` per rig/dataset of the roster's master table; one
``DataSource`` per downloadable file (a *sequence*: a bag, a zip, a tarball).
Selection reaches sequence level: ``catalog.select(["tum_mic/MIC05"])``.

Terminology: the "EuRoC folder format" (a.k.a. the historical ``mav0/`` layout
from the EuRoC MAV dataset) is called ``euroc_folder`` throughout.
"""
from __future__ import annotations

import json
import os
import urllib.parse
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

# The default root, ~/Datasets. A source's `local_path` follows `core.paths.datasets_root()`
# (``$MVIS_DATASETS`` when set) read at call time, like the configs' ${MVIS_DATASETS}.
# Assigning another path to DATASETS_ROOT (as the tests do) pins the root instead.
_DEFAULT_ROOT = Path.home() / "Datasets"
DATASETS_ROOT = _DEFAULT_ROOT


def catalog_root() -> Path:
    """The folder `DataSource.local_relative_path` is relative to: a pinned `DATASETS_ROOT`,
    else ``$MVIS_DATASETS`` (a leading ``~`` expanded), else ``~/Datasets``."""
    if DATASETS_ROOT is not _DEFAULT_ROOT:
        return Path(DATASETS_ROOT)
    from mvis.core.paths import datasets_root   # lazy: mvis.core loads numpy and yaml
    return Path(os.path.expanduser(datasets_root()))

LICENSE_BUCKETS = {"own", "cc_by", "cc_by_nc_sa", "no_license", "citation_only",
                   "gpl_v2", "partner"}
# Buckets whose terms forbid re-hosting: those datasets stay recipe_only and
# are fetched from their original host. "own" (VI-Rig) and "partner"
# (recordings hosted with their owner's permission) are CC BY 4.0, commercial
# use included, behind the Hub's access form; "cc_by" sets may be re-hosted
# as conversions with attribution.
RESTRICTED_BUCKETS = {"cc_by_nc_sa", "no_license", "citation_only", "gpl_v2"}
HOSTING_POLICIES = {"host_both_formats", "host_rosbag", "already_hosted",
                    "recipe_only"}
HUB_HOSTS = {"huggingface.co", "hf.co"}
SUITES = {"camera_only", "camera_imu"}
NATIVE_FORMATS = {"rosbag", "euroc_folder", "tumvie_vidata", "mcap",
                  "image_folder", "tar_archive", "zip_archive"}

# Converter registry keys (functions land in mvis.datasets.converters at M2;
# the catalog validates chain names against this set already).
KNOWN_CONVERTERS = {
    "convert_euroc_folder_to_rosbag",
    "convert_tumvie_vidata_to_euroc_folder",
    "convert_rosbag_mono16_to_mono8",
    "convert_mcap_to_euroc_folder",
    "convert_penncosyvio_sequence_to_rosbag",
    "convert_image_folder_to_rosbag",
    "convert_rosbag_to_mcap",
    "extract_archive",
}


@dataclass(frozen=True)
class DataSource:
    """One downloadable file (one sequence of a dataset)."""
    sequence: str                    # human label: "april_01", "imu1", "MIC05"
    local_relative_path: str        # relative to catalog_root()
    url: str | None = None          # None: no public link
    sha256: str | None = None       # None: not yet computed
    size_bytes: int | None = None
    suite: str | None = None        # "camera_only"/"camera_imu"; None = both

    @property
    def local_path(self) -> Path:
        return catalog_root() / self.local_relative_path


@dataclass(frozen=True)
class ConversionStep:
    """One converter invocation in a dataset's chain."""
    converter: str
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Attribution:
    """Who made a dataset and under which terms it is used here.

    Drives the licence line of ``mvis datasets list``, the Hub dataset cards
    and the generated ``recipes/SOURCES.md`` (CC BY attribution for the
    re-hosted conversions).
    """
    creator: str
    license: str                      # "CC BY 4.0", "CC BY-NC-SA 3.0", ...
    license_url: str | None = None
    source_url: str | None = None     # homepage / original host
    citation: str | None = None       # one-line reference (authors, title, venue, year)
    doi: str | None = None


@dataclass
class CalibrationDataset:
    """One dataset (a master-table row)."""
    key: str
    title: str
    suites: list[str]
    license_bucket: str
    hosting_policy: str
    native_format: str
    sources: list[DataSource]
    conversion_chain: list[ConversionStep] = field(default_factory=list)
    notes: str = ""
    attribution: Attribution | None = None

    @property
    def license(self) -> str:
        """Human-readable licence ("CC BY 4.0") from the attribution block
        (every packaged entry has one); the raw bucket name without it."""
        if self.attribution is not None:
            return self.attribution.license
        return self.license_bucket

    @property
    def hosted_on_hub(self) -> bool:
        """True when calibench-vi/datasets carries this dataset's files: the
        policy allows hosting and the files are either conversions we make
        or bags whose catalog URLs already point at the Hub. ROS1-native sets
        served from their original host (TUM-VI, TUM RGB-D) are not."""
        if self.hosting_policy == "recipe_only":
            return False
        if self.native_format != "rosbag":
            return True
        return any(_on_hub(s.url) for s in self.sources)

    def sequences(self, *, suite: str | None = None) -> list[DataSource]:
        return [s for s in self.sources
                if suite is None or s.suite is None or s.suite == suite]

    def get_sequence(self, name: str) -> DataSource:
        for s in self.sources:
            if s.sequence == name:
                return s
        valid = ", ".join(s.sequence for s in self.sources)
        raise KeyError(
            f"unknown sequence {name!r} in dataset {self.key!r}; valid: {valid}")

    # Behavior delegates to sibling modules (kept boring on purpose).
    def download(self, *, sequences: list[str] | None = None, **kw):
        from . import download as _dl
        return _dl.download_dataset(self, sequences=sequences, **kw)

    def verify_local_files(self, *, sequences: list[str] | None = None):
        from . import download as _dl
        return _dl.verify_local_files(self, sequences=sequences)

    def convert_to_rosbag(self, *, sequences: list[str] | None = None,
                          also_mcap: bool = False):
        from . import conversion as _cv
        return _cv.convert_dataset_to_rosbag(
            self, sequences=sequences, also_mcap=also_mcap)


def _on_hub(url: str | None) -> bool:
    host = urllib.parse.urlparse(url or "").hostname or ""
    return host in HUB_HOSTS or any(host.endswith("." + h) for h in HUB_HOSTS)


@dataclass(frozen=True)
class SelectionItem:
    dataset_key: str
    sequences: tuple[str, ...] = ()   # empty = every sequence


class DatasetCatalog:
    """Load, filter and select from the packaged dataset catalog."""

    #: built-in named selections; "smoke" = one small sequence per major rig.
    BUILTIN_SELECTIONS: dict[str, dict[str, list[str]]] = {
        "smoke": {
            "mvis_4imus_3cams": ["april_01"],
            "tum_vi_raw": ["imu1"],
            "tum_vi_standard_512": ["imu1"],
            "tum_mic": ["MIC05"],
            "monado_moc": ["MOC02_imucalib_1"],
            "tum_event": ["imu_calibA1"],
            "uzh_fpv_davis": ["indoor_forward_imu"],
            "penncosyvio": ["c3"],
        },
    }

    def __init__(self, datasets: list[CalibrationDataset]):
        self._by_key = {d.key: d for d in datasets}

    # ---- construction -----------------------------------------------------
    @classmethod
    def load(cls, path: str | Path | None = None) -> "DatasetCatalog":
        if path is not None:
            raw = json.loads(Path(path).read_text())
        else:
            raw = json.loads(
                resources.files("mvis.datasets")
                .joinpath("dataset_catalog.json").read_text())
        datasets = []
        for entry in raw["datasets"]:
            sources = [DataSource(**s) for s in entry.pop("sources")]
            chain = [ConversionStep(**c)
                     for c in entry.pop("conversion_chain", [])]
            raw_attribution = entry.pop("attribution", None)
            attribution = (Attribution(**raw_attribution)
                           if raw_attribution is not None else None)
            datasets.append(CalibrationDataset(
                sources=sources, conversion_chain=chain,
                attribution=attribution, **entry))
        catalog = cls(datasets)
        catalog._validate()
        return catalog

    def _validate(self) -> None:
        for d in self:
            assert d.suites and set(d.suites) <= SUITES, d.key
            assert d.license_bucket in LICENSE_BUCKETS, d.key
            assert d.hosting_policy in HOSTING_POLICIES, d.key
            assert d.native_format in NATIVE_FORMATS, d.key
            assert d.sources, f"{d.key}: no sources"
            for step in d.conversion_chain:
                assert step.converter in KNOWN_CONVERTERS, (
                    f"{d.key}: unknown converter {step.converter!r}")
            if d.license_bucket in RESTRICTED_BUCKETS:
                assert d.hosting_policy == "recipe_only", (
                    f"{d.key}: restricted license cannot be hosted")
            # Every entry states who made it and under which terms (the
            # licence column of `mvis datasets list`, SOURCES.md); hosting
            # additionally needs the licence URL and origin that CC BY and
            # the Hub cards require.
            assert d.attribution is not None, (
                f"{d.key}: missing attribution block")
            assert d.attribution.creator and d.attribution.license, d.key
            if d.hosting_policy != "recipe_only":
                assert d.attribution.license_url and d.attribution.source_url, (
                    f"{d.key}: attribution needs license_url and source_url")

    # ---- access -----------------------------------------------------------
    def __iter__(self):
        return iter(self._by_key.values())

    def keys(self) -> list[str]:
        return list(self._by_key)

    def get(self, key: str) -> CalibrationDataset:
        try:
            return self._by_key[key]
        except KeyError:
            raise KeyError(f"unknown dataset {key!r}; "
                           f"valid: {', '.join(self._by_key)}") from None

    def filter(self, *, suite: str | None = None,
               license_bucket: str | None = None,
               hosting_policy: str | None = None) -> list[CalibrationDataset]:
        out = []
        for d in self:
            if suite and suite not in d.suites:
                continue
            if license_bucket and d.license_bucket != license_bucket:
                continue
            if hosting_policy and d.hosting_policy != hosting_policy:
                continue
            out.append(d)
        return out

    # ---- selection --------------------------------------------------------
    def select(self, spec) -> list[SelectionItem]:
        """Resolve a selection into items, failing loudly on unknown names.

        ``spec`` may be: a built-in selection name ("smoke"), a mapping
        {dataset_key: [sequences...]} (empty list = all), or an iterable of
        "dataset" / "dataset/sequence" strings.
        """
        if isinstance(spec, str):
            if spec not in self.BUILTIN_SELECTIONS:
                raise KeyError(
                    f"unknown built-in selection {spec!r}; "
                    f"valid: {', '.join(self.BUILTIN_SELECTIONS)}")
            spec = self.BUILTIN_SELECTIONS[spec]
        if isinstance(spec, dict):
            pairs = [(k, tuple(v)) for k, v in spec.items()]
        else:
            merged: dict[str, list[str]] = {}
            for token in spec:
                key, _, seq = str(token).partition("/")
                merged.setdefault(key, [])
                if seq:
                    merged[key].append(seq)
            pairs = [(k, tuple(v)) for k, v in merged.items()]
        items = []
        for key, seqs in pairs:
            dataset = self.get(key)                 # raises with valid keys
            for name in seqs:
                dataset.get_sequence(name)          # raises with valid names
            items.append(SelectionItem(key, seqs))
        return items

    # ---- audits -----------------------------------------------------------
    def verify_all_local_files(self, *, suite: str | None = None) -> dict:
        report: dict[str, dict] = {}
        for d in (self.filter(suite=suite) if suite else self):
            report[d.key] = d.verify_local_files()
        return report
