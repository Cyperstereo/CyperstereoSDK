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
"""Archive extraction (zip / tar / tar.gz) with a no-op when already done."""
from __future__ import annotations

import os
import posixpath
import tarfile
import zipfile
from pathlib import Path


def _check_tar_members(tf: tarfile.TarFile, destination: Path) -> None:
    """Reject members that would land outside ``destination``: ``..`` or
    absolute names, links pointing out of the tree, paths routed through an
    earlier symlink member, and device nodes.

    ``tarfile``'s ``filter="data"`` (Python >= 3.11.4) does the same during
    extraction; this check runs first so a hostile archive is refused on
    every interpreter (the standalone recipes embed this module), and
    before the already-extracted shortcut.
    """
    root = os.path.realpath(destination)
    symlinks: set[str] = set()      # normalised names of symlink members so far
    for member in tf.getmembers():
        name = member.name
        if member.isdev():
            raise ValueError(f"{tf.name}: refusing device member {name!r}")
        # Walk the raw components: "a/link/../x" passes through "a/link"
        # even though it normalises to "a/x".
        prefix = ""
        for part in name.split("/")[:-1]:
            prefix = posixpath.normpath(posixpath.join(prefix, part))
            if prefix in symlinks:
                raise ValueError(f"{tf.name}: member {name!r} is routed "
                                 f"through symlink member {prefix!r}")
        target = os.path.realpath(os.path.join(root, name))
        if os.path.commonpath([root, target]) != root:
            raise ValueError(
                f"{tf.name}: member {name!r} would extract outside {destination}")
        if member.issym() or member.islnk():
            base = root if member.islnk() else os.path.dirname(target)
            link = os.path.realpath(os.path.join(base, member.linkname))
            if os.path.commonpath([root, link]) != root:
                raise ValueError(
                    f"{tf.name}: link member {name!r} points outside {destination}")
            if member.issym():
                symlinks.add(posixpath.normpath(name))


def _extract_tar(tf: tarfile.TarFile, destination: Path) -> None:
    if hasattr(tarfile, "data_filter"):      # extraction filters: Python >= 3.11.4
        tf.extractall(destination, filter="data")
    else:
        tf.extractall(destination)


def extract_archive(archive_path: str | Path,
                    destination_dir: str | Path | None = None) -> Path:
    """Extract ``archive_path`` next to itself (or into ``destination_dir``).

    Returns the extraction directory. Skips work when the archive's top-level
    entries already exist at the destination. Tar members that would escape
    the destination (path traversal, outward links) raise ``ValueError``;
    ``zipfile`` already sanitises such names.
    """
    archive_path = Path(archive_path)
    destination = Path(destination_dir) if destination_dir \
        else archive_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            tops = {n.split("/")[0] for n in zf.namelist()}
            if not all((destination / t).exists() for t in tops):
                zf.extractall(destination)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            _check_tar_members(tf, destination)
            tops = {m.name.split("/")[0] for m in tf.getmembers()}
            if not all((destination / t).exists() for t in tops):
                _extract_tar(tf, destination)
    else:
        raise ValueError(f"not a zip or tar archive: {archive_path}")
    return destination
