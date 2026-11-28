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
"""Resumable download engine + local-file verification for the catalog.

Stdlib only (urllib, plus the stdlib-only `mvis.io.fileutil`), so the recipe exporter can reuse it.
"""
from __future__ import annotations

import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from mvis.io.fileutil import compute_sha256  # noqa: F401  (re-exported: download.compute_sha256)

CHUNK_BYTES = 1 << 20  # 1 MiB
HF_HOST = "huggingface.co"
HF_HOSTS = (HF_HOST, "hf.co")  # the Hub hosts the catalog accepts (catalog.HUB_HOSTS)


def hf_token() -> str | None:
    """Hugging Face token: ``HF_TOKEN``, else the file ``huggingface-cli login`` writes."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    path = Path(os.environ.get("HF_TOKEN_PATH", os.path.join(home, "token")))
    return path.read_text().strip() or None if path.is_file() else None


def _is_hf(url: str) -> bool:
    """True for an ``https://`` Hugging Face URL — the only place the bearer
    token may go (a plain-http URL would send it in cleartext)."""
    parts = urllib.parse.urlparse(url)
    host = parts.hostname or ""
    return parts.scheme == "https" and any(host == h or host.endswith("." + h) for h in HF_HOSTS)


class _DropAuthOnHostChange(urllib.request.HTTPRedirectHandler):
    """Follow redirects, but never forward the token to a different host or
    over anything but https."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is None:
            return None
        target = urllib.parse.urlparse(newurl)
        if (target.scheme != "https"
                or target.hostname != urllib.parse.urlparse(req.full_url).hostname):
            new.remove_header("Authorization")
        return new


_OPENER = urllib.request.build_opener(_DropAuthOnHostChange)


def download_file(url: str, destination: Path, *,
                  expected_sha256: str | None = None,
                  expected_size_bytes: int | None = None,
                  progress: bool = False) -> Path:
    """Download ``url`` to ``destination`` with resume + verification.

    Skips the transfer entirely when the destination already matches the
    expected size/sha256. Resumes a partial file via HTTP Range. Raises
    ``ValueError`` on checksum or size mismatch.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        status = _verify_one(destination, expected_sha256, expected_size_bytes)
        if status == "ok":
            return destination
        if status == "size_mismatch_partial":
            pass  # fall through to resume
        elif status in ("checksum_mismatch", "size_mismatch"):
            destination.unlink()  # corrupt: refetch from scratch

    offset = destination.stat().st_size if destination.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    token = hf_token() if _is_hf(url) else None
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        response = _OPENER.open(request)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403) and _is_hf(url):
            raise PermissionError(
                f"{url}: HTTP {e.code}. This dataset is gated: accept its terms on "
                "the Hugging Face page, then run `hf auth login` (older CLI: "
                "`huggingface-cli login`) or set HF_TOKEN."
                + ("" if token else " (no token found)")) from e
        raise
    with response:
        mode = "ab" if offset and response.status == 206 else "wb"
        with open(destination, mode) as out:
            while chunk := response.read(CHUNK_BYTES):
                out.write(chunk)
                if progress:
                    print(f"\r{destination.name}: {out.tell()/1e9:.2f} GB",
                          end="", flush=True)
    if progress:
        print()

    status = _verify_one(destination, expected_sha256, expected_size_bytes)
    if status != "ok" and status != "unverified":
        raise ValueError(f"{destination.name}: {status} after download")
    return destination


def _verify_one(path: Path, sha256: str | None, size: int | None) -> str:
    """Return ok / unverified / missing / size_mismatch(_partial) /
    checksum_mismatch for one local file."""
    if not path.exists():
        return "missing"
    if size is not None:
        actual = path.stat().st_size
        if actual < size:
            return "size_mismatch_partial"
        if actual != size:
            return "size_mismatch"
    if sha256 is not None:
        if compute_sha256(path) != sha256:
            return "checksum_mismatch"
        return "ok"
    return "ok" if size is not None else "unverified"


# ---- dataset-level operations (called by CalibrationDataset) --------------

def _pick(dataset, sequences):
    if sequences is None:
        return dataset.sources
    return [dataset.get_sequence(name) for name in sequences]


def download_dataset(dataset, *, sequences=None, progress=True,
                     allow_unverified=False) -> dict:
    """Fetch every selected source; returns ``{sequence: status}``.

    A source without a catalog ``sha256`` cannot be integrity-checked after
    the transfer. It is skipped (status ``"unverified_skipped"``, with a
    warning on stderr) unless ``allow_unverified`` is set, which fetches it
    anyway and only warns.
    """
    results = {}
    for source in _pick(dataset, sequences):
        if source.url is None:
            results[source.sequence] = "no_public_url"
            continue
        if source.sha256 is None:
            label = f"{dataset.key}/{source.sequence}"
            if not allow_unverified:
                print(f"WARNING: {label}: the catalog has no sha256 for "
                      f"{source.url}; skipped. Pass allow_unverified=True "
                      "(`mvis datasets download --allow-unverified`) to fetch "
                      "it without an integrity check.", file=sys.stderr)
                results[source.sequence] = "unverified_skipped"
                continue
            print(f"WARNING: {label}: no sha256 in the catalog — downloading "
                  f"{source.url} without an integrity check.", file=sys.stderr)
        download_file(source.url, source.local_path,
                      expected_sha256=source.sha256,
                      expected_size_bytes=source.size_bytes,
                      progress=progress)
        results[source.sequence] = "ok"
    return results


def verify_local_files(dataset, *, sequences=None) -> dict:
    """Size (+sha256 when known) verification without any network access."""
    return {
        source.sequence: _verify_one(source.local_path, source.sha256,
                                     source.size_bytes)
        for source in _pick(dataset, sequences)
    }
