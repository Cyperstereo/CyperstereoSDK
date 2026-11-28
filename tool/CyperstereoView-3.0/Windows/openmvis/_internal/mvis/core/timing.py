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
"""Hierarchical wall-clock timing for the calibration pipeline.

A ``Timer`` accumulates nested timing sections so the pipeline can record
not just per-stage totals but per-camera and per-sub-step breakdowns.
The intent is that every regression run produces actionable timing data
so we can answer questions like:

- "Is `top_left` actually 3× faster, or only 3× faster on Stage 4?"
- "For MIC, where does the wall-clock go — detection, Stage 4, or rerun?"
- "Does Stage 5 (rerun) earn its keep relative to its time cost?"

Design notes:

- **Single-threaded.** The calibration pipeline is sequential; we don't
  need locks, and overlapping ``with`` blocks are explicitly rejected
  rather than silently corrupting the call stack.
- **Cheap.** ``time.perf_counter`` deltas plus dict updates — well under
  a microsecond per ``section()`` call. Safe to wrap inner loops if
  needed.
- **No GTSAM/numpy coupling.** Pure Python so it can be unit-tested in
  isolation.
- **Optional metadata.** Each section can carry tags like
  ``n_observations`` or ``n_iterations`` so downstream analysis can
  compute normalised cost (s / 100k obs, s / iter, etc.).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class _Section:
    """One timed node in the `Timer` tree: name, wall-clock bounds, metadata, and
    the nested child sections opened while it was on the stack."""
    name: str
    start_perf: float
    metadata: dict = field(default_factory=dict)
    elapsed: float = 0.0          # filled in on close
    children: list["_Section"] = field(default_factory=list)


class Timer:
    """Accumulates a tree of timed sections.

    Usage::

        timer = Timer()
        with timer.section("stage1_init"):
            ...
            with timer.section("camera_0", n_observations=10000):
                ...
        timer.to_dict()
    """

    def __init__(self) -> None:
        """Start with an empty tree under a synthetic `<root>` section."""
        self._root: _Section = _Section(
            name="<root>", start_perf=time.perf_counter(),
        )
        self._stack: list[_Section] = [self._root]

    @contextmanager
    def section(self, name: str, **metadata) -> Iterator[None]:
        """Open a timed section. Sections nest based on entry order."""
        if not name:
            raise ValueError("Section name must be non-empty.")
        sec = _Section(
            name=name, start_perf=time.perf_counter(),
            metadata=dict(metadata),
        )
        self._stack[-1].children.append(sec)
        self._stack.append(sec)
        try:
            yield
        finally:
            popped = self._stack.pop()
            if popped is not sec:
                # Defensive: somebody manipulated the stack mid-section.
                raise RuntimeError(
                    f"Section stack corrupted: expected {sec.name!r}, "
                    f"got {popped.name!r}."
                )
            sec.elapsed = time.perf_counter() - sec.start_perf

    def record(self, name: str, seconds: float, **metadata) -> None:
        """Add a section with a known duration (no context manager)."""
        sec = _Section(
            name=name, start_perf=0.0, metadata=dict(metadata),
            elapsed=float(seconds),
        )
        self._stack[-1].children.append(sec)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a nested dict suitable for YAML / JSON output.

        Top-level: ``{"total_seconds": float, "sections": [...]}``.
        Each section: ``{"name", "seconds", "self_seconds", "metadata",
        "children"}``.
        """
        # Close any sections still open (treat as if exiting now); the
        # normal flow always closes via ``with``, so this is for
        # defensive serialization mid-pipeline.
        if len(self._stack) != 1:
            raise RuntimeError(
                "to_dict() called with open sections still on the stack."
            )
        sections = [self._serialize(c) for c in self._root.children]
        total = sum(s["seconds"] for s in sections)
        return {"total_seconds": total, "sections": sections}

    @staticmethod
    def _serialize(sec: _Section) -> dict:
        """One section as a plain dict, recursively; `self_seconds` = elapsed minus
        the children's total (clamped at 0 against clock skew)."""
        children = [Timer._serialize(c) for c in sec.children]
        children_total = sum(c["seconds"] for c in children)
        return {
            "name": sec.name,
            "seconds": sec.elapsed,
            "self_seconds": max(sec.elapsed - children_total, 0.0),
            "metadata": dict(sec.metadata),
            "children": children,
        }

    def to_flat_rows(self) -> list[dict]:
        """One row per section in tree order, with depth + path columns.

        Useful for dumping as a flat table (e.g. a CSV or HTML table).
        """
        rows: list[dict] = []

        def _walk(sec: _Section, depth: int, path: list[str]) -> None:
            """Append this section's row, then recurse into its children (pre-order)."""
            children_total = sum(c.elapsed for c in sec.children)
            rows.append({
                "path": ".".join(path + [sec.name]),
                "name": sec.name,
                "depth": depth,
                "seconds": sec.elapsed,
                "self_seconds": max(sec.elapsed - children_total, 0.0),
                "metadata": dict(sec.metadata),
            })
            for c in sec.children:
                _walk(c, depth + 1, path + [sec.name])

        for child in self._root.children:
            _walk(child, 0, [])
        return rows
