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
"""Safe YAML loading for result files, numpy-aware.

Result YAMLs are plain YAML today, but files written by old versions carried PyYAML
python tags for numpy scalars and arrays (and tuples), which ``yaml.safe_load`` refuses.
`NumpySafeLoader` is a ``yaml.SafeLoader`` that also builds those few tags from their data
without importing or calling anything named in the file: numpy scalars and arrays
(``numpy.core.multiarray.scalar`` / ``_reconstruct`` and the ``numpy._core`` spellings),
``numpy.dtype``, the ``numpy.ndarray`` name and ``python/tuple``. Any other python tag still raises, as in safe_load.
"""
from __future__ import annotations

import numpy as np
import yaml

_APPLY = "tag:yaml.org,2002:python/object/apply:"
_MODULES = ("numpy.core.multiarray", "numpy._core.multiarray")


class NumpySafeLoader(yaml.SafeLoader):
    """SafeLoader plus numpy scalar / array / dtype and tuple tags (data only)."""


def _apply_parts(loader, node):
    """(args, state) of a ``!!python/object/apply`` node, in its sequence or mapping form."""
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True), None
    value = loader.construct_mapping(node, deep=True)
    return list(value.get("args", [])), value.get("state")


def _as_bytes(data) -> bytes:
    # pickle protocol 2 writes the raw buffer as a latin-1 str (yaml.dump shows it as text)
    return data.encode("latin-1") if isinstance(data, str) else bytes(data)


def _dtype(loader, node):
    args, state = _apply_parts(loader, node)
    dt = np.dtype(str(args[0]))
    if isinstance(state, (list, tuple)) and len(state) > 1 and state[1] in "<>" \
            and dt.itemsize > 1:
        dt = dt.newbyteorder(state[1])
    return dt


def _scalar(loader, node):
    args, _ = _apply_parts(loader, node)
    dt = args[0]
    if not isinstance(dt, np.dtype):
        raise yaml.constructor.ConstructorError(None, None, "numpy scalar without a dtype",
                                                node.start_mark)
    return np.frombuffer(_as_bytes(args[1]), dtype=dt)[0]


def _reconstruct(loader, node):
    _, state = _apply_parts(loader, node)
    if not (isinstance(state, (list, tuple)) and len(state) == 5):
        raise yaml.constructor.ConstructorError(None, None, "numpy array without a state",
                                                node.start_mark)
    _, shape, dt, fortran, data = state
    order = "F" if fortran else "C"
    if isinstance(data, list):                       # object arrays pickle their items
        return np.array(data, dtype=object).reshape(shape, order=order)
    return np.frombuffer(_as_bytes(data), dtype=dt).reshape(shape, order=order).copy()


NumpySafeLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                                lambda loader, node: tuple(loader.construct_sequence(node)))
NumpySafeLoader.add_constructor(_APPLY + "numpy.dtype", _dtype)
# the array reducer names its class; only this one name is resolved
NumpySafeLoader.add_constructor("tag:yaml.org,2002:python/name:numpy.ndarray",
                                lambda loader, node: np.ndarray)
for _m in _MODULES:
    NumpySafeLoader.add_constructor(_APPLY + _m + ".scalar", _scalar)
    NumpySafeLoader.add_constructor(_APPLY + _m + "._reconstruct", _reconstruct)


def load_yaml_safe(stream):
    """``yaml.load(stream, NumpySafeLoader)``: a string or an open file."""
    return yaml.load(stream, Loader=NumpySafeLoader)


def load_yaml_file_safe(path):
    """The parsed YAML file at ``path`` through `NumpySafeLoader`."""
    with open(path) as f:
        return load_yaml_safe(f)
