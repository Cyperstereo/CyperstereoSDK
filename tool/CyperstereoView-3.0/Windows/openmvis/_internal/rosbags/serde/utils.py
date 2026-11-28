# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Helpers used by code generators."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_loader
from typing import TYPE_CHECKING

from rosbags.interfaces import Nodetype

from .errors import SerdeError

if TYPE_CHECKING:
    from types import ModuleType
    from typing import TypeVar

    from rosbags.interfaces import Typestore
    from rosbags.interfaces.typing import Basename, FieldDesc

    T = TypeVar('T')


SIZEMAP: dict[Basename, int] = {
    'bool': 1,
    'byte': 1,
    'char': 1,
    'int8': 1,
    'int16': 2,
    'int32': 4,
    'int64': 8,
    'uint8': 1,
    'uint16': 2,
    'uint32': 4,
    'uint64': 8,
    'float32': 4,
    'float64': 8,
    'float128': 16,
}


def check_string(rawdata: bytes | memoryview, pos: int, size: int, *, cdr: bool) -> None:
    """Validate a length-prefixed string before decoding or converting it."""
    end = pos + 4 + size
    if size < int(cdr) or pos < 0 or end > len(rawdata):
        msg = f'Invalid string length {size} at offset {pos}.'
        raise SerdeError(msg)
    if cdr and rawdata[end - 1] != 0:
        msg = f'Missing string terminator at offset {end - 1}.'
        raise SerdeError(msg)


def check_sequence(
    rawdata: bytes | memoryview,
    pos: int,
    size: int,
    minimum: int,
    maximum: int,
) -> None:
    """Reject sequence counts that cannot fit in the available bytes."""
    if pos + size * minimum > len(rawdata) or size > maximum:
        msg = f'Invalid sequence length {size} at offset {pos}.'
        raise SerdeError(msg)


def align(entry: FieldDesc, typestore: Typestore) -> int:
    """Get alignment requirement for entry.

    Args:
        entry: Field.
        typestore: Typestore.

    Returns:
        Required alignment in bytes.

    """
    if entry[0] == Nodetype.BASE:
        if entry[1][0] == 'string':
            return 4
        return SIZEMAP[entry[1][0]]
    if entry[0] == Nodetype.NAME:
        return align(typestore.get_msgdef(entry[1]).fields[0][1], typestore)
    if entry[0] == Nodetype.ARRAY:
        return align(entry[1][0], typestore)
    assert entry[0] == Nodetype.SEQUENCE
    return 4


def align_after(entry: FieldDesc, typestore: Typestore) -> int:
    """Get alignment after entry.

    Args:
        entry: Field.
        typestore: Typestore.

    Returns:
        Memory alignment after entry.

    """
    if entry[0] == Nodetype.BASE:
        if entry[1][0] == 'string':
            return 1
        return SIZEMAP[entry[1][0]]
    if entry[0] == Nodetype.NAME:
        return align_after(typestore.get_msgdef(entry[1]).fields[-1][1], typestore)
    if entry[0] == Nodetype.ARRAY:
        return align_after(entry[1][0], typestore)
    assert entry[0] == Nodetype.SEQUENCE
    return min([4, align_after(entry[1][0], typestore)])


def compile_lines(lines: list[str]) -> ModuleType:
    """Compile lines of code to module.

    Args:
        lines: Lines of python code.

    Returns:
        Compiled and loaded module.

    """
    spec = spec_from_loader('tmpmod', loader=None)
    assert spec
    module = module_from_spec(spec)
    exec('\n'.join(lines), module.__dict__)  # noqa: S102
    return module


def ndtype(typ: str) -> str:
    """Normalize numpy dtype."""
    return {'bool': 'bool_', 'char': 'uint8'}.get(typ, typ)
