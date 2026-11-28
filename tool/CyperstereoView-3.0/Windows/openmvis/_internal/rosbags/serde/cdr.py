# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Code generators for CDR.

Common Data Representation `CDR`_ is the serialization format used by most ROS2
middlewares.

.. _CDR: https://www.omg.org/cgi-bin/doc?formal/02-06-51

"""

from __future__ import annotations

import sys
from itertools import pairwise
from typing import TYPE_CHECKING, cast

from rosbags.interfaces import Nodetype

from .utils import SIZEMAP, align, align_after, compile_lines, ndtype

if TYPE_CHECKING:
    from typing import TypeVar

    from rosbags.interfaces import (
        CDRDeser,
        CDRSer,
        CDRSerSize,
        Typestore,
    )
    from rosbags.interfaces.typing import (
        Fielddefs,
        FieldDesc,
    )

    T = TypeVar('T')


def generate_getsize_cdr(fields: Fielddefs, typestore: Typestore) -> tuple[CDRSerSize, int]:
    """Generate cdr size calculation function.

    Args:
        fields: Fields of message.
        typestore: Typestore.

    Returns:
        Size calculation function and static size.

    """
    size = 0
    is_stat = True

    aligned = 8
    lines = [
        'import sys',
        'def getsize_cdr(pos, message, typestore):',
    ]
    for fcurr, fnext in pairwise([*fields, None]):
        fieldname, desc = cast('tuple[str, FieldDesc]', fcurr)

        if desc[0] == Nodetype.NAME:
            msgdef = typestore.get_msgdef(desc[1])

            if msgdef.size_cdr:
                lines.append(f'  pos += {msgdef.size_cdr}')
                size += msgdef.size_cdr
            else:
                lines.append(f'  func = typestore.get_msgdef("{desc[1]}").getsize_cdr')
                lines.append(f'  pos = func(pos, message.{fieldname}, typestore)')
                is_stat = False
            aligned = align_after(desc, typestore)

        elif desc[0] == Nodetype.BASE:
            if desc[1][0] == 'string':
                lines.append(f'  pos += 4 + len(message.{fieldname}.encode()) + 1')
                aligned = 1
                is_stat = False
            else:
                lines.append(f'  pos += {SIZEMAP[desc[1][0]]}')
                aligned = SIZEMAP[desc[1][0]]
                size += SIZEMAP[desc[1][0]]

        elif desc[0] == Nodetype.ARRAY:
            subdesc, length = desc[1]

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append(f'  val = message.{fieldname}')
                    for idx in range(length):
                        lines.append('  pos = (pos + 4 - 1) & -4')
                        lines.append(f'  pos += 4 + len(val[{idx}].encode()) + 1')
                    aligned = 1
                    is_stat = False
                else:
                    lines.append(f'  pos += {length * SIZEMAP[subdesc[1][0]]}')
                    size += length * SIZEMAP[subdesc[1][0]]
                    aligned = SIZEMAP[subdesc[1][0]]

            else:
                assert subdesc[0] == Nodetype.NAME
                msgdef = typestore.get_msgdef(subdesc[1])
                anext_before = align(subdesc, typestore)
                anext_after = align_after(subdesc, typestore)

                if msgdef.size_cdr:
                    for _ in range(length):
                        if anext_before > anext_after:
                            lines.append(f'  pos = (pos + {anext_before} - 1) & -{anext_before}')
                            size = (size + anext_before - 1) & -anext_before
                        lines.append(f'  pos += {msgdef.size_cdr}')
                        size += msgdef.size_cdr
                else:
                    lines.append(
                        f'  func = typestore.get_msgdef("{subdesc[1]}").getsize_cdr',
                    )
                    lines.append(f'  val = message.{fieldname}')
                    for idx in range(length):
                        if anext_before > anext_after:
                            lines.append(f'  pos = (pos + {anext_before} - 1) & -{anext_before}')
                        lines.append(f'  pos = func(pos, val[{idx}], typestore)')
                    is_stat = False
                aligned = align_after(subdesc, typestore)
        else:
            assert desc[0] == Nodetype.SEQUENCE
            lines.append('  pos += 4')
            aligned = 4
            subdesc = desc[1][0]
            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append(f'  for val in message.{fieldname}:')
                    lines.append('    pos = (pos + 4 - 1) & -4')
                    lines.append('    pos += 4 + len(val.encode()) + 1')
                    aligned = 1
                else:
                    anext_before = align(subdesc, typestore)
                    if aligned < anext_before:
                        lines.append(f'  if len(message.{fieldname}):')
                        lines.append(f'    pos = (pos + {anext_before} - 1) & -{anext_before}')
                    lines.append(f'  pos += len(message.{fieldname}) * {SIZEMAP[subdesc[1][0]]}')
                    aligned = anext_before

            else:
                assert subdesc[0] == Nodetype.NAME
                msgdef = typestore.get_msgdef(subdesc[1])
                anext_before = align(subdesc, typestore)
                anext_after = align_after(subdesc, typestore)

                lines.append(f'  val = message.{fieldname}')
                if msgdef.size_cdr:
                    if aligned < anext_before <= anext_after:
                        lines.append('  if len(val):')
                        lines.append(f'    pos = (pos + {anext_before} - 1) & -{anext_before}')
                    lines.append('  for _ in val:')
                    if anext_before > anext_after:
                        lines.append(f'    pos = (pos + {anext_before} - 1) & -{anext_before}')
                    lines.append(f'    pos += {msgdef.size_cdr}')

                else:
                    lines.append(
                        f'  func = typestore.get_msgdef("{subdesc[1]}").getsize_cdr',
                    )
                    if aligned < anext_before <= anext_after:
                        lines.append('  if len(val):')
                        lines.append(f'    pos = (pos + {anext_before} - 1) & -{anext_before}')
                    lines.append('  for item in val:')
                    if anext_before > anext_after:
                        lines.append(f'    pos = (pos + {anext_before} - 1) & -{anext_before}')
                    lines.append('    pos = func(pos, item, typestore)')
                aligned = align_after(subdesc, typestore)

            aligned = min([aligned, 4])
            is_stat = False

        if fnext and aligned < (anext_before := align(fnext[1], typestore)):
            lines.append(f'  pos = (pos + {anext_before} - 1) & -{anext_before}')
            aligned = anext_before
            is_stat = False
    lines.append('  return pos')
    funcname = 'getsize_cdr'
    return cast('CDRSerSize', getattr(compile_lines(lines), funcname)), is_stat * size


def generate_serialize_cdr(fields: Fielddefs, typestore: Typestore, endianness: str) -> CDRSer:
    """Generate cdr serialization function.

    Args:
        fields: Fields of message.
        typestore: Typestore.
        endianness: Endianness of rawdata.

    Returns:
        Serializer function.

    """
    aligned = 8
    lines = [
        'import sys',
        'import numpy',
        'from rosbags.serde import SerdeError',
        f'from rosbags.serde.primitives import pack_bool_{endianness}',
        f'from rosbags.serde.primitives import pack_byte_{endianness}',
        f'from rosbags.serde.primitives import pack_char_{endianness}',
        f'from rosbags.serde.primitives import pack_int8_{endianness}',
        f'from rosbags.serde.primitives import pack_int16_{endianness}',
        f'from rosbags.serde.primitives import pack_int32_{endianness}',
        f'from rosbags.serde.primitives import pack_int64_{endianness}',
        f'from rosbags.serde.primitives import pack_uint8_{endianness}',
        f'from rosbags.serde.primitives import pack_uint16_{endianness}',
        f'from rosbags.serde.primitives import pack_uint32_{endianness}',
        f'from rosbags.serde.primitives import pack_uint64_{endianness}',
        f'from rosbags.serde.primitives import pack_float32_{endianness}',
        f'from rosbags.serde.primitives import pack_float64_{endianness}',
        'def serialize_cdr(rawdata, pos, message, typestore):',
    ]
    for fcurr, fnext in pairwise([*fields, None]):
        fieldname, desc = cast('tuple[str, FieldDesc]', fcurr)

        lines.append(f'  val = message.{fieldname}')
        if desc[0] == Nodetype.NAME:
            lines.append(f'  func = typestore.get_msgdef("{desc[1]}").serialize_cdr_{endianness}')
            lines.append('  pos = func(rawdata, pos, val, typestore)')
            aligned = align_after(desc, typestore)

        elif desc[0] == Nodetype.BASE:
            if desc[1][0] == 'string':
                lines.append('  bval = memoryview(val.encode())')
                lines.append('  length = len(bval) + 1')
                lines.append(f'  pack_int32_{endianness}(rawdata, pos, length)')
                lines.append('  pos += 4')
                lines.append('  rawdata[pos:pos + length - 1] = bval')
                lines.append('  pos += length')
                aligned = 1
            else:
                lines.append(f'  pack_{desc[1][0]}_{endianness}(rawdata, pos, val)')
                lines.append(f'  pos += {SIZEMAP[desc[1][0]]}')
                aligned = SIZEMAP[desc[1][0]]

        elif desc[0] == Nodetype.ARRAY:
            subdesc, length = desc[1]
            lines.append(f'  if len(val) != {length}:')
            lines.append("    raise SerdeError('Unexpected array length')")

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    for idx in range(length):
                        lines.append(f'  bval = memoryview(val[{idx}].encode())')
                        lines.append('  length = len(bval) + 1')
                        lines.append('  pos = (pos + 4 - 1) & -4')
                        lines.append(f'  pack_int32_{endianness}(rawdata, pos, length)')
                        lines.append('  pos += 4')
                        lines.append('  rawdata[pos:pos + length - 1] = bval')
                        lines.append('  pos += length')
                    aligned = 1
                else:
                    if (endianness == 'le') != (sys.byteorder == 'little'):
                        lines.append('  val = val.byteswap()')
                    size = length * SIZEMAP[subdesc[1][0]]
                    lines.append(f'  rawdata[pos:pos + {size}] = val.view(numpy.uint8)')
                    lines.append(f'  pos += {size}')
                    aligned = SIZEMAP[subdesc[1][0]]

            else:
                assert subdesc[0] == Nodetype.NAME
                anext_before = align(subdesc, typestore)
                anext_after = align_after(subdesc, typestore)
                lines.append(
                    f'  func = typestore.get_msgdef("{subdesc[1]}").serialize_cdr_{endianness}',
                )
                for idx in range(length):
                    if anext_before > anext_after:
                        lines.append(f'  pos = (pos + {anext_before} - 1) & -{anext_before}')
                    lines.append(f'  pos = func(rawdata, pos, val[{idx}], typestore)')
                aligned = align_after(subdesc, typestore)
        else:
            assert desc[0] == Nodetype.SEQUENCE
            lines.append(f'  pack_int32_{endianness}(rawdata, pos, len(val))')
            lines.append('  pos += 4')
            aligned = 4
            subdesc = desc[1][0]

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append('  for item in val:')
                    lines.append('    bval = memoryview(item.encode())')
                    lines.append('    length = len(bval) + 1')
                    lines.append('    pos = (pos + 4 - 1) & -4')
                    lines.append(f'    pack_int32_{endianness}(rawdata, pos, length)')
                    lines.append('    pos += 4')
                    lines.append('    rawdata[pos:pos + length - 1] = bval')
                    lines.append('    pos += length')
                    aligned = 1
                else:
                    lines.append(f'  size = len(val) * {SIZEMAP[subdesc[1][0]]}')
                    if (endianness == 'le') != (sys.byteorder == 'little'):
                        lines.append('  val = val.byteswap()')
                    if aligned < (anext_before := align(subdesc, typestore)):
                        lines.append('  if size:')
                        lines.append(f'    pos = (pos + {anext_before} - 1) & -{anext_before}')
                    lines.append('  rawdata[pos:pos + size] = val.view(numpy.uint8)')
                    lines.append('  pos += size')
                    aligned = anext_before

            if subdesc[0] == Nodetype.NAME:
                anext_before = align(subdesc, typestore)
                lines.append(
                    f'  func = typestore.get_msgdef("{subdesc[1]}").serialize_cdr_{endianness}',
                )
                lines.append('  for item in val:')
                lines.append(f'    pos = (pos + {anext_before} - 1) & -{anext_before}')
                lines.append('    pos = func(rawdata, pos, item, typestore)')
                aligned = align_after(subdesc, typestore)

            aligned = min([4, aligned])

        if fnext and aligned < (anext_before := align(fnext[1], typestore)):
            lines.append(f'  pos = (pos + {anext_before} - 1) & -{anext_before}')
            aligned = anext_before
    lines.append('  return pos')
    funcname = 'serialize_cdr'
    return cast('CDRSer', getattr(compile_lines(lines), funcname))


def generate_deserialize_cdr(fields: Fielddefs, typestore: Typestore, endianness: str) -> CDRDeser:
    """Generate cdr deserialization function.

    Args:
        fields: Fields of message.
        typestore: Typestore.
        endianness: Endianness of rawdata.

    Returns:
        Deserializer function.

    """
    aligned = 8
    lines = [
        'import sys',
        'import numpy',
        'from rosbags.serde import SerdeError',
        'from rosbags.serde.utils import check_string, check_sequence',
        f'from rosbags.serde.primitives import unpack_bool_{endianness}',
        f'from rosbags.serde.primitives import unpack_byte_{endianness}',
        f'from rosbags.serde.primitives import unpack_char_{endianness}',
        f'from rosbags.serde.primitives import unpack_int8_{endianness}',
        f'from rosbags.serde.primitives import unpack_int16_{endianness}',
        f'from rosbags.serde.primitives import unpack_int32_{endianness}',
        f'from rosbags.serde.primitives import unpack_int64_{endianness}',
        f'from rosbags.serde.primitives import unpack_uint8_{endianness}',
        f'from rosbags.serde.primitives import unpack_uint16_{endianness}',
        f'from rosbags.serde.primitives import unpack_uint32_{endianness}',
        f'from rosbags.serde.primitives import unpack_uint64_{endianness}',
        f'from rosbags.serde.primitives import unpack_float32_{endianness}',
        f'from rosbags.serde.primitives import unpack_float64_{endianness}',
        'def deserialize_cdr(rawdata, pos, cls, typestore):',
    ]

    funcname = f'deserialize_cdr_{endianness}'
    lines.append('  values = []')
    for fcurr, fnext in pairwise([*fields, None]):
        _, desc = cast('tuple[str, FieldDesc]', fcurr)

        if desc[0] == Nodetype.NAME:
            lines.append(f'  msgdef = typestore.get_msgdef("{desc[1]}")')
            lines.append(f'  obj, pos = msgdef.{funcname}(rawdata, pos, msgdef.cls, typestore)')
            lines.append('  values.append(obj)')
            aligned = align_after(desc, typestore)

        elif desc[0] == Nodetype.BASE:
            if desc[1][0] == 'string':
                lines.append(f'  length = unpack_uint32_{endianness}(rawdata, pos)[0]')
                lines.append('  check_string(rawdata, pos, length, cdr=True)')
                lines.append('  string = bytes(rawdata[pos + 4:pos + 4 + length - 1]).decode()')
                lines.append('  values.append(string)')
                lines.append('  pos += 4 + length')
                aligned = 1
            else:
                lines.append(f'  value = unpack_{desc[1][0]}_{endianness}(rawdata, pos)[0]')
                lines.append('  values.append(value)')
                lines.append(f'  pos += {SIZEMAP[desc[1][0]]}')
                aligned = SIZEMAP[desc[1][0]]

        elif desc[0] == Nodetype.ARRAY:
            subdesc, length = desc[1]
            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append('  value = []')
                    for idx in range(length):
                        if idx:
                            lines.append('  pos = (pos + 4 - 1) & -4')
                        lines.append(f'  length = unpack_uint32_{endianness}(rawdata, pos)[0]')
                        lines.append('  check_string(rawdata, pos, length, cdr=True)')
                        lines.append(
                            '  value.append(bytes(rawdata[pos + 4:pos + 4 + length - 1]).decode())',
                        )
                        lines.append('  pos += 4 + length')
                    lines.append('  values.append(value)')
                    aligned = 1
                else:
                    size = length * SIZEMAP[subdesc[1][0]]
                    lines.append(
                        (
                            f'  val = numpy.frombuffer(rawdata, '
                            f'dtype=numpy.{ndtype(subdesc[1][0])}, count={length}, offset=pos)'
                        ),
                    )
                    if (endianness == 'le') != (sys.byteorder == 'little'):
                        lines.append('  val = val.byteswap()')
                    lines.append('  values.append(val)')
                    lines.append(f'  pos += {size}')
                    aligned = SIZEMAP[subdesc[1][0]]
            else:
                assert subdesc[0] == Nodetype.NAME
                anext_before = align(subdesc, typestore)
                anext_after = align_after(subdesc, typestore)
                lines.append(f'  msgdef = typestore.get_msgdef("{subdesc[1]}")')
                lines.append('  value = []')
                for _ in range(length):
                    if anext_before > anext_after:
                        lines.append(f'  pos = (pos + {anext_before} - 1) & -{anext_before}')
                    lines.append(
                        f'  obj, pos = msgdef.{funcname}(rawdata, pos, msgdef.cls, typestore)',
                    )
                    lines.append('  value.append(obj)')
                lines.append('  values.append(value)')
                aligned = align_after(subdesc, typestore)

        else:
            assert desc[0] == Nodetype.SEQUENCE
            lines.append(f'  size = unpack_uint32_{endianness}(rawdata, pos)[0]')
            lines.append('  pos += 4')
            aligned = 4
            subdesc = desc[1][0]
            minimum = (
                5
                if subdesc[0] == Nodetype.BASE and subdesc[1][0] == 'string'
                else SIZEMAP[subdesc[1][0]]
                if subdesc[0] == Nodetype.BASE
                else 1
            )
            lines.append(
                f'  check_sequence(rawdata, pos, size, {minimum}, typestore.max_sequence_length)'
            )

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append('  value = []')
                    lines.append('  for _ in range(size):')
                    lines.append('    pos = (pos + 4 - 1) & -4')
                    lines.append(f'    length = unpack_uint32_{endianness}(rawdata, pos)[0]')
                    lines.append('    check_string(rawdata, pos, length, cdr=True)')
                    lines.append(
                        '    value.append(bytes(rawdata[pos + 4:pos + 4 + length - 1]).decode())',
                    )
                    lines.append('    pos += 4 + length')
                    lines.append('  values.append(value)')
                    aligned = 1
                else:
                    lines.append(f'  length = size * {SIZEMAP[subdesc[1][0]]}')
                    if aligned < (anext_before := align(subdesc, typestore)):
                        lines.append('  if size:')
                        lines.append(f'    pos = (pos + {anext_before} - 1) & -{anext_before}')
                    lines.append(
                        (
                            f'  val = numpy.frombuffer(rawdata, '
                            f'dtype=numpy.{ndtype(subdesc[1][0])}, count=size, offset=pos)'
                        ),
                    )
                    if (endianness == 'le') != (sys.byteorder == 'little'):
                        lines.append('  val = val.byteswap()')
                    lines.append('  values.append(val)')
                    lines.append('  pos += length')
                    aligned = anext_before

            if subdesc[0] == Nodetype.NAME:
                anext_before = align(subdesc, typestore)
                lines.append(f'  msgdef = typestore.get_msgdef("{subdesc[1]}")')
                lines.append('  value = []')
                lines.append('  for _ in range(size):')
                lines.append(f'    pos = (pos + {anext_before} - 1) & -{anext_before}')
                lines.append(
                    f'    obj, pos = msgdef.{funcname}(rawdata, pos, msgdef.cls, typestore)',
                )
                lines.append('    value.append(obj)')
                lines.append('  values.append(value)')
                aligned = align_after(subdesc, typestore)

            aligned = min([4, aligned])

        if fnext and aligned < (anext_before := align(fnext[1], typestore)):
            lines.append(f'  pos = (pos + {anext_before} - 1) & -{anext_before}')
            aligned = anext_before

    lines.append('  return cls(*values), pos')
    funcname = 'deserialize_cdr'
    return cast('CDRDeser', getattr(compile_lines(lines), funcname))
