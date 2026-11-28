# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Code generators for ROS1.

`ROS1`_ uses a serialization format. This module supports fast byte-level
conversion of ROS1 to CDR.

.. _ROS1: http://wiki.ros.org/ROS/Technical%20Overview

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
        Bitcvt,
        BitcvtSize,
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


def generate_ros1_to_cdr(
    fields: Fielddefs,
    typename: str,
    typestore: Typestore,
    *,
    copy: bool,
) -> Bitcvt | BitcvtSize:
    """Generate ROS1 to CDR conversion function.

    Args:
        fields: Fields of message.
        typename: Message type name.
        typestore: Typestore.
        copy: Generate conversion or sizing function.

    Returns:
        ROS1 to CDR conversion function.

    """
    aligned = 8
    funcname = 'ros1_to_cdr' if copy else 'getsize_ros1_to_cdr'
    lines = [
        'import sys',
        'import numpy',
        'from rosbags.serde import SerdeError',
        'from rosbags.serde.utils import check_string, check_sequence',
        'from rosbags.serde.primitives import pack_uint32_le',
        'from rosbags.serde.primitives import unpack_uint32_le',
        f'def {funcname}(input, ipos, output, opos, typestore):',
    ]

    if typename == 'std_msgs/msg/Header':
        lines.append('  ipos += 4')

    for fcurr, fnext in pairwise([*fields, None]):
        fieldname, desc = cast('tuple[str, FieldDesc]', fcurr)

        if fieldname == 'structure_needs_at_least_one_member':
            lines.append('  opos += 1')
            aligned = 1

        elif desc[0] == Nodetype.NAME:
            lines.append(f'  func = typestore.get_msgdef("{desc[1]}").{funcname}')
            lines.append('  ipos, opos = func(input, ipos, output, opos, typestore)')
            aligned = align_after(desc, typestore)

        elif desc[0] == Nodetype.BASE:
            if desc[1][0] == 'string':
                lines.append('  length = unpack_uint32_le(input, ipos)[0] + 1')
                lines.append('  check_string(input, ipos, length - 1, cdr=False)')
                if copy:
                    lines.append('  pack_uint32_le(output, opos, length)')
                lines.append('  ipos += 4')
                lines.append('  opos += 4')
                if copy:
                    lines.append('  output[opos:opos + length - 1] = input[ipos:ipos + length - 1]')
                lines.append('  ipos += length - 1')
                lines.append('  opos += length')
                aligned = 1
            else:
                size = SIZEMAP[desc[1][0]]
                if copy:
                    lines.append(f'  output[opos:opos + {size}] = input[ipos:ipos + {size}]')
                lines.append(f'  ipos += {size}')
                lines.append(f'  opos += {size}')
                aligned = size

        elif desc[0] == Nodetype.ARRAY:
            subdesc, length = desc[1]

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    for _ in range(length):
                        lines.append('  opos = (opos + 4 - 1) & -4')
                        lines.append('  length = unpack_uint32_le(input, ipos)[0] + 1')
                        lines.append('  check_string(input, ipos, length - 1, cdr=False)')
                        if copy:
                            lines.append('  pack_uint32_le(output, opos, length)')
                        lines.append('  ipos += 4')
                        lines.append('  opos += 4')
                        if copy:
                            lines.append(
                                '  output[opos:opos + length - 1] = input[ipos:ipos + length - 1]',
                            )
                        lines.append('  ipos += length - 1')
                        lines.append('  opos += length')
                    aligned = 1
                else:
                    size = length * SIZEMAP[subdesc[1][0]]
                    if copy:
                        lines.append(f'  output[opos:opos + {size}] = input[ipos:ipos + {size}]')
                    lines.append(f'  ipos += {size}')
                    lines.append(f'  opos += {size}')
                    aligned = SIZEMAP[subdesc[1][0]]

            if subdesc[0] == Nodetype.NAME:
                anext_before = align(subdesc, typestore)
                anext_after = align_after(subdesc, typestore)

                lines.append(f'  func = typestore.get_msgdef("{subdesc[1]}").{funcname}')
                for _ in range(length):
                    if anext_before > anext_after:
                        lines.append(f'  opos = (opos + {anext_before} - 1) & -{anext_before}')
                    lines.append('  ipos, opos = func(input, ipos, output, opos, typestore)')
                aligned = anext_after
        else:
            assert desc[0] == Nodetype.SEQUENCE
            lines.append('  size = unpack_uint32_le(input, ipos)[0]')
            if copy:
                lines.append('  pack_uint32_le(output, opos, size)')
            lines.append('  ipos += 4')
            lines.append('  opos += 4')
            subdesc = desc[1][0]
            minimum = (
                4
                if subdesc[0] == Nodetype.BASE and subdesc[1][0] == 'string'
                else SIZEMAP[subdesc[1][0]]
                if subdesc[0] == Nodetype.BASE
                else 0
            )
            lines.append(
                f'  check_sequence(input, ipos, size, {minimum}, typestore.max_sequence_length)'
            )
            aligned = 4

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append('  for _ in range(size):')
                    lines.append('    length = unpack_uint32_le(input, ipos)[0] + 1')
                    lines.append('    check_string(input, ipos, length - 1, cdr=False)')
                    lines.append('    opos = (opos + 4 - 1) & -4')
                    if copy:
                        lines.append('    pack_uint32_le(output, opos, length)')
                    lines.append('    ipos += 4')
                    lines.append('    opos += 4')
                    if copy:
                        lines.append(
                            '    output[opos:opos + length - 1] = input[ipos:ipos + length - 1]',
                        )
                    lines.append('    ipos += length - 1')
                    lines.append('    opos += length')
                    aligned = 1
                else:
                    if aligned < (anext_before := align(subdesc, typestore)):
                        lines.append('  if size:')
                        lines.append(f'    opos = (opos + {anext_before} - 1) & -{anext_before}')
                    lines.append(f'  length = size * {SIZEMAP[subdesc[1][0]]}')
                    if copy:
                        lines.append('  output[opos:opos + length] = input[ipos:ipos + length]')
                    lines.append('  ipos += length')
                    lines.append('  opos += length')
                    aligned = anext_before

            else:
                assert subdesc[0] == Nodetype.NAME
                anext_before = align(subdesc, typestore)
                lines.append(f'  func = typestore.get_msgdef("{subdesc[1]}").{funcname}')
                lines.append('  for _ in range(size):')
                lines.append(f'    opos = (opos + {anext_before} - 1) & -{anext_before}')
                lines.append('    ipos, opos = func(input, ipos, output, opos, typestore)')
                aligned = align_after(subdesc, typestore)

            aligned = min([aligned, 4])

        if fnext and aligned < (anext_before := align(fnext[1], typestore)):
            lines.append(f'  opos = (opos + {anext_before} - 1) & -{anext_before}')
            aligned = anext_before

    lines.append('  return ipos, opos')
    return cast('Bitcvt | BitcvtSize', getattr(compile_lines(lines), funcname))


def generate_cdr_to_ros1(
    fields: Fielddefs,
    typename: str,
    typestore: Typestore,
    *,
    copy: bool,
) -> Bitcvt | BitcvtSize:
    """Generate CDR to ROS1 conversion function.

    Args:
        fields: Fields of message.
        typename: Message type name.
        typestore: Typestore.
        copy: Generate conversion or sizing function.

    Returns:
        CDR to ROS1 conversion function.

    """
    aligned = 8
    funcname = 'cdr_to_ros1' if copy else 'getsize_cdr_to_ros1'
    lines = [
        'import sys',
        'import numpy',
        'from rosbags.serde import SerdeError',
        'from rosbags.serde.utils import check_string, check_sequence',
        'from rosbags.serde.primitives import pack_uint32_le',
        'from rosbags.serde.primitives import unpack_uint32_le',
        f'def {funcname}(input, ipos, output, opos, typestore):',
    ]

    if typename == 'std_msgs/msg/Header':
        lines.append('  opos += 4')

    for fcurr, fnext in pairwise([*fields, None]):
        fieldname, desc = cast('tuple[str, FieldDesc]', fcurr)

        if fieldname == 'structure_needs_at_least_one_member':
            lines.append('  ipos += 1')
            aligned = 1

        elif desc[0] == Nodetype.NAME:
            lines.append(f'  func = typestore.get_msgdef("{desc[1]}").{funcname}')
            lines.append('  ipos, opos = func(input, ipos, output, opos, typestore)')
            aligned = align_after(desc, typestore)

        elif desc[0] == Nodetype.BASE:
            if desc[1][0] == 'string':
                lines.append('  length = unpack_uint32_le(input, ipos)[0] - 1')
                lines.append('  check_string(input, ipos, length + 1, cdr=True)')
                if copy:
                    lines.append('  pack_uint32_le(output, opos, length)')
                lines.append('  ipos += 4')
                lines.append('  opos += 4')
                if copy:
                    lines.append('  output[opos:opos + length] = input[ipos:ipos + length]')
                lines.append('  ipos += length + 1')
                lines.append('  opos += length')
                aligned = 1
            else:
                size = SIZEMAP[desc[1][0]]
                if copy:
                    lines.append(f'  output[opos:opos + {size}] = input[ipos:ipos + {size}]')
                lines.append(f'  ipos += {size}')
                lines.append(f'  opos += {size}')
                aligned = size

        elif desc[0] == Nodetype.ARRAY:
            subdesc, length = desc[1]

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    for _ in range(length):
                        lines.append('  ipos = (ipos + 4 - 1) & -4')
                        lines.append('  length = unpack_uint32_le(input, ipos)[0] - 1')
                        lines.append('  check_string(input, ipos, length + 1, cdr=True)')
                        if copy:
                            lines.append('  pack_uint32_le(output, opos, length)')
                        lines.append('  ipos += 4')
                        lines.append('  opos += 4')
                        if copy:
                            lines.append('  output[opos:opos + length] = input[ipos:ipos + length]')
                        lines.append('  ipos += length + 1')
                        lines.append('  opos += length')
                    aligned = 1
                else:
                    size = length * SIZEMAP[subdesc[1][0]]
                    if copy:
                        lines.append(f'  output[opos:opos + {size}] = input[ipos:ipos + {size}]')
                    lines.append(f'  ipos += {size}')
                    lines.append(f'  opos += {size}')
                    aligned = SIZEMAP[subdesc[1][0]]

            if subdesc[0] == Nodetype.NAME:
                anext_before = align(subdesc, typestore)
                anext_after = align_after(subdesc, typestore)

                lines.append(f'  func = typestore.get_msgdef("{subdesc[1]}").{funcname}')
                for _ in range(length):
                    if anext_before > anext_after:
                        lines.append(f'  ipos = (ipos + {anext_before} - 1) & -{anext_before}')
                    lines.append('  ipos, opos = func(input, ipos, output, opos, typestore)')
                aligned = anext_after
        else:
            assert desc[0] == Nodetype.SEQUENCE
            lines.append('  size = unpack_uint32_le(input, ipos)[0]')
            if copy:
                lines.append('  pack_uint32_le(output, opos, size)')
            lines.append('  ipos += 4')
            lines.append('  opos += 4')
            subdesc = desc[1][0]
            minimum = (
                5
                if subdesc[0] == Nodetype.BASE and subdesc[1][0] == 'string'
                else SIZEMAP[subdesc[1][0]]
                if subdesc[0] == Nodetype.BASE
                else 1
            )
            lines.append(
                f'  check_sequence(input, ipos, size, {minimum}, typestore.max_sequence_length)'
            )
            aligned = 4

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append('  for _ in range(size):')
                    lines.append('    ipos = (ipos + 4 - 1) & -4')
                    lines.append('    length = unpack_uint32_le(input, ipos)[0] - 1')
                    lines.append('    check_string(input, ipos, length + 1, cdr=True)')
                    if copy:
                        lines.append('    pack_uint32_le(output, opos, length)')
                    lines.append('    ipos += 4')
                    lines.append('    opos += 4')
                    if copy:
                        lines.append('    output[opos:opos + length] = input[ipos:ipos + length]')
                    lines.append('    ipos += length + 1')
                    lines.append('    opos += length')
                    aligned = 1
                else:
                    if aligned < (anext_before := align(subdesc, typestore)):
                        lines.append('  if size:')
                        lines.append(f'    ipos = (ipos + {anext_before} - 1) & -{anext_before}')
                    lines.append(f'  length = size * {SIZEMAP[subdesc[1][0]]}')
                    if copy:
                        lines.append('  output[opos:opos + length] = input[ipos:ipos + length]')
                    lines.append('  ipos += length')
                    lines.append('  opos += length')
                    aligned = anext_before

            else:
                assert subdesc[0] == Nodetype.NAME
                anext_before = align(subdesc, typestore)
                lines.append(f'  func = typestore.get_msgdef("{subdesc[1]}").{funcname}')
                lines.append('  for _ in range(size):')
                lines.append(f'    ipos = (ipos + {anext_before} - 1) & -{anext_before}')
                lines.append('    ipos, opos = func(input, ipos, output, opos, typestore)')
                aligned = align_after(subdesc, typestore)

            aligned = min([aligned, 4])

        if fnext and aligned < (anext_before := align(fnext[1], typestore)):
            lines.append(f'  ipos = (ipos + {anext_before} - 1) & -{anext_before}')
            aligned = anext_before

    lines.append('  return ipos, opos')
    return cast('Bitcvt | BitcvtSize', getattr(compile_lines(lines), funcname))


def generate_getsize_ros1(fields: Fielddefs, typestore: Typestore) -> tuple[CDRSerSize, int]:
    """Generate ros1 size calculation function.

    Args:
        fields: Fields of message.
        typestore: Typestore.

    Returns:
        Size calculation function and static size.

    """
    size = 0
    is_stat = True

    lines = [
        'import sys',
        'def getsize_ros1(pos, message, typestore):',
    ]

    for fcurr in fields:
        fieldname, desc = fcurr

        if fieldname == 'structure_needs_at_least_one_member':
            continue

        if desc[0] == Nodetype.NAME:
            msgdef = typestore.get_msgdef(desc[1])
            if msgdef.size_ros1:
                lines.append(f'  pos += {msgdef.size_ros1}')
                size += msgdef.size_ros1
            else:
                lines.append(f'  func = typestore.get_msgdef("{desc[1]}").getsize_ros1')
                lines.append(f'  pos = func(pos, message.{fieldname}, typestore)')
                is_stat = False

        elif desc[0] == Nodetype.BASE:
            if desc[1][0] == 'string':
                lines.append(f'  pos += 4 + len(message.{fieldname}.encode())')
                is_stat = False
            else:
                lines.append(f'  pos += {SIZEMAP[desc[1][0]]}')
                size += SIZEMAP[desc[1][0]]

        elif desc[0] == Nodetype.ARRAY:
            subdesc, length = desc[1]

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append(f'  val = message.{fieldname}')
                    lines.extend(f'  pos += 4 + len(val[{idx}].encode())' for idx in range(length))
                    is_stat = False
                else:
                    lines.append(f'  pos += {length * SIZEMAP[subdesc[1][0]]}')
                    size += length * SIZEMAP[subdesc[1][0]]

            else:
                assert subdesc[0] == Nodetype.NAME
                msgdef = typestore.get_msgdef(subdesc[1])
                if msgdef.size_ros1:
                    for _ in range(length):
                        lines.append(f'  pos += {msgdef.size_ros1}')
                        size += msgdef.size_ros1
                else:
                    lines.append(
                        f'  func = typestore.get_msgdef("{subdesc[1]}").getsize_ros1',
                    )
                    lines.append(f'  val = message.{fieldname}')
                    lines.extend(
                        f'  pos = func(pos, val[{idx}], typestore)' for idx in range(length)
                    )
                    is_stat = False
        else:
            assert desc[0] == Nodetype.SEQUENCE
            lines.append('  pos += 4')
            subdesc = desc[1][0]
            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append(f'  for val in message.{fieldname}:')
                    lines.append('    pos += 4 + len(val.encode())')
                else:
                    lines.append(f'  pos += len(message.{fieldname}) * {SIZEMAP[subdesc[1][0]]}')

            else:
                assert subdesc[0] == Nodetype.NAME
                msgdef = typestore.get_msgdef(subdesc[1])
                lines.append(f'  val = message.{fieldname}')
                if msgdef.size_ros1:
                    lines.append(f'  pos += {msgdef.size_ros1} * len(val)')

                else:
                    lines.append(
                        f'  func = typestore.get_msgdef("{msgdef.name}").getsize_ros1',
                    )
                    lines.append('  for item in val:')
                    lines.append('    pos = func(pos, item, typestore)')

            is_stat = False
    lines.append('  return pos')
    funcname = 'getsize_ros1'
    return cast('CDRSerSize', getattr(compile_lines(lines), funcname)), is_stat * size


def generate_serialize_ros1(fields: Fielddefs, typestore: Typestore) -> CDRSer:
    """Generate ros1 serialization function.

    Args:
        fields: Fields of message.
        typestore: Typestore.

    Returns:
        Serializer function.

    """
    _ = typestore
    lines = [
        'import sys',
        'import numpy',
        'from rosbags.serde import SerdeError',
        'from rosbags.serde.utils import check_string, check_sequence',
        'from rosbags.serde.primitives import pack_bool_le',
        'from rosbags.serde.primitives import pack_byte_le',
        'from rosbags.serde.primitives import pack_char_le',
        'from rosbags.serde.primitives import pack_int8_le',
        'from rosbags.serde.primitives import pack_int16_le',
        'from rosbags.serde.primitives import pack_int32_le',
        'from rosbags.serde.primitives import pack_int64_le',
        'from rosbags.serde.primitives import pack_uint8_le',
        'from rosbags.serde.primitives import pack_uint16_le',
        'from rosbags.serde.primitives import pack_uint32_le',
        'from rosbags.serde.primitives import pack_uint64_le',
        'from rosbags.serde.primitives import pack_float32_le',
        'from rosbags.serde.primitives import pack_float64_le',
        'def serialize_ros1(rawdata, pos, message, typestore):',
    ]

    be_syms = ('>',) if sys.byteorder == 'little' else ('=', '>')

    for fcurr in fields:
        fieldname, desc = fcurr

        if fieldname == 'structure_needs_at_least_one_member':
            continue

        lines.append(f'  val = message.{fieldname}')
        if desc[0] == Nodetype.NAME:
            lines.append(f'  func = typestore.get_msgdef("{desc[1]}").serialize_ros1')
            lines.append('  pos = func(rawdata, pos, val, typestore)')

        elif desc[0] == Nodetype.BASE:
            if desc[1][0] == 'string':
                lines.append('  bval = memoryview(val.encode())')
                lines.append('  length = len(bval)')
                lines.append('  pack_uint32_le(rawdata, pos, length)')
                lines.append('  pos += 4')
                lines.append('  rawdata[pos:pos + length] = bval')
                lines.append('  pos += length')
            else:
                lines.append(f'  pack_{desc[1][0]}_le(rawdata, pos, val)')
                lines.append(f'  pos += {SIZEMAP[desc[1][0]]}')

        elif desc[0] == Nodetype.ARRAY:
            subdesc, length = desc[1]
            lines.append(f'  if len(val) != {length}:')
            lines.append("    raise SerdeError('Unexpected array length')")

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    for idx in range(length):
                        lines.append(f'  bval = memoryview(val[{idx}].encode())')
                        lines.append('  length = len(bval)')
                        lines.append('  pack_uint32_le(rawdata, pos, length)')
                        lines.append('  pos += 4')
                        lines.append('  rawdata[pos:pos + length] = bval')
                        lines.append('  pos += length')
                else:
                    lines.append(f'  if val.dtype.byteorder in {be_syms}:')
                    lines.append('    val = val.byteswap()')
                    size = length * SIZEMAP[subdesc[1][0]]
                    lines.append(f'  rawdata[pos:pos + {size}] = val.view(numpy.uint8)')
                    lines.append(f'  pos += {size}')

            else:
                assert subdesc[0] == Nodetype.NAME
                lines.append(f'  func = typestore.get_msgdef("{subdesc[1]}").serialize_ros1')
                lines.extend(
                    f'  pos = func(rawdata, pos, val[{idx}], typestore)' for idx in range(length)
                )
        else:
            assert desc[0] == Nodetype.SEQUENCE
            lines.append('  pack_uint32_le(rawdata, pos, len(val))')
            lines.append('  pos += 4')
            subdesc = desc[1][0]

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append('  for item in val:')
                    lines.append('    bval = memoryview(item.encode())')
                    lines.append('    length = len(bval)')
                    lines.append('    pack_uint32_le(rawdata, pos, length)')
                    lines.append('    pos += 4')
                    lines.append('    rawdata[pos:pos + length] = bval')
                    lines.append('    pos += length')
                else:
                    lines.append(f'  size = len(val) * {SIZEMAP[subdesc[1][0]]}')
                    lines.append(f'  if val.dtype.byteorder in {be_syms}:')
                    lines.append('    val = val.byteswap()')
                    lines.append('  rawdata[pos:pos + size] = val.view(numpy.uint8)')
                    lines.append('  pos += size')

            if subdesc[0] == Nodetype.NAME:
                lines.append(f'  func = typestore.get_msgdef("{subdesc[1]}").serialize_ros1')
                lines.append('  for item in val:')
                lines.append('    pos = func(rawdata, pos, item, typestore)')

    lines.append('  return pos')
    funcname = 'serialize_ros1'
    return cast('CDRSer', getattr(compile_lines(lines), funcname))


def generate_deserialize_ros1(fields: Fielddefs, typestore: Typestore) -> CDRDeser:
    """Generate ros1 deserialization function.

    Args:
        fields: Fields of message.
        typestore: Typestore.

    Returns:
        Deserializer function.

    """
    _ = typestore
    lines = [
        'import sys',
        'import numpy',
        'from rosbags.serde import SerdeError',
        'from rosbags.serde.utils import check_string, check_sequence',
        'from rosbags.serde.primitives import unpack_bool_le',
        'from rosbags.serde.primitives import unpack_byte_le',
        'from rosbags.serde.primitives import unpack_char_le',
        'from rosbags.serde.primitives import unpack_int8_le',
        'from rosbags.serde.primitives import unpack_int16_le',
        'from rosbags.serde.primitives import unpack_int32_le',
        'from rosbags.serde.primitives import unpack_uint32_le',
        'from rosbags.serde.primitives import unpack_int64_le',
        'from rosbags.serde.primitives import unpack_uint8_le',
        'from rosbags.serde.primitives import unpack_uint16_le',
        'from rosbags.serde.primitives import unpack_uint64_le',
        'from rosbags.serde.primitives import unpack_float32_le',
        'from rosbags.serde.primitives import unpack_float64_le',
        'def deserialize_ros1(rawdata, pos, cls, typestore):',
    ]

    be_syms = ('>',) if sys.byteorder == 'little' else ('=', '>')

    funcname = 'deserialize_ros1'
    lines.append('  values = []')
    for fcurr in fields:
        fieldname, desc = fcurr

        if fieldname == 'structure_needs_at_least_one_member':
            continue

        if desc[0] == Nodetype.NAME:
            lines.append(f'  msgdef = typestore.get_msgdef("{desc[1]}")')
            lines.append(f'  obj, pos = msgdef.{funcname}(rawdata, pos, msgdef.cls, typestore)')
            lines.append('  values.append(obj)')

        elif desc[0] == Nodetype.BASE:
            if desc[1][0] == 'string':
                lines.append('  length = unpack_uint32_le(rawdata, pos)[0]')
                lines.append('  check_string(rawdata, pos, length, cdr=False)')
                lines.append('  string = bytes(rawdata[pos + 4:pos + 4 + length]).decode()')
                lines.append('  values.append(string)')
                lines.append('  pos += 4 + length')
            else:
                lines.append(f'  value = unpack_{desc[1][0]}_le(rawdata, pos)[0]')
                lines.append('  values.append(value)')
                lines.append(f'  pos += {SIZEMAP[desc[1][0]]}')

        elif desc[0] == Nodetype.ARRAY:
            subdesc, length = desc[1]
            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append('  value = []')
                    for _ in range(length):
                        lines.append('  length = unpack_uint32_le(rawdata, pos)[0]')
                        lines.append('  check_string(rawdata, pos, length, cdr=False)')
                        lines.append(
                            '  value.append(bytes(rawdata[pos + 4:pos + 4 + length]).decode())',
                        )
                        lines.append('  pos += 4 + length')
                    lines.append('  values.append(value)')
                else:
                    size = length * SIZEMAP[subdesc[1][0]]
                    lines.append(
                        (
                            f'  val = numpy.frombuffer(rawdata, '
                            f'dtype=numpy.{ndtype(subdesc[1][0])}, count={length}, offset=pos)'
                        ),
                    )
                    lines.append(f'  if val.dtype.byteorder in {be_syms}:')
                    lines.append('    val = val.byteswap()')
                    lines.append('  values.append(val)')
                    lines.append(f'  pos += {size}')
            else:
                assert subdesc[0] == Nodetype.NAME
                lines.append(f'  msgdef = typestore.get_msgdef("{subdesc[1]}")')
                lines.append('  value = []')
                for _ in range(length):
                    lines.append(
                        f'  obj, pos = msgdef.{funcname}(rawdata, pos, msgdef.cls, typestore)',
                    )
                    lines.append('  value.append(obj)')
                lines.append('  values.append(value)')

        else:
            assert desc[0] == Nodetype.SEQUENCE
            lines.append('  size = unpack_uint32_le(rawdata, pos)[0]')
            lines.append('  pos += 4')
            subdesc = desc[1][0]
            minimum = (
                4
                if subdesc[0] == Nodetype.BASE and subdesc[1][0] == 'string'
                else SIZEMAP[subdesc[1][0]]
                if subdesc[0] == Nodetype.BASE
                else 0
            )
            lines.append(
                f'  check_sequence(rawdata, pos, size, {minimum}, typestore.max_sequence_length)'
            )

            if subdesc[0] == Nodetype.BASE:
                if subdesc[1][0] == 'string':
                    lines.append('  value = []')
                    lines.append('  for _ in range(size):')
                    lines.append('    length = unpack_uint32_le(rawdata, pos)[0]')
                    lines.append('    check_string(rawdata, pos, length, cdr=False)')
                    lines.append(
                        '    value.append(bytes(rawdata[pos + 4:pos + 4 + length]).decode())',
                    )
                    lines.append('    pos += 4 + length')
                    lines.append('  values.append(value)')
                else:
                    lines.append(f'  length = size * {SIZEMAP[subdesc[1][0]]}')
                    lines.append(
                        (
                            f'  val = numpy.frombuffer(rawdata, '
                            f'dtype=numpy.{ndtype(subdesc[1][0])}, count=size, offset=pos)'
                        ),
                    )
                    lines.append(f'  if val.dtype.byteorder in {be_syms}:')
                    lines.append('    val = val.byteswap()')
                    lines.append('  values.append(val)')
                    lines.append('  pos += length')

            if subdesc[0] == Nodetype.NAME:
                lines.append(f'  msgdef = typestore.get_msgdef("{subdesc[1]}")')
                lines.append('  value = []')
                lines.append('  for _ in range(size):')
                lines.append(
                    f'    obj, pos = msgdef.{funcname}(rawdata, pos, msgdef.cls, typestore)',
                )
                lines.append('    value.append(obj)')
                lines.append('  values.append(value)')

    lines.append('  return cls(*values), pos')
    funcname = 'deserialize_ros1'
    return cast('CDRDeser', getattr(compile_lines(lines), funcname))
