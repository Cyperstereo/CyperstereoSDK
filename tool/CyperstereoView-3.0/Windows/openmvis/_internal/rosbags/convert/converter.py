# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Rosbag1 to Rosbag2 Converter."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, cast

import numpy as np

from rosbags.highlevel.anyreader import AnyReader, AnyReaderError
from rosbags.interfaces import (
    Connection,
    ConnectionExtRosbag1,
    ConnectionExtRosbag2,
    Nodetype,
    Qos,
    QosDurability,
    QosHistory,
    QosLiveliness,
    QosReliability,
    QosTime,
)
from rosbags.rosbag1 import (
    Writer as Writer1,
    WriterError as WriterError1,
)
from rosbags.rosbag2 import (
    StoragePlugin,
    Writer as Writer2,
    WriterError as WriterError2,
)
from rosbags.rosbag2.enums import CompressionFormat, CompressionMode
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.typesys.base import normalize_fieldname

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from typing import Literal, TypeAlias

    from rosbags.interfaces.typing import FieldDesc
    from rosbags.typesys.store import Msg, Msgarg, Typestore

    MigrationKey: TypeAlias = tuple[Typestore, Typestore, str, str]
    MigrationPlan: TypeAlias = list[tuple[str, FieldDesc, str | None, FieldDesc | None]]


LATCH = [
    Qos(
        QosHistory.UNKNOWN,
        0,
        QosReliability.RELIABLE,
        QosDurability.TRANSIENT_LOCAL,
        QosTime(2147483647, 4294967295),
        QosTime(2147483647, 4294967295),
        QosLiveliness.AUTOMATIC,
        QosTime(2147483647, 4294967295),
        avoid_ros_namespace_conventions=False,
    )
]

# Legacy message types that will always be renamed
STATIC_MSGTYPE_RENAMES = {
    'tf/msg/tfMessage': 'tf2_msgs/msg/TFMessage',
}


class ConverterError(Exception):
    """Converter Error."""


def echo(text: str) -> None:
    """Print text."""
    print(text)  # noqa: T201


def is_same_wireformat(
    src_typestore: Typestore,
    dst_typestore: Typestore,
    src_msgtype: str,
    dst_msgtype: str,
) -> bool:
    """Check if wireformat of messages is identical."""
    if src_msgtype == 'std_msgs/msg/Header':
        return True
    src_fields = src_typestore.fielddefs[src_msgtype][1]
    dst_fields = dst_typestore.fielddefs[dst_msgtype][1]

    if len(src_fields) != len(dst_fields):
        return False

    for (_, src), (_, dst) in zip(src_fields, dst_fields, strict=True):
        if src != dst:
            return False

        sub_src = src[1][0] if src[0] == Nodetype.SEQUENCE or src[0] == Nodetype.ARRAY else src
        sub_dst = dst[1][0] if dst[0] == Nodetype.SEQUENCE or dst[0] == Nodetype.ARRAY else dst
        if sub_src[0] == Nodetype.NAME:
            assert isinstance(sub_dst[1], str)
            if not is_same_wireformat(src_typestore, dst_typestore, sub_src[1], sub_dst[1]):
                return False

    return True


def default_message(
    typestore: Typestore,
    msgtype: str,
) -> object:
    """Create default message."""
    values: list[Msgarg] = []

    for _, typ in typestore.fielddefs[msgtype][1]:
        if typ[0] == Nodetype.BASE:
            values.append('' if typ[1][0] == 'string' else 0)
        elif typ[0] == Nodetype.NAME:
            values.append(cast('Msg', default_message(typestore, typ[1])))
        else:
            assert typ[0] in {Nodetype.SEQUENCE, Nodetype.ARRAY}
            subtyp = typ[1][0]
            size = 0 if typ[0] == Nodetype.SEQUENCE else typ[1][1]
            if subtyp[0] == Nodetype.BASE:
                if subtyp[1][0] == 'string':
                    values.append([''] * size)
                else:
                    dtype = 'uint8' if subtyp[1][0] == 'char' else subtyp[1][0]
                    values.append(np.zeros(size, dtype=np.dtype(dtype)))
            else:
                values.append(
                    [cast('Msg', default_message(typestore, subtyp[1])) for _ in range(size)],
                )

    return typestore.types[msgtype](*values)


def _create_migration_plan(
    src_def: list[tuple[str, FieldDesc]],
    dst_def: list[tuple[str, FieldDesc]],
) -> MigrationPlan:
    """Create migration plan for src dst pair."""
    src_map = dict(src_def)
    dst_map = dict(dst_def)

    dst_src: dict[str, str | None] = {x: x for x in dst_map if x in src_map}

    removed = [x[0] for x in src_def if x[0] not in dst_map]
    added = [x[0] for x in dst_def if x[0] not in src_map]
    for dst_name in added:
        for src_name in removed:
            if src_map[src_name] == dst_map[dst_name]:
                dst_src[dst_name] = src_name
                removed.remove(src_name)
                break
        else:
            dst_src[dst_name] = None

    return [
        (name, typ, dst_src[name], src_map.get(dst_src[name] or ''))
        for name, typ in dst_map.items()
    ]


def migrate_message(
    src_typestore: Typestore,
    dst_typestore: Typestore,
    src_msgtype: str,
    dst_msgtype: str,
    cache: dict[str, object],
    src_msg: object,
) -> object:
    """Migrate message."""
    values: list[Msgarg] = []

    plans = cast(
        'dict[MigrationKey, MigrationPlan]',
        cache.setdefault('__migration_plans__', {}),
    )
    key = (src_typestore, dst_typestore, src_msgtype, dst_msgtype)
    if key not in plans:
        src_def = src_typestore.fielddefs[src_msgtype][1]
        dst_def = dst_typestore.fielddefs[dst_msgtype][1]
        plans[key] = _create_migration_plan(src_def, dst_def)

    if dst_msgtype not in cache or not isinstance(
        cache[dst_msgtype],
        dst_typestore.types[dst_msgtype],
    ):
        cache[dst_msgtype] = default_message(dst_typestore, dst_msgtype)
    def_msg = cache[dst_msgtype]

    for dst_field, dst_type, src_field, src_type in plans[key]:
        dst_name = normalize_fieldname(dst_field)
        src_name = normalize_fieldname(src_field) if src_field is not None else None
        if src_name is None:
            values.append(cast('Msgarg', deepcopy(getattr(def_msg, dst_name))))
            continue

        assert src_type is not None

        if dst_type[0] == Nodetype.BASE:
            if src_type[0] == Nodetype.BASE and (dst_type[1][0] == 'string') == (
                src_type[1][0] == 'string'
            ):
                values.append(cast('Msgarg', getattr(src_msg, src_name)))
            else:
                values.append(cast('Msgarg', deepcopy(getattr(def_msg, dst_name))))
        elif dst_type[0] == Nodetype.NAME:
            if src_type[0] == Nodetype.NAME:
                values.append(
                    cast(
                        'Msg',
                        migrate_message(
                            src_typestore,
                            dst_typestore,
                            src_type[1],
                            dst_type[1],
                            cache,
                            cast('Msgarg', getattr(src_msg, src_name)),
                        ),
                    ),
                )
            else:
                values.append(cast('Msgarg', deepcopy(getattr(def_msg, dst_name))))
        else:
            assert dst_type[0] in {Nodetype.SEQUENCE, Nodetype.ARRAY}
            if src_type[0] not in {Nodetype.SEQUENCE, Nodetype.ARRAY}:
                values.append(cast('Msgarg', deepcopy(getattr(def_msg, dst_name))))
                continue
            src_sub = src_type[1][0]
            dst_sub = dst_type[1][0]
            if src_sub[0] != dst_sub[0] or (
                dst_sub[0] == Nodetype.BASE
                and (src_sub[1][0] == 'string') != (dst_sub[1][0] == 'string')
            ):
                values.append(cast('Msgarg', deepcopy(getattr(def_msg, dst_name))))
            else:
                size = dst_type[1][1]
                src_value = cast(
                    'list[str] | list[Msg] | np.ndarray[tuple[int, ...], np.dtype[np.uint8]]',
                    getattr(src_msg, src_name),
                )
                if size and size < len(src_value):
                    src_value = src_value[:size]
                if dst_sub[0] == Nodetype.BASE and dst_sub[1][0] != 'string':
                    dtype = 'uint8' if dst_sub[1][0] == 'char' else dst_sub[1][0]
                    assert not isinstance(src_value, list)
                    src_value = src_value.astype(dtype)
                elif dst_sub[0] == Nodetype.NAME:
                    assert isinstance(src_sub[1], str)
                    src_value = [
                        cast(
                            'Msg',
                            migrate_message(
                                src_typestore,
                                dst_typestore,
                                src_sub[1],
                                dst_sub[1],
                                cache,
                                x,
                            ),
                        )
                        for x in src_value
                    ]
                if dst_type[0] == Nodetype.ARRAY and len(src_value) != size:
                    oldsize = len(src_value)
                    if isinstance(src_value, np.ndarray):
                        src_value = np.resize(src_value, size)
                        src_value[oldsize:] = 0
                    else:
                        defaults = cast('list[str]', getattr(def_msg, dst_name))
                        src_value = cast(
                            'list[str] | list[Msg]',
                            [*cast('list[str]', src_value), *deepcopy(defaults[oldsize:])],
                        )
                elif isinstance(src_value, list):
                    src_value = src_value.copy()
                values.append(src_value)

    return dst_typestore.types[dst_msgtype](*values)


def migrate_bytes(
    src_typestore: Typestore,
    dst_typestore: Typestore,
    src_msgtype: str,
    dst_msgtype: str,
    cache: dict[str, object],
    data: bytes | memoryview,
    *,
    src_is2: bool,
    dst_is2: bool,
) -> memoryview:
    """Migrate message."""
    src_msg = (
        src_typestore.deserialize_cdr(data, src_msgtype)
        if src_is2
        else src_typestore.deserialize_ros1(data, src_msgtype)
    )

    dst_msg = migrate_message(
        src_typestore,
        dst_typestore,
        src_msgtype,
        dst_msgtype,
        cache,
        src_msg,
    )

    return (
        dst_typestore.serialize_cdr(dst_msg, dst_msgtype, little_endian=True)
        if dst_is2
        else dst_typestore.serialize_ros1(dst_msg, dst_msgtype)
    )


def generate_message_converter(
    src_typestore: Typestore,
    dst_typestore: Typestore,
    src_msgtype: str,
    dst_msgtype: str,
    cache: dict[str, object],
    *,
    src_is2: bool,
    dst_is2: bool,
) -> Callable[[bytes | memoryview], memoryview]:
    """Generate message converter."""
    if is_same_wireformat(src_typestore, dst_typestore, src_msgtype, dst_msgtype):
        if src_is2 == dst_is2:
            return memoryview
        if src_is2:
            return partial(src_typestore.cdr_to_ros1, typename=src_msgtype)
        return partial(dst_typestore.ros1_to_cdr, typename=dst_msgtype)

    echo(f'Msgtype will be migrated: {dst_msgtype}')
    return partial(
        migrate_bytes,
        src_typestore,
        dst_typestore,
        src_msgtype,
        dst_msgtype,
        cache,
        src_is2=src_is2,
        dst_is2=dst_is2,
    )


def create_connections_converters(
    connections: Sequence[Connection],
    typestore: Typestore | None,
    reader: AnyReader,
    writer: Writer1 | Writer2,
) -> tuple[
    dict[tuple[int, object], Connection],
    dict[str, Callable[[bytes | memoryview], memoryview]],
]:
    """Create writer connections and return mapping."""
    is2 = isinstance(writer, Writer2)

    connmap: dict[tuple[int, object], Connection] = {}
    convmap: dict[str, Callable[[bytes | memoryview], memoryview]] = {}

    cache: dict[str, object] = {}

    if not typestore:
        if reader.is2 == is2:
            typestore = reader.typestore
        else:
            header = get_typestore(Stores.ROS2_FOXY if is2 else Stores.ROS1_NOETIC).fielddefs[
                'std_msgs/msg/Header'
            ]
            typestore = get_typestore(Stores.EMPTY)
            typestore.register(
                {
                    **reader.typestore.fielddefs,
                    'std_msgs/msg/Header': header,
                },
            )

    for rconn in connections:
        topic = rconn.topic
        msgtype = STATIC_MSGTYPE_RENAMES.get(rconn.msgtype, rconn.msgtype)

        if rconn.msgtype not in convmap:
            if msgtype not in typestore.fielddefs:
                echo(f'Missing msgtype in destination, copying from source: {msgtype!r}')
                typs = get_types_from_msg(
                    reader.typestore.generate_msgdef(
                        rconn.msgtype,
                        ros_version=2 if is2 else 1,
                    )[0],
                    msgtype,
                )
                _ = typs.pop('std_msgs/msg/Header', None)
                typestore.register(typs)
            convmap[rconn.msgtype] = generate_message_converter(
                reader.typestore,
                typestore,
                rconn.msgtype,
                msgtype,
                cache,
                src_is2=reader.is2,
                dst_is2=is2,
            )

        if isinstance(writer, Writer2):
            ext2 = (
                rconn.ext
                if isinstance(rconn.ext, ConnectionExtRosbag2)
                else ConnectionExtRosbag2('cdr', LATCH if rconn.ext.latching else [])
            )
            for conn in writer.connections:
                if topic == conn.topic and msgtype == conn.msgtype and conn.ext == ext2:
                    break
            else:
                conn = writer.add_connection(
                    topic,
                    msgtype,
                    typestore=typestore,
                    serialization_format=ext2.serialization_format,
                    offered_qos_profiles=ext2.offered_qos_profiles,
                )
        else:
            ext1 = (
                rconn.ext
                if isinstance(rconn.ext, ConnectionExtRosbag1)
                else ConnectionExtRosbag1(
                    None,
                    int(any(x.durability.value == 1 for x in rconn.ext.offered_qos_profiles)),
                )
            )
            for conn in writer.connections:
                if topic == conn.topic and msgtype == conn.msgtype and conn.ext == ext1:
                    break
            else:
                conn = writer.add_connection(
                    topic,
                    msgtype,
                    typestore=typestore,
                    callerid=ext1.callerid,
                    latching=ext1.latching,
                )
        connmap[(rconn.id, rconn.owner)] = conn

    return connmap, convmap


def convert(
    srcs: Sequence[Path],
    dst: Path,
    dst_storage: str,
    dst_version: int,
    compress: str | None,
    compress_mode: str,
    default_typestore: Typestore | None,
    typestore: Typestore | None,
    exclude_topics: Sequence[str],
    include_topics: Sequence[str],
    exclude_msgtypes: Sequence[str],
    include_msgtypes: Sequence[str],
) -> None:
    """Convert between Rosbag1 and Rosbag2.

    Args:
        srcs: Rosbag files to read from.
        dst: Destination path to write rosbag to.
        dst_storage: Destination file storage backend.
        dst_version: Destination file format version.
        compress: Compression algorithm ``bz2`` or ``lz4`` for rosbag1,
            ``zstd`` for rosbag2.
        compress_mode: Compression mode ``file`` or ``message`` rosbag2.
        default_typestore: Default typestore if source files have not message definitions.
        typestore: Convert messages to this destination typestore.
        exclude_topics: Topics to exclude from conversion, even if included explicitly.
        include_topics: Topics to include in conversion, instead of all.
        exclude_msgtypes: Message types to exclude from conversion, even if included explicitly.
        include_msgtypes: Message types to include in conversion, instead of all.

    ExcGroups:
        topics: Select topics to exclude or include. (options are mutually exclusive)

    """
    try:
        writer: Writer1 | Writer2
        is2 = dst.suffix != '.bag'
        if is2:
            storage_plugin = StoragePlugin.MCAP if dst_storage == 'mcap' else StoragePlugin.SQLITE3
            writer = Writer2(
                dst,
                version=cast('Literal[8, 9]', dst_version),
                storage_plugin=storage_plugin,
            )
            if compress:
                writer.set_compression(
                    CompressionMode[compress_mode.upper()],
                    CompressionFormat[compress.upper()],
                )
        else:
            writer = Writer1(dst)
            if compress:
                writer.set_compression(writer.CompressionFormat[compress.upper()])

        with (
            AnyReader(srcs, default_typestore=default_typestore) as reader,
            writer,
        ):
            connections = [
                x
                for x in reader.connections
                if x.topic not in exclude_topics
                and x.msgtype not in exclude_msgtypes
                and (
                    x.topic in include_topics or x.msgtype in include_msgtypes
                    if include_topics and include_msgtypes
                    else x.topic in include_topics
                    if include_topics
                    else x.msgtype in include_msgtypes
                    if include_msgtypes
                    else True
                )
            ]
            if not connections:
                return

            connmap, convmap = create_connections_converters(
                connections,
                typestore,
                reader,
                writer,
            )
            for rconn, timestamp, data in reader.messages(connections=connections):
                writer.write(
                    connmap[(rconn.id, rconn.owner)],
                    timestamp,
                    convmap[rconn.msgtype](data),
                )

    except AnyReaderError as err:
        msg = f'Reading source bag: {err}'
        raise ConverterError(msg) from err
    except (WriterError1, WriterError2) as err:
        msg = f'Writing destination bag: {err}'
        raise ConverterError(msg) from err
    except AssertionError:
        raise
    except Exception as err:
        msg = f'Converting rosbag: {err!r}'
        raise ConverterError(msg) from err
