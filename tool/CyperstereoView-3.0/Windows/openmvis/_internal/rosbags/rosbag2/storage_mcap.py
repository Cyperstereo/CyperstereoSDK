# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Mcap storage."""

from __future__ import annotations

import heapq
import struct
import sys
from binascii import crc32
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import version
from io import BytesIO
from struct import iter_unpack, unpack_from
from typing import TYPE_CHECKING, NamedTuple, cast

from lz4.frame import decompress as lz4_decompress  # type: ignore[import-untyped]

if sys.version_info >= (3, 14):
    from compression import zstd
else:  # pragma: no cover
    from backports import zstd

from rosbags.interfaces import (
    Connection,
    ConnectionExtRosbag2,
    MessageDefinition,
    MessageDefinitionFormat,
    Qos,
)

from .enums import CompressionMode
from .errors import ReaderError
from .metadata import ReaderMetadata, parse_qos

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Generator
    from pathlib import Path
    from typing import BinaryIO

    from rosbags.interfaces.typing import RPath

    Unpack = Callable[[bytes], 'tuple[int]']
    Unpack2 = Callable[[bytes], 'tuple[int, int]']
    Unpack4 = Callable[[bytes], 'tuple[int, int, int, int]']
    Unpack5 = Callable[[bytes], 'tuple[int, int, int, int, int]']


class Schema(NamedTuple):
    """Schema."""

    id: int
    name: str
    encoding: str
    data: str


class Channel(NamedTuple):
    """Channel."""

    id: int
    schema: str
    topic: str
    message_encoding: str
    metadata: bytes  # dict[str, str]


class ChunkInfo(NamedTuple):
    """Chunk."""

    message_start_time: int
    message_end_time: int
    chunk_start_offset: int
    chunk_length: int
    message_index_offsets: dict[int, int]
    message_index_length: int
    compression: str
    compressed_size: int
    uncompressed_size: int
    channel_count: dict[int, int]


class Statistics(NamedTuple):
    """Statistics."""

    message_count: int
    schema_count: int
    channel_count: int
    attachement_count: int
    metadata_count: int
    chunk_count: int
    start_time: int
    end_time: int
    channel_message_counts: dict[int, int]


class Msg(NamedTuple):
    """Message wrapper."""

    timestamp: int
    offset: int
    connection: Connection | None
    data: bytes | None


MAXSIZE: int = 2**63 - 1

deserialize_uint16 = cast('Unpack', struct.Struct('<H').unpack)
deserialize_uint32 = cast('Unpack', struct.Struct('<I').unpack)
deserialize_uint64 = cast('Unpack', struct.Struct('<Q').unpack)

deserialize_hq = cast('Unpack2', struct.Struct('<HQ').unpack)
deserialize_qq = cast('Unpack2', struct.Struct('<QQ').unpack)
deserialize_qqqi = cast('Unpack4', struct.Struct('<QQQI').unpack)
deserialize_qqqq = cast('Unpack4', struct.Struct('<QQQQ').unpack)
deserialize_hiqq = cast('Unpack4', struct.Struct('<HIQQ').unpack)
deserialize_qhiqq = cast('Unpack5', struct.Struct('<QHIQQ').unpack)


def read_exact(bio: BinaryIO, size: int) -> bytes:
    """Read exact number of bytes."""
    pos = bio.tell()
    data = bio.read(size)
    if len(data) != size:
        msg = f'Truncated record at offset {pos}: expected {size} bytes, got {len(data)}.'
        raise ReaderError(msg)
    return data


def skip_exact(bio: BinaryIO, size: int) -> None:
    """Skip exact number of bytes."""
    pos = bio.tell()
    end = bio.seek(pos + size)
    if end - pos != size:
        msg = f'Truncated record at offset {pos}: expected {size} bytes, got {end - pos}.'
        raise ReaderError(msg)


def read_sized(bio: BinaryIO) -> bytes:
    """Read one record."""
    return read_exact(bio, deserialize_uint64(read_exact(bio, 8))[0])


def skip_sized(bio: BinaryIO) -> None:
    """Read one record."""
    size = deserialize_uint64(read_exact(bio, 8))[0]
    skip_exact(bio, size)


def read_bytes(bio: BinaryIO) -> bytes:
    """Read string."""
    return read_exact(bio, deserialize_uint32(read_exact(bio, 4))[0])


def read_string(bio: BinaryIO) -> str:
    """Read string."""
    return read_bytes(bio).decode()


def get_msgdef(schema: Schema | None) -> MessageDefinition:
    """Get message definition from schema."""
    fmtmap = {
        'ros2msg': MessageDefinitionFormat.MSG,
        'ros2idl': MessageDefinitionFormat.IDL,
        'omgidl': MessageDefinitionFormat.IDL,
        '': MessageDefinitionFormat.NONE,
    }
    if schema is not None:
        return MessageDefinition(fmtmap[schema.encoding], schema.data)
    return MessageDefinition(MessageDefinitionFormat.NONE, '')


def get_qos(metadata: bytes) -> list[Qos]:
    """Get QoS from metadata."""
    bio = BytesIO(metadata)
    while bio.tell() < len(metadata):
        key = read_string(bio)
        value = read_string(bio)
        if key == 'offered_qos_profiles':
            return parse_qos(value)
    return []


DECOMPRESSORS: dict[str, Callable[[bytes, int], bytes]] = {
    '': lambda x, _: x,
    'lz4': lambda x, _: lz4_decompress(x),
    'zstd': lambda x, _: zstd.decompress(x),
}


def decompress(
    data: bytes,
    compression: str,
    size: int,
    crc: int,
) -> bytes:
    """Validate a decoded chunk and its optional checksum."""
    if compression not in DECOMPRESSORS:
        msg = f'Unsupported chunk compression {compression!r}.'
        raise ReaderError(msg)
    raw = DECOMPRESSORS[compression](data, size)

    if len(raw) != size:
        msg = f'Chunk size mismatch: expected {size} bytes, got {len(raw)}.'
        raise ReaderError(msg)
    if crc and crc32(raw) != crc:
        msg = 'Chunk checksum mismatch.'
        raise ReaderError(msg)
    return raw


def msgsrc(
    chunk: ChunkInfo,
    channel_map: dict[int, Connection],
    start: int,
    stop: int,
    bio: BinaryIO,
) -> Generator[Msg, None, None]:
    """Yield messages from chunk in time order."""
    yield Msg(chunk.message_start_time, 0, None, None)

    _ = bio.seek(chunk.chunk_start_offset + 9 + 24)
    crc = deserialize_uint32(read_exact(bio, 4))[0]
    _ = bio.seek(chunk.chunk_start_offset + 9 + 40 + len(chunk.compression))
    compressed_data = read_exact(bio, chunk.compressed_size)
    subio = BytesIO(decompress(compressed_data, chunk.compression, chunk.uncompressed_size, crc))

    messages: list[Msg] = []
    while subio.tell() < chunk.uncompressed_size:
        op_ = ord(subio.read(1))
        if op_ == 0x05:
            recio = BytesIO(read_sized(subio))
            channel_id, _, log_time, _ = deserialize_hiqq(recio.read(22))
            if start <= log_time < stop and channel_id in channel_map:
                messages.append(
                    Msg(
                        log_time,
                        chunk.chunk_start_offset,
                        channel_map[channel_id],
                        recio.read(),
                    ),
                )
        else:
            skip_sized(subio)

    yield from sorted(messages, key=lambda x: x.timestamp)


class McapReader:
    """Mcap format reader."""

    def __init__(self, path: RPath) -> None:
        """Initialize."""
        self.path = path
        self.bio: BinaryIO | None = None
        self.data_start = 0
        self.data_end = 0
        self.schemas: dict[int, Schema] = {}
        self.channels: dict[int, Channel] = {}
        self.chunks: list[ChunkInfo] = []
        self.statistics: Statistics | None = None
        self.connections: list[Connection] = []
        self.metadata = ReaderMetadata(0, 2**63 - 1, 0, 0, None, None, None, None)

    def _open(self) -> None:
        """Read MCAP metadata and indexes."""
        assert self.bio
        magic = self.bio.read(8)
        if not magic:
            msg = f'File {str(self.path)!r} seems to be empty.'
            raise ReaderError(msg)

        if magic != b'\x89MCAP0\r\n':
            msg = 'File magic is invalid.'
            raise ReaderError(msg)

        op_ = ord(self.bio.read(1))
        if op_ != 0x01:
            msg = 'Unexpected record.'
            raise ReaderError(msg)

        recio = BytesIO(read_sized(self.bio))
        profile = read_string(recio)
        if profile != 'ros2':
            msg = 'Profile is not ros2.'
            raise ReaderError(msg)
        self.data_start = self.bio.tell()

        _ = self.bio.seek(-37, 2)
        footer_start = self.bio.tell()
        data = self.bio.read()
        magic = data[-8:]
        if magic != b'\x89MCAP0\r\n':
            msg = 'File end magic is invalid.'
            raise ReaderError(msg)

        assert len(data) == 37
        if data[0:9] != b'\x02\x14\x00\x00\x00\x00\x00\x00\x00':
            msg = 'Invalid MCAP footer.'
            raise ReaderError(msg)

        (summary_start,) = deserialize_uint64(data[9:17])
        if summary_start and not self.data_start <= summary_start <= footer_start:
            msg = 'Invalid MCAP summary offset.'
            raise ReaderError(msg)
        if summary_start:
            self.data_end = summary_start
            self.read_index()
            if self.statistics:
                if not self.schemas:
                    self.meta_scan()
            elif self.chunks:
                message_count = sum(sum(x.channel_count.values()) for x in self.chunks)
                start_time = min(x.message_start_time for x in self.chunks)
                end_time = max(x.message_end_time for x in self.chunks)
                cstats: dict[int, int] = defaultdict(int)
                for chunk in self.chunks:
                    for cid, count in chunk.channel_count.items():
                        cstats[cid] += count
                self.statistics = Statistics(
                    message_count,
                    len(self.schemas),
                    len(self.channels),
                    0,
                    0,
                    len(self.chunks),
                    start_time,
                    end_time,
                    cstats,
                )
            else:
                self.meta_scan()
        else:
            self.data_end = footer_start
            self.meta_scan()

        assert self.statistics
        channel_message_counts = self.statistics.channel_message_counts
        message_count = self.statistics.message_count
        start_time = self.statistics.start_time
        end_time = self.statistics.end_time
        duration = end_time - start_time

        self.connections = [
            x._replace(msgcount=channel_message_counts.get(x.id, 0)) for x in self.connections
        ]

        self.metadata = self.metadata._replace(
            duration=duration + 1 if message_count else 0,
            start_time=start_time if message_count else MAXSIZE,
            end_time=end_time + 1 if message_count else 0,
            message_count=message_count,
        )

    def open(self) -> None:
        """Open storage."""
        try:
            self.bio = self.path.open('rb')
        except OSError as err:
            msg = f'Could not open file {str(self.path)!r}: {err.strerror}.'
            raise ReaderError(msg) from err

        try:
            self._open()
        except BaseException:
            with suppress(Exception):
                self.close()
            raise

    def read_index(self) -> None:
        """Read index from file."""
        bio = self.bio
        assert bio

        schemas = self.schemas
        channels = self.channels
        connections = self.connections
        conn_ids = {x.id for x in self.connections}
        chunks = self.chunks

        _ = bio.seek(self.data_end)
        while True:
            op_ = ord(bio.read(1))

            if op_ in {0x02, 0x0E}:
                break

            rec = BytesIO(read_sized(bio))

            if op_ == 0x03:
                (key,) = deserialize_uint16(rec.read(2))
                schemas[key] = Schema(
                    key,
                    read_string(rec),
                    read_string(rec),
                    read_string(rec),
                )

            elif op_ == 0x04:
                (key,) = deserialize_uint16(rec.read(2))
                schema_name = schemas.get(
                    (schema_id := deserialize_uint16(rec.read(2))[0]),
                    Schema(0, '__schemaless__', 'cdr', ''),
                ).name
                channels[key] = Channel(
                    key,
                    schema_name,
                    read_string(rec),
                    read_string(rec),
                    read_bytes(rec),
                )
                if key not in conn_ids:
                    connections.append(
                        Connection(
                            key,
                            channels[key].topic,
                            channels[key].schema,
                            get_msgdef(self.schemas.get(schema_id)),
                            '',
                            0,
                            ConnectionExtRosbag2(
                                channels[key].message_encoding,
                                get_qos(channels[key].metadata),
                            ),
                            self,
                        )
                    )
                    conn_ids.add(key)

            elif op_ == 0x08:
                chunk = ChunkInfo(
                    *deserialize_qqqq(rec.read(32)),
                    {
                        x[0]: x[1]
                        for x in cast(
                            'Collection[tuple[int, int]]',
                            iter_unpack('<HQ', read_bytes(rec)),
                        )
                    },
                    *deserialize_uint64(rec.read(8)),
                    read_string(rec),
                    *deserialize_qq(rec.read(16)),
                    {},
                )
                offset_channel = sorted((v, k) for k, v in chunk.message_index_offsets.items())
                offsets = [
                    *[x[0] for x in offset_channel],
                    chunk.chunk_start_offset + chunk.chunk_length + chunk.message_index_length,
                ]
                chunk.channel_count.update(
                    {
                        x[1]: count // 16
                        for x, y, z in zip(offset_channel, offsets[1:], offsets, strict=False)
                        if (count := y - z - 15)
                    },
                )
                chunks.append(chunk)

            elif op_ == 0x0B:
                self.statistics = Statistics(
                    *cast(
                        'tuple[int, int, int, int, int, int, int ,int]',
                        unpack_from('<QHIIIIQQ', rec.read(42), 0),
                    ),
                    dict(
                        deserialize_hq(rec.read(10))
                        for _ in range(deserialize_uint32(rec.read(4))[0] // 10)
                    ),
                )

    def close(self) -> None:
        """Close storage."""
        bio, self.bio = self.bio, None
        assert bio

        self.schemas.clear()
        self.channels.clear()
        self.chunks.clear()
        self.statistics = None
        self.connections.clear()
        self.metadata = ReaderMetadata(0, 2**63 - 1, 0, 0, None, None, None, None)
        bio.close()

    def meta_scan(self) -> None:
        """Generate metadata by scanning through file."""
        assert self.bio
        bio = self.bio
        bio_size = self.data_end
        _ = bio.seek(self.data_start)

        msgcount = 0
        start_time = 2**63 - 1
        end_time = 0
        nchunks = 0
        cstats: dict[int, int] = defaultdict(int)

        schemas = self.schemas
        channels = self.channels
        connections = self.connections
        conn_ids = {x.id for x in self.connections}

        while bio.tell() < bio_size:
            op_ = ord(bio.read(1))

            if op_ == 0x03:
                rec = BytesIO(read_sized(bio))
                (key,) = deserialize_uint16(rec.read(2))
                schemas[key] = Schema(
                    key,
                    read_string(rec),
                    read_string(rec),
                    read_string(rec),
                )
            elif op_ == 0x04:
                rec = BytesIO(read_sized(bio))
                (key,) = deserialize_uint16(rec.read(2))
                schema_name = schemas.get(
                    (schema_id := deserialize_uint16(rec.read(2))[0]),
                    Schema(0, '__schemaless__', 'cdr', ''),
                ).name
                channels[key] = Channel(
                    key,
                    schema_name,
                    read_string(rec),
                    read_string(rec),
                    read_bytes(rec),
                )
                if key not in conn_ids:
                    connections.append(
                        Connection(
                            key,
                            channels[key].topic,
                            channels[key].schema,
                            get_msgdef(self.schemas.get(schema_id)),
                            '',
                            0,
                            ConnectionExtRosbag2(
                                channels[key].message_encoding,
                                get_qos(channels[key].metadata),
                            ),
                            self,
                        )
                    )
                    conn_ids.add(key)
            elif op_ == 0x05:
                (size,) = deserialize_uint64(bio.read(8))
                if size < 22:
                    msg = f'Invalid message record length {size}.'
                    raise ReaderError(msg)
                (cid,) = deserialize_uint16(read_exact(bio, 2))
                _ = bio.seek(4, 1)
                (timestamp,) = deserialize_uint64(bio.read(8))
                msgcount += 1
                start_time = min(timestamp, start_time)
                end_time = max(timestamp, end_time)
                cstats[cid] += 1
                skip_exact(bio, size - 14)
            elif op_ == 0x06:
                if bio is not self.bio:
                    msg = 'Nested MCAP chunks are not supported.'
                    raise ReaderError(msg)
                rec = BytesIO(read_sized(bio))
                _, _, uncompressed_size, crc = deserialize_qqqi(read_exact(rec, 28))
                compression = read_string(rec)
                (compressed_size,) = deserialize_uint64(read_exact(rec, 8))
                bio = BytesIO(
                    decompress(
                        read_exact(rec, compressed_size),
                        compression,
                        uncompressed_size,
                        crc,
                    ),
                )
                bio_size = uncompressed_size
                nchunks += 1
            else:
                skip_sized(bio)

            if bio.tell() == bio_size and bio != self.bio:
                bio = self.bio
                bio_size = self.data_end

        self.statistics = Statistics(
            msgcount,
            len(schemas),
            len(channels),
            0,
            0,
            nchunks,
            start_time,
            end_time,
            cstats,
        )

    def messages_scan(
        self,
        connections: Collection[Connection],
        start: int | None = None,
        stop: int | None = None,
    ) -> Generator[tuple[Connection, int, bytes], None, None]:
        """Read messages by scanning whole bag."""
        assert self.bio
        bio = self.bio
        bio_size = self.data_end
        _ = bio.seek(self.data_start)

        cmap = {x.id: x for x in connections}

        if start is None:
            start = 0
        if stop is None:
            stop = MAXSIZE

        while bio.tell() < bio_size:
            op_ = ord(bio.read(1))

            if op_ == 0x05:
                size, channel_id, _, timestamp, _ = deserialize_qhiqq(bio.read(30))
                if size < 22:
                    msg = f'Invalid message record length {size}.'
                    raise ReaderError(msg)
                data = read_exact(bio, size - 22)
                if start <= timestamp < stop and channel_id in cmap:
                    yield cmap[channel_id], timestamp, data
            elif op_ == 0x06:
                (size,) = deserialize_uint64(read_exact(bio, 8))
                if size < 40 or bio is not self.bio:
                    msg = f'Invalid chunk record length {size} or nested chunk.'
                    raise ReaderError(msg)
                record_start = bio.tell()
                start_time, end_time, uncompressed_size, crc = deserialize_qqqi(read_exact(bio, 28))
                if start <= end_time and start_time < stop:
                    compression = read_string(bio)
                    (compressed_size,) = deserialize_uint64(read_exact(bio, 8))
                    if bio.tell() + compressed_size > record_start + size:
                        msg = 'Compressed data exceeds its chunk record boundary.'
                        raise ReaderError(msg)
                    compressed = read_exact(bio, compressed_size)
                    skip_exact(bio, record_start + size - bio.tell())
                    bio = BytesIO(
                        decompress(
                            compressed,
                            compression,
                            uncompressed_size,
                            crc,
                        ),
                    )
                    bio_size = uncompressed_size
                else:
                    skip_exact(bio, size - 28)
            else:
                skip_sized(bio)

            if bio.tell() == bio_size and bio != self.bio:
                bio = self.bio
                bio_size = self.data_end

    def messages(
        self,
        connections: Collection[Connection],
        start: int | None = None,
        stop: int | None = None,
    ) -> Generator[tuple[Connection, int, bytes], None, None]:
        """Read messages.

        Args:
            connections: Collection with connections to filter for.
            start: Yield only messages at or after this timestamp (ns).
            stop: Yield only messages before this timestamp (ns).

        Yields:
            Tuples of connection, timestamp (ns), and rawdata.

        """
        assert self.bio

        if not self.chunks:
            previous: int = -(2**63)
            for message in self.messages_scan(connections, start, stop):
                if message[1] < previous:
                    msg = 'Unindexed MCAP messages are not in timestamp order. Use an indexed file.'
                    raise ReaderError(msg)
                previous = message[1]
                yield message
            return

        channel_map = {conn.id: conn for conn in connections}

        chunks = [
            msgsrc(
                x,
                channel_map,
                x.message_start_time if start is None else start,
                x.message_end_time + 1 if stop is None else stop,
                self.bio,
            )
            for x in self.chunks
            if (start is None or start < x.message_end_time + 1)
            and (stop is None or x.message_start_time < stop)
            and (any(x.channel_count.get(cid, 0) for cid in channel_map))
        ]

        for timestamp, offset, connection, data in heapq.merge(*chunks):
            if not offset:
                continue
            assert connection
            assert data is not None
            yield connection, timestamp, data


def write_uint64(bio: BinaryIO, uint: int) -> None:
    """Serialize and write uint64."""
    bio.write(uint.to_bytes(8, byteorder='little'))


def write_uint32(bio: BinaryIO, uint: int) -> None:
    """Serialize and write uint32."""
    bio.write(uint.to_bytes(4, byteorder='little'))


def write_uint16(bio: BinaryIO, uint: int) -> None:
    """Serialize and write uint16."""
    bio.write(uint.to_bytes(2, byteorder='little'))


def write_string(bio: BinaryIO, string: str) -> None:
    """Serialize and write string."""
    data = string.encode()
    write_uint32(bio, len(data))
    bio.write(data)


def write_schema(bio: BinaryIO, schema: Schema) -> None:
    """Write schema."""
    rec = BytesIO()
    write_uint16(rec, schema.id)
    write_string(rec, schema.name)
    write_string(rec, schema.encoding)
    write_string(rec, schema.data)
    write_record(bio, 0x03, rec)


def write_channel(bio: BinaryIO, channel: Channel, schemas: Collection[Schema]) -> None:
    """Write schema."""
    rec = BytesIO()
    write_uint16(rec, channel.id)
    write_uint16(rec, next(x.id for x in schemas if x.name == channel.schema))
    write_string(rec, channel.topic)
    write_string(rec, channel.message_encoding)
    write_uint32(rec, len(channel.metadata))
    rec.write(channel.metadata)
    write_record(bio, 0x04, rec)


def write_record(bio: BinaryIO, op_: int, record: BytesIO) -> None:
    """Write record."""
    bio.write(op_.to_bytes(1, byteorder='little'))
    write_uint64(bio, record.tell())
    bio.write(record.getbuffer())


@dataclass
class PendingChunk:
    """Chunk."""

    message_start_time: int
    message_end_time: int
    bio: BytesIO
    msgs: dict[int, list[tuple[int, int]]]


class McapWriter:
    """Mcap Storage Writer."""

    def __init__(self, path: Path, compression: CompressionMode) -> None:
        """Initialize mcap storage."""
        self.path = path / f'{path.name}.mcap'
        self.bio = self.path.open('xb')

        _ = self.bio.write(b'\x89MCAP\x30\r\n')
        rec = BytesIO()
        write_string(rec, 'ros2')
        write_string(rec, f'rosbags-{version("rosbags")}')
        write_record(self.bio, 0x01, rec)

        self.schemas: list[Schema] = []
        self.channels: list[Channel] = []
        self.chunks: list[BytesIO] = []
        self.chunk = PendingChunk(2**63 - 1, 0, BytesIO(), defaultdict(list))

        self.compressor: Callable[[bytes], bytes]
        if compression == CompressionMode.STORAGE:
            self.compression = 'zstd'
            self.compressor = zstd.compress
        else:
            self.compression = ''
            self.compressor = lambda x: x

        self.message_start_time = 2**63 - 1
        self.message_end_time = 0
        self.channel_stats: dict[int, int] = {}

    def add_msgtype(self, connection: Connection) -> None:
        """Add msgtype.

        Args:
            connection: Connection.

        """
        self.schemas.append(
            Schema(
                len(self.schemas) + 1,
                connection.msgtype,
                'ros2msg' if connection.msgdef.format == MessageDefinitionFormat.MSG else 'ros2idl',
                connection.msgdef.data,
            )
        )
        write_schema(self.chunk.bio, self.schemas[-1])

    def add_connection(self, connection: Connection, offered_qos_profiles: str) -> None:
        """Add connection.

        Args:
            connection: Connection.
            offered_qos_profiles: Serialized QoS profiles.

        """
        metadata = BytesIO()
        write_string(metadata, 'offered_qos_profiles')
        write_string(metadata, offered_qos_profiles)

        self.channels.append(
            Channel(
                connection.id,
                connection.msgtype,
                connection.topic,
                cast('ConnectionExtRosbag2', connection.ext).serialization_format,
                metadata.getvalue(),
            )
        )
        self.channel_stats[connection.id] = 0
        write_channel(self.chunk.bio, self.channels[-1], self.schemas)

    def close_chunk(self) -> None:
        """Close pending chunk."""
        rec = BytesIO()
        write_uint64(rec, self.chunk.message_start_time)
        write_uint64(rec, self.chunk.message_end_time)
        write_uint64(rec, self.chunk.bio.tell())
        write_uint32(rec, 0)

        compressed = self.compressor(self.chunk.bio.getvalue())
        write_string(rec, self.compression)
        write_uint64(rec, len(compressed))
        rec.write(compressed)

        chunk_start = self.bio.tell()
        write_record(self.bio, 0x06, rec)
        chunk_end = self.bio.tell()

        offsets: list[tuple[int, int]] = []
        for cid, msgs in self.chunk.msgs.items():
            offsets.append((cid, self.bio.tell()))
            rec = BytesIO()
            write_uint16(rec, cid)
            write_uint32(rec, len(msgs) * 16)
            for ts, offset in msgs:
                write_uint64(rec, ts)
                write_uint64(rec, offset)
            write_record(self.bio, 0x07, rec)

        rec = BytesIO()
        write_uint64(rec, self.chunk.message_start_time)
        write_uint64(rec, self.chunk.message_end_time)
        write_uint64(rec, chunk_start)
        write_uint64(rec, chunk_end - chunk_start)
        write_uint32(rec, len(offsets) * 10)
        for cid, offset in offsets:
            write_uint16(rec, cid)
            write_uint64(rec, offset)
        write_uint64(rec, self.bio.tell() - chunk_end)
        write_string(rec, self.compression)
        write_uint64(rec, len(compressed))
        write_uint64(rec, self.chunk.bio.tell())
        self.chunks.append(rec)

        self.chunk = PendingChunk(2**63 - 1, 0, BytesIO(), defaultdict(list))

    def write(self, connection: Connection, timestamp: int, data: bytes | memoryview) -> None:
        """Write message.

        Args:
            connection: Connection to write message to.
            timestamp: Message timestamp (ns).
            data: Serialized message data.

        """
        self.message_start_time = min(timestamp, self.message_start_time)
        self.message_end_time = max(timestamp, self.message_end_time)
        self.channel_stats[connection.id] += 1
        self.chunk.message_start_time = min(timestamp, self.chunk.message_start_time)
        self.chunk.message_end_time = max(timestamp, self.chunk.message_end_time)
        self.chunk.msgs[connection.id].append((timestamp, self.chunk.bio.tell()))

        rec = BytesIO()
        write_uint16(rec, connection.id)
        write_uint32(rec, 0)
        write_uint64(rec, timestamp)
        write_uint64(rec, timestamp)
        rec.write(data)
        write_record(self.chunk.bio, 0x05, rec)

        if self.chunk.bio.tell() > 2**20:
            self.close_chunk()

    def close(self, version: int, metadata: str) -> None:
        """Close after writing.

        Closes open database transactions and writes metadata.yaml.

        """
        _ = version
        if self.chunk.bio.tell():
            self.close_chunk()

        metadata_start = self.bio.tell()
        rec = BytesIO()
        write_string(rec, 'rosbag2')
        sub = BytesIO()
        write_string(sub, 'serialized_metadata')
        write_string(sub, metadata)
        write_uint32(rec, sub.tell())
        rec.write(sub.getbuffer())
        write_record(self.bio, 0x0C, rec)
        metadata_end = self.bio.tell()

        rec = BytesIO()
        write_uint32(rec, 0)
        write_record(self.bio, 0x0F, rec)

        schema_start = self.bio.tell()
        for schema in self.schemas:
            write_schema(self.bio, schema)

        channel_start = self.bio.tell()
        for channel in self.channels:
            write_channel(self.bio, channel, self.schemas)

        chunk_start = self.bio.tell()
        for chunk in self.chunks:
            write_record(self.bio, 0x08, chunk)

        metadata_index_start = self.bio.tell()
        rec = BytesIO()
        write_uint64(rec, metadata_start)
        write_uint64(rec, metadata_end - metadata_start)
        write_string(rec, 'rosbag2')
        write_record(self.bio, 0x0D, rec)

        statistics_offset_start = self.bio.tell()
        rec = BytesIO()
        write_uint64(rec, sum(self.channel_stats.values()))
        write_uint16(rec, len(self.schemas))
        write_uint32(rec, len(self.channels))
        write_uint32(rec, 0)  # attachments
        write_uint32(rec, 1)  # metadata
        write_uint32(rec, len(self.chunks))
        write_uint64(rec, self.message_start_time)
        write_uint64(rec, self.message_end_time)
        write_uint32(rec, len(self.channel_stats) * 10)
        for cid, count in sorted(self.channel_stats.items()):
            write_uint16(rec, cid)
            write_uint64(rec, count)
        write_record(self.bio, 0x0B, rec)

        summary_offset_start = self.bio.tell()

        if schema_start != channel_start:
            rec = BytesIO()
            rec.write(b'\x03')
            write_uint64(rec, schema_start)
            write_uint64(rec, channel_start - schema_start)
            write_record(self.bio, 0x0E, rec)

        if channel_start != chunk_start:
            rec = BytesIO()
            rec.write(b'\x04')
            write_uint64(rec, channel_start)
            write_uint64(rec, chunk_start - channel_start)
            write_record(self.bio, 0x0E, rec)

        if chunk_start != metadata_index_start:
            rec = BytesIO()
            rec.write(b'\x08')
            write_uint64(rec, chunk_start)
            write_uint64(rec, metadata_index_start - chunk_start)
            write_record(self.bio, 0x0E, rec)

        rec = BytesIO()
        rec.write(b'\x0d')
        write_uint64(rec, metadata_index_start)
        write_uint64(rec, statistics_offset_start - metadata_index_start)
        write_record(self.bio, 0x0E, rec)

        rec = BytesIO()
        rec.write(b'\x0b')
        write_uint64(rec, statistics_offset_start)
        write_uint64(rec, summary_offset_start - statistics_offset_start)
        write_record(self.bio, 0x0E, rec)

        rec = BytesIO()
        write_uint64(rec, schema_start)
        write_uint64(rec, summary_offset_start)
        write_uint32(rec, 0)
        write_record(self.bio, 0x02, rec)
        _ = self.bio.write(b'\x89MCAP\x30\r\n')

        self.bio.close()

    def abort(self) -> None:
        """Close without writing index and footer."""
        self.bio.close()
