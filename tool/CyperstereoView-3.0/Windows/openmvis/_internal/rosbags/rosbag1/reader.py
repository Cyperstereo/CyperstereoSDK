# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Rosbag1 v2.0 reader."""

from __future__ import annotations

import heapq
import os
import re
import struct
import sys
from bz2 import decompress as bz2_decompress
from collections import defaultdict
from contextlib import suppress
from enum import Enum, IntEnum
from functools import reduce
from io import BytesIO
from itertools import groupby
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover
    from typing_extensions import override

from lz4.frame import decompress as lz4_decompress  # type: ignore[import-untyped]

from rosbags.interfaces import (
    Connection,
    ConnectionExtRosbag1,
    MessageDefinition,
    MessageDefinitionFormat,
    TopicInfo,
)
from rosbags.typesys.msg import normalize_msgtype

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Generator
    from types import TracebackType
    from typing import BinaryIO, Literal

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from rosbags.interfaces.typing import RPath

    Unpack = Callable[[bytes], tuple[int]]
    UnpackFrom = Callable[[bytes, int], tuple[int]]


class ReaderError(Exception):
    """Reader Error."""

    args: tuple[str]


class Compression(Enum):
    """Compression mode."""

    NONE = 'none'
    BZ2 = 'bz2'
    LZ4 = 'lz4'


class RecordType(IntEnum):
    """Record type."""

    MSGDATA = 2
    BAGHEADER = 3
    IDXDATA = 4
    CHUNK = 5
    CHUNK_INFO = 6
    CONNECTION = 7


class ChunkInfo(NamedTuple):
    """Chunk information."""

    pos: int
    start_time: int
    end_time: int
    connection_counts: dict[int, int]


class Chunk(NamedTuple):
    """Chunk metadata."""

    datasize: int
    datapos: int
    decompressor: Callable[[bytes], bytes]


class IndexData(NamedTuple):
    """Index data."""

    time: int
    chunk_pos: int
    offset: int

    @override
    def __lt__(self, other: tuple[int, ...]) -> bool:
        """Compare by time only."""
        return self.time < other[0]

    @override
    def __le__(self, other: tuple[int, ...]) -> bool:
        """Compare by time only."""
        return self.time <= other[0]

    @override
    def __eq__(self, other: object) -> bool:
        """Compare by time only."""
        if isinstance(other, IndexData):
            return self.time == other[0]
        return NotImplemented  # pragma: no cover

    @override
    def __ge__(self, other: tuple[int, ...]) -> bool:
        """Compare by time only."""
        return self.time >= other[0]

    @override
    def __gt__(self, other: tuple[int, ...]) -> bool:
        """Compare by time only."""
        return self.time > other[0]

    @override
    def __ne__(self, other: object) -> bool:
        """Compare by time only."""
        if isinstance(other, IndexData):
            return self.time != other[0]
        return NotImplemented  # pragma: no cover

    @override
    def __hash__(self) -> int:
        """Use normal tuple hash."""
        return hash(tuple(self))  # pragma: no cover


decompressors: dict[str, Callable[[bytes], bytes]] = {
    Compression.NONE.value: lambda x: x,
    Compression.BZ2.value: bz2_decompress,
    Compression.LZ4.value: cast('Callable[[bytes], bytes]', lz4_decompress),
}

deserialize_uint8 = cast('Unpack', struct.Struct('<B').unpack)
deserialize_uint32 = cast('UnpackFrom', struct.Struct('<L').unpack_from)
deserialize_uint64 = cast('Unpack', struct.Struct('<Q').unpack)


def deserialize_time(val: bytes) -> int:
    """Deserialize time value.

    Args:
        val: Serialized bytes.

    Returns:
        Deserialized value.

    """
    unpacked: tuple[int, int] = struct.unpack('<LL', val)
    sec, nsec = unpacked
    return sec * 10**9 + nsec


class Header(dict[str, bytes]):
    """Record header."""

    def get_uint8(self, name: str) -> int:
        """Get uint8 value from field.

        Args:
            name: Name of field.

        Returns:
            Deserialized value.

        Raises:
            ReaderError: Field not present or not deserializable.

        """
        try:
            return deserialize_uint8(self[name])[0]
        except (KeyError, struct.error) as err:
            msg = f'Could not read uint8 field {name!r}.'
            raise ReaderError(msg) from err

    def get_uint32(self, name: str) -> int:
        """Get uint32 value from field.

        Args:
            name: Name of field.

        Returns:
            Deserialized value.

        Raises:
            ReaderError: Field not present or not deserializable.

        """
        try:
            return deserialize_uint32(self[name], 0)[0]
        except (KeyError, struct.error) as err:
            msg = f'Could not read uint32 field {name!r}.'
            raise ReaderError(msg) from err

    def get_uint64(self, name: str) -> int:
        """Get uint64 value from field.

        Args:
            name: Name of field.

        Returns:
            Deserialized value.

        Raises:
            ReaderError: Field not present or not deserializable.

        """
        try:
            return deserialize_uint64(self[name])[0]
        except (KeyError, struct.error) as err:
            msg = f'Could not read uint64 field {name!r}.'
            raise ReaderError(msg) from err

    def get_string(self, name: str) -> str:
        """Get string value from field.

        Args:
            name: Name of field.

        Returns:
            Deserialized value.

        Raises:
            ReaderError: Field not present or not deserializable.

        """
        try:
            value = self[name]
            assert isinstance(value, bytes)
            return value.decode()
        except (KeyError, ValueError) as err:
            msg = f'Could not read string field {name!r}.'
            raise ReaderError(msg) from err

    def get_time(self, name: str) -> int:
        """Get time value from field.

        Args:
            name: Name of field.

        Returns:
            Deserialized value.

        Raises:
            ReaderError: Field not present or not deserializable.

        """
        try:
            return deserialize_time(self[name])
        except (KeyError, struct.error) as err:
            msg = f'Could not read time field {name!r}.'
            raise ReaderError(msg) from err

    @classmethod
    def read(cls: type[Header], src: BinaryIO, expect: RecordType | None = None) -> Header:
        """Read header from file handle.

        Args:
            src: File handle.
            expect: Expected record op.

        Returns:
            Header object.

        Raises:
            ReaderError: Header could not parsed.

        """
        try:
            binary = read_bytes(src, read_uint32(src))
        except ReaderError as err:
            msg = 'Header could not be read from file.'
            raise ReaderError(msg) from err

        header = cls()
        pos = 0
        length = len(binary)
        while pos < length:
            try:
                size = deserialize_uint32(binary, pos)[0]
            except struct.error as err:
                msg = 'Header field size could not be read.'
                raise ReaderError(msg) from err
            pos += 4

            if pos + size > length:
                msg = 'Declared field size is too large for header.'
                raise ReaderError(msg)

            name, sep, value = binary[pos : pos + size].partition(b'=')
            if not sep:
                msg = 'Header field could not be parsed.'
                raise ReaderError(msg)
            pos += size

            header[name.decode()] = value

        if expect:
            have = header.get_uint8('op')
            if expect != have:
                msg = f'Record of type {RecordType(have).name!r} is unexpected.'
                raise ReaderError(msg)

        return header


def read_uint32(src: BinaryIO) -> int:
    """Read uint32 from source.

    Args:
        src: File handle.

    Returns:
        Uint32 value.

    Raises:
        ReaderError: Value unreadable or not deserializable.

    """
    try:
        return deserialize_uint32(src.read(4), 0)[0]
    except struct.error as err:
        msg = 'Could not read uint32.'
        raise ReaderError(msg) from err


def read_bytes(src: BinaryIO, size: int) -> bytes:
    """Read bytes from source.

    Args:
        src: File handle.
        size: Number of bytes to read.

    Returns:
        Read bytes.

    Raises:
        ReaderError: Not enough bytes available.

    """
    data = src.read(size)
    if len(data) != size:
        msg = f'Got only {len(data)} of requested {size} bytes.'
        raise ReaderError(msg)
    return data


def normalize(name: str) -> str:
    """Normalize topic name.

    Args:
        name: Topic name.

    Returns:
        Normalized name.

    """
    return f'{"/" * (name[0] == "/")}{"/".join(x for x in name.split("/") if x)}'


class Reader:
    """Rosbag 1 version 2.0 reader.

    This class is designed for a ROS2 world, it will automatically normalize
    message type names to be in line with their ROS2 counterparts.

    """

    def __init__(self, path: str | RPath) -> None:
        """Initialize.

        Args:
            path: Filesystem path to bag.

        Raises:
            ReaderError: Path does not exist.

        """
        self.path: RPath = Path(path) if isinstance(path, str) else path
        if not self.path.exists():
            msg = f'File {str(self.path)!r} does not exist.'
            raise FileNotFoundError(msg)

        self.bio: BinaryIO | None = None
        self.connections: list[Connection] = []
        self.indexes: dict[int, list[IndexData]] = {}
        self.index_data_header_offsets: tuple[int, int] | None = None
        self.chunk_infos: list[ChunkInfo] = []
        self.chunks: dict[int, Chunk] = {}
        self.current_chunk: tuple[int, BinaryIO] = (-1, BytesIO())

    def _check_open(self) -> None:
        """Ensure reader is open."""
        if not self.bio:
            msg = 'Reader is not open.'
            raise ReaderError(msg)

    @property
    def duration(self) -> int:
        """Duration in nanoseconds between earliest and latest messages."""
        self._check_open()
        duration = self.end_time - self.start_time
        return max(duration, 0)

    @property
    def start_time(self) -> int:
        """Timestamp in nanoseconds of the earliest message."""
        self._check_open()
        return min(x.start_time for x in self.chunk_infos) if self.chunk_infos else 2**63 - 1

    @property
    def end_time(self) -> int:
        """Timestamp in nanoseconds after the latest message."""
        self._check_open()
        return max(x.end_time for x in self.chunk_infos) if self.chunk_infos else 0

    @property
    def message_count(self) -> int:
        """Total message count."""
        self._check_open()
        return reduce(lambda x, y: x + y, (x.msgcount for x in self.topics.values()), 0)

    @property
    def topics(self) -> dict[str, TopicInfo]:
        """Topic information."""
        self._check_open()
        topics: dict[str, TopicInfo] = {}
        for topic, group in groupby(
            sorted(self.connections, key=lambda x: x.topic),
            key=lambda x: x.topic,
        ):
            connections = list(group)
            msgcount = reduce(
                lambda x, y: x + y,
                (y.connection_counts.get(x.id, 0) for x in connections for y in self.chunk_infos),
                0,
            )

            topics[topic] = TopicInfo(
                msgtypes.pop() if len(msgtypes := {x.msgtype for x in connections}) == 1 else None,
                msgdefs.pop()
                if len(msgdefs := {x.msgdef for x in connections}) == 1
                else MessageDefinition(MessageDefinitionFormat.NONE, ''),
                msgcount,
                connections,
            )
        return topics

    def open(self) -> None:
        """Open rosbag and read metadata."""
        if self.bio is not None:
            msg = 'Reader is already open.'
            raise ReaderError(msg)
        try:
            self.bio = self.path.open('rb')
        except PermissionError:
            raise
        except OSError as err:
            msg = f'Could not open file {str(self.path)!r}: {err.strerror}.'
            raise ReaderError(msg) from err

        try:
            magic = self.bio.readline().decode()
            if not magic:
                msg = f'File {str(self.path)!r} seems to be empty.'
                raise ReaderError(msg)  # noqa: TRY301

            matches = re.match(r'#ROSBAG V(\d+).(\d+)\n', magic)
            if not matches:
                msg = 'File magic is invalid.'
                raise ReaderError(msg)  # noqa: TRY301
            major, minor = matches.groups()
            version = int(major) * 100 + int(minor)
            if version != 200:
                msg = f'Bag version {version!r} is not supported.'
                raise ReaderError(msg)  # noqa: TRY301

            header = Header.read(self.bio, RecordType.BAGHEADER)
            index_pos = header.get_uint64('index_pos')
            conn_count = header.get_uint32('conn_count')
            chunk_count = header.get_uint32('chunk_count')
            try:
                encryptor: str | None = header.get_string('encryptor')
            except ReaderError:
                encryptor = None
            if encryptor:
                msg = f'Bag encryption {encryptor!r} is not supported.'
                raise ReaderError(msg) from None  # noqa: TRY301

            if index_pos == 0:
                msg = 'Bag is not indexed, reindex before reading.'
                raise ReaderError(msg)  # noqa: TRY301

            if chunk_count == 0:
                return

            _ = self.bio.seek(index_pos)
            try:
                self.connections = [self.read_connection() for _ in range(conn_count)]
                self.chunk_infos = [self.read_chunk_info() for _ in range(chunk_count)]
            except ReaderError as err:
                msg = f'Bag index looks damaged: {err.args}'
                raise ReaderError(msg) from None

            self.chunks = {}
            indexes: dict[int, list[IndexData]] = defaultdict(list)
            for chunk_info in self.chunk_infos:
                _ = self.bio.seek(chunk_info.pos)
                self.chunks[chunk_info.pos] = self.read_chunk()

                for _ in range(len(chunk_info.connection_counts)):
                    self.read_index_data(chunk_info.pos, indexes)

            self.indexes = {x.id: sorted(indexes[x.id]) for x in self.connections}

            self.connections = [
                Connection(*x[0:5], len(self.indexes[x.id]), *x[6:]) for x in self.connections
            ]
        except BaseException:
            with suppress(Exception):
                self.close()
            raise

    def close(self) -> None:
        """Close rosbag."""
        self._check_open()
        bio, self.bio = self.bio, None
        assert bio

        self.connections.clear()
        self.indexes.clear()
        self.index_data_header_offsets = None
        self.chunk_infos.clear()
        self.chunks.clear()
        self.current_chunk[1].close()
        self.current_chunk = (-1, BytesIO())
        bio.close()

    def read_connection(self) -> Connection:
        """Read connection record from current position."""
        assert self.bio
        header = Header.read(self.bio, RecordType.CONNECTION)
        conn = header.get_uint32('conn')
        topic = normalize(header.get_string('topic'))

        header = Header.read(self.bio)
        typ = header.get_string('type')
        md5sum = header.get_string('md5sum')
        msgdef = MessageDefinition(
            MessageDefinitionFormat.MSG,
            header.get_string('message_definition'),
        )

        callerid = header.get_string('callerid') if 'callerid' in header else None
        _latching_raw = header.get_string('latching') if 'latching' in header else None
        latching = int(_latching_raw) if _latching_raw not in (None, '') else None

        return Connection(
            conn,
            topic,
            normalize_msgtype(typ),
            msgdef,
            md5sum,
            0,
            ConnectionExtRosbag1(callerid, latching),
            self,
        )

    def read_chunk_info(self) -> ChunkInfo:
        """Read chunk info record from current position."""
        assert self.bio
        header = Header.read(self.bio, RecordType.CHUNK_INFO)

        ver = header.get_uint32('ver')
        if ver != 1:
            msg = f'CHUNK_INFO version {ver} is not supported.'
            raise ReaderError(msg)

        chunk_pos = header.get_uint64('chunk_pos')
        count = header.get_uint32('count')
        start_time = header.get_time('start_time') if count else 2**63 - 1
        end_time = header.get_time('end_time') + 1 if count else 0

        _ = self.bio.seek(4, os.SEEK_CUR)

        return ChunkInfo(
            chunk_pos,
            start_time,
            end_time,
            {read_uint32(self.bio): read_uint32(self.bio) for _ in range(count)},
        )

    def read_chunk(self) -> Chunk:
        """Read chunk record header from current position."""
        assert self.bio
        header = Header.read(self.bio, RecordType.CHUNK)
        compression = header.get_string('compression')
        datasize = read_uint32(self.bio)
        datapos = self.bio.tell()
        _ = self.bio.seek(datasize, os.SEEK_CUR)
        try:
            decompressor = decompressors[compression]
        except KeyError:
            msg = f'Compression {compression!r} is not supported.'
            raise ReaderError(msg) from None

        return Chunk(datasize, datapos, decompressor)

    def read_index_data(self, pos: int, indexes: dict[int, list[IndexData]]) -> None:
        """Read index data from position.

        The implementation purposely avoids the generic Header class and
        its costly string processing.

        Args:
            pos: Seek position.
            indexes: Accumulated index data.

        Raises:
            ReaderError: Record unreadable.

        """
        assert self.bio

        buf = self.bio.read(55)
        if not self.index_data_header_offsets:
            (size,) = deserialize_uint32(buf, 0)
            assert size == 47
            idx = 4
            connpos = -1
            countpos = -1
            while idx < size:
                char = buf[idx + 6]
                if char == 61:  # ord(b'=')
                    assert buf[idx + 7] == 4
                    idx += 8
                elif char == 114:  # ord(b'r')
                    if (ver := buf[idx + 8]) != 1:
                        msg = f'IDXDATA version {ver} is not supported.'
                        raise ReaderError(msg)
                    idx += 12
                elif char == 110:  # ord(b'n')
                    connpos = idx + 9
                    idx += 13
                else:
                    assert char == 117  # ord(b'u')
                    countpos = idx + 10
                    idx += 14
            self.index_data_header_offsets = (connpos, countpos)
        connpos, countpos = self.index_data_header_offsets

        (conn,) = deserialize_uint32(buf, connpos)
        (count,) = deserialize_uint32(buf, countpos)
        (size,) = deserialize_uint32(buf, 51)
        assert size == count * 12

        index = indexes[conn]
        buf = self.bio.read(size)
        idx = 0
        while idx < size:
            time = deserialize_uint32(buf, idx)[0] * 10**9 + deserialize_uint32(buf, idx + 4)[0]
            (offset,) = deserialize_uint32(buf, idx + 8)
            idx += 12
            index.append(IndexData(time, pos, offset))

    def messages(
        self,
        connections: Collection[Connection] = (),
        start: int | None = None,
        stop: int | None = None,
    ) -> Generator[tuple[Connection, int, bytes], None, None]:
        """Read messages.

        Args:
            connections: Collection with connections to filter for. An empty
                iterable disables filtering on connections.
            start: Yield only messages at or after this timestamp (ns).
            stop: Yield only messages before this timestamp (ns).

        Yields:
            Tuples of connection, timestamp (ns), and rawdata.

        Raises:
            ReaderError: Bag not open or data corrupt.

        """
        self._check_open()
        assert self.bio
        if not connections:
            connections = self.connections

        connmap = {x.id: x for x in self.connections}

        indexes = [self.indexes[x.id] for x in connections]
        for entry in heapq.merge(*indexes):
            if start is not None and entry.time < start:
                continue
            if stop is not None and entry.time >= stop:
                return

            if self.current_chunk[0] != entry.chunk_pos:
                self.current_chunk[1].close()

                chunk_header = self.chunks[entry.chunk_pos]
                _ = self.bio.seek(chunk_header.datapos)
                rawbytes = chunk_header.decompressor(read_bytes(self.bio, chunk_header.datasize))
                self.current_chunk = (entry.chunk_pos, BytesIO(rawbytes))

            chunk = self.current_chunk[1]
            _ = chunk.seek(entry.offset)

            while True:
                header = Header.read(chunk)
                have = header.get_uint8('op')
                if have != RecordType.CONNECTION:
                    break
                _ = chunk.seek(read_uint32(chunk), os.SEEK_CUR)

            if have != RecordType.MSGDATA:
                msg = 'Expected to find message data.'
                raise ReaderError(msg)

            data = read_bytes(chunk, read_uint32(chunk))
            connection = connmap[header.get_uint32('conn')]
            assert entry.time == header.get_time('time')
            yield connection, entry.time, data

    def __enter__(self) -> Self:
        """Open reader when entering contextmanager."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Close reader when exiting contextmanager."""
        self.close()
        return False
