# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Rosbag1 writer."""

from __future__ import annotations

import struct
import warnings
from bz2 import compress as bz2_compress
from collections import defaultdict
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum, auto
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, cast

from lz4.frame import compress as lz4_compress  # type: ignore[import-untyped]

from rosbags.interfaces import (
    Connection,
    ConnectionExtRosbag1,
    MessageDefinition,
    MessageDefinitionFormat,
)
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.msg import denormalize_msgtype

from .reader import RecordType

if TYPE_CHECKING:
    import sys
    from collections.abc import Callable
    from types import TracebackType
    from typing import BinaryIO, Literal

    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from rosbags.typesys.store import Typestore


class WriterError(Exception):
    """Writer Error."""


@dataclass
class WriteChunk:
    """In progress chunk."""

    data: BytesIO
    pos: int
    start: int
    end: int
    connections: dict[int, list[tuple[int, int]]]


MAXSIZE: int = 2**63 - 1

serialize_uint8 = struct.Struct('<B').pack
serialize_uint32 = struct.Struct('<L').pack
serialize_uint64 = struct.Struct('<Q').pack


def serialize_time(val: int) -> bytes:
    """Serialize time value.

    Args:
        val: Time value.

    Returns:
        Serialized bytes.

    """
    sec, nsec = val // 10**9, val % 10**9
    return struct.pack('<LL', sec, nsec)


class Header(dict[str, bytes]):
    """Record header."""

    def set_uint32(self, name: str, value: int) -> None:
        """Set field to uint32 value.

        Args:
            name: Field name.
            value: Field value.

        """
        self[name] = serialize_uint32(value)

    def set_uint64(self, name: str, value: int) -> None:
        """Set field to uint64 value.

        Args:
            name: Field name.
            value: Field value.

        """
        self[name] = serialize_uint64(value)

    def set_string(self, name: str, value: str) -> None:
        """Set field to string value.

        Args:
            name: Field name.
            value: Field value.

        """
        self[name] = value.encode()

    def set_time(self, name: str, value: int) -> None:
        """Set field to time value.

        Args:
            name: Field name.
            value: Field value.

        """
        self[name] = serialize_time(value)

    def write(self, dst: BinaryIO, opcode: RecordType | None = None) -> int:
        """Write to file handle.

        Args:
            dst: File handle.
            opcode: Record type code.

        Returns:
            Bytes written.

        """
        data = b''

        if opcode:
            keqv = b'op=' + serialize_uint8(opcode)
            data += serialize_uint32(len(keqv)) + keqv

        for key, value in self.items():
            keqv = f'{key}='.encode() + value
            data += serialize_uint32(len(keqv)) + keqv

        size = len(data)
        _ = dst.write(serialize_uint32(size) + data)
        return size + 4


class Writer:
    """Rosbag1 writer.

    This class implements writing of rosbag1 files in version 2.0. It should be
    used as a contextmanager.

    """

    class CompressionFormat(IntEnum):
        """Compession formats."""

        BZ2 = auto()
        LZ4 = auto()

    def __init__(self, path: Path | str) -> None:
        """Initialize writer.

        Args:
            path: Filesystem path to bag.

        Raises:
            WriterError: Target path exists already, Writer can only create new rosbags.

        """
        path = Path(path)
        self.path = path
        if path.exists():
            msg = f'{path} exists already, not overwriting.'
            raise WriterError(msg)
        self.bio: BinaryIO | None = None
        self.compressor: Callable[[bytes], bytes] = lambda x: x
        self.compression_format = 'none'
        self.connections: list[Connection] = []
        self.chunks: list[WriteChunk] = [WriteChunk(BytesIO(), -1, MAXSIZE, 0, defaultdict(list))]
        self.chunk_threshold = 1 * (1 << 20)

    def set_compression(self, fmt: Writer.CompressionFormat) -> None:
        """Enable compression on rosbag1.

        This function has to be called before opening.

        Args:
            fmt: Compressor to use, bz2 or lz4

        Raises:
            WriterError: Bag already open.

        """
        if self.bio:
            msg = f'Cannot set compression, bag {self.path} already open.'
            raise WriterError(msg)

        self.compression_format = fmt.name.lower()

        def bz2(x: bytes) -> bytes:
            return bz2_compress(x, 9)

        def lz4(x: bytes) -> bytes:
            return cast('bytes', lz4_compress(x, 0))

        self.compressor = {'bz2': bz2, 'lz4': lz4}[self.compression_format]

    def open(self) -> None:
        """Open for writing."""
        try:
            self.bio = self.path.open('xb')
        except FileExistsError:
            msg = f'{self.path} exists already, not overwriting.'
            raise WriterError(msg) from None

        assert self.bio
        _ = self.bio.write(b'#ROSBAG V2.0\n')
        header = Header()
        header.set_uint64('index_pos', 0)
        header.set_uint32('conn_count', 0)
        header.set_uint32('chunk_count', 0)
        size = header.write(self.bio, RecordType.BAGHEADER)
        padsize = 4096 - 4 - size
        _ = self.bio.write(serialize_uint32(padsize) + b' ' * padsize)

    def add_connection(
        self,
        topic: str,
        msgtype: str,
        *,
        typestore: Typestore | None = None,
        msgdef: str | None = None,
        md5sum: str | None = None,
        callerid: str | None = None,
        latching: int | None = None,
    ) -> Connection:
        """Add a connection.

        This function can only be called after opening a bag.

        Args:
            topic: Topic name.
            msgtype: Message type.
            typestore: Typestore.
            msgdef: Message definition.
            md5sum: Message hash.
            callerid: Caller id.
            latching: Latching information.

        Returns:
            Connection id.

        Raises:
            WriterError: Bag not open or identical topic previously registered.

        """
        if not self.bio:
            msg = 'Bag was not opened.'
            raise WriterError(msg)

        if msgdef is None or md5sum is None:
            if not typestore:
                warnings.warn(
                    'Writer.add_connection should be called with typestore or msgdef/md5sum pair.',
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                typestore = get_typestore(Stores.ROS1_NOETIC)
            msgdef, md5sum = typestore.generate_msgdef(msgtype)
        assert msgdef is not None
        assert md5sum

        connection = Connection(
            len(self.connections),
            topic,
            msgtype,
            MessageDefinition(MessageDefinitionFormat.MSG, msgdef),
            md5sum,
            -1,
            ConnectionExtRosbag1(callerid, latching),
            self,
        )

        if any(x[1:] == connection[1:] for x in self.connections):
            msg = f'Connections can only be added once with same arguments: {connection!r}.'
            raise WriterError(msg)

        bio = self.chunks[-1].data
        self.write_connection(connection, bio)

        self.connections.append(connection)
        return connection

    def write(self, connection: Connection, timestamp: int, data: bytes | memoryview) -> None:
        """Write message.

        Args:
            connection: Connection to write message to.
            timestamp: Message timestamp (ns).
            data: Serialized message data.

        Raises:
            WriterError: Bag not open or connection not registered.

        """
        if not self.bio:
            msg = 'Bag was not opened.'
            raise WriterError(msg)

        if connection not in self.connections:
            msg = f'There is no connection {connection!r}.'
            raise WriterError(msg) from None

        chunk = self.chunks[-1]
        chunk.connections[connection.id].append((timestamp, chunk.data.tell()))
        chunk.start = min(timestamp, chunk.start)
        chunk.end = max(timestamp, chunk.end)

        header = Header()
        header.set_uint32('conn', connection.id)
        header.set_time('time', timestamp)

        _ = header.write(chunk.data, RecordType.MSGDATA)
        _ = chunk.data.write(serialize_uint32(len(data)))
        _ = chunk.data.write(data)
        if chunk.data.tell() > self.chunk_threshold:
            self.write_chunk(chunk)

    @staticmethod
    def write_connection(connection: Connection, bio: BinaryIO) -> None:
        """Write connection record."""
        header = Header()
        header.set_uint32('conn', connection.id)
        header.set_string('topic', connection.topic)
        _ = header.write(bio, RecordType.CONNECTION)

        header = Header()
        header.set_string('topic', connection.topic)
        header.set_string('type', denormalize_msgtype(connection.msgtype))
        header.set_string('md5sum', connection.digest)
        header.set_string('message_definition', connection.msgdef.data)
        assert isinstance(connection.ext, ConnectionExtRosbag1)
        if connection.ext.callerid is not None:
            header.set_string('callerid', connection.ext.callerid)
        if connection.ext.latching is not None:
            header.set_string('latching', str(connection.ext.latching))
        _ = header.write(bio)

    def write_chunk(self, chunk: WriteChunk) -> None:
        """Write open chunk to file."""
        assert self.bio

        if (size := chunk.data.tell()) > 0:
            chunk.pos = self.bio.tell()

            header = Header()
            header.set_string('compression', self.compression_format)
            header.set_uint32('size', size)
            _ = header.write(self.bio, RecordType.CHUNK)
            data = self.compressor(chunk.data.getvalue())
            _ = self.bio.write(serialize_uint32(len(data)))
            _ = self.bio.write(data)

            for cid, items in chunk.connections.items():
                header = Header()
                header.set_uint32('ver', 1)
                header.set_uint32('conn', cid)
                header.set_uint32('count', len(items))
                _ = header.write(self.bio, RecordType.IDXDATA)
                _ = self.bio.write(serialize_uint32(len(items) * 12))
                for time, offset in items:
                    _ = self.bio.write(serialize_time(time) + serialize_uint32(offset))

            chunk.data.close()
            self.chunks.append(WriteChunk(BytesIO(), -1, MAXSIZE, 0, defaultdict(list)))

    def close(self) -> None:
        """Close after writing.

        Closes open chunks and writes index.

        """
        assert self.bio
        for chunk in self.chunks:
            if chunk.pos == -1:
                self.write_chunk(chunk)

        index_pos = self.bio.tell()

        for connection in self.connections:
            self.write_connection(connection, self.bio)

        for chunk in self.chunks:
            if chunk.pos == -1:
                continue
            header = Header()
            header.set_uint32('ver', 1)
            header.set_uint64('chunk_pos', chunk.pos)
            header.set_time('start_time', 0 if chunk.start == MAXSIZE else chunk.start)
            header.set_time('end_time', chunk.end)
            header.set_uint32('count', len(chunk.connections))
            _ = header.write(self.bio, RecordType.CHUNK_INFO)
            _ = self.bio.write(serialize_uint32(len(chunk.connections) * 8))
            for cid, items in chunk.connections.items():
                _ = self.bio.write(serialize_uint32(cid) + serialize_uint32(len(items)))

        _ = self.bio.seek(13)
        header = Header()
        header.set_uint64('index_pos', index_pos)
        header.set_uint32('conn_count', len(self.connections))
        header.set_uint32('chunk_count', len([x for x in self.chunks if x.pos != -1]))
        size = header.write(self.bio, RecordType.BAGHEADER)
        padsize = 4096 - 4 - size
        _ = self.bio.write(serialize_uint32(padsize) + b' ' * padsize)

        self.bio.close()
        self.bio = None

    def abort(self) -> None:
        """Close without writing index."""
        bio, self.bio = self.bio, None
        if bio is not None:
            bio.close()

    def __enter__(self) -> Self:
        """Open writer when entering contextmanager."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        """Close or abort writer when exiting contextmanager."""
        if exc_type is None:
            try:
                self.close()
            except BaseException:
                with suppress(Exception):
                    self.abort()
                raise
        else:
            with suppress(Exception):
                self.abort()
        return False
