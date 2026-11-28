# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Sqlite3 storage."""

from __future__ import annotations

import sqlite3
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, cast

import apsw

from rosbags.interfaces import (
    Connection,
    ConnectionExtRosbag2,
    MessageDefinition,
    MessageDefinitionFormat,
)

from .enums import CompressionMode
from .errors import ReaderError, WriterError
from .metadata import ReaderMetadata, parse_qos

if sys.version_info >= (3, 12):
    from typing import override
else:  # pragma: no cover
    from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Collection, Generator

    from rosbags.interfaces.typing import RPath


def make_vfs(rpath: RPath) -> apsw.VFS:  # pragma: no cover
    """Create sqlite vfs driver for path."""

    class VFS(apsw.VFS):
        class VFSFile(apsw.VFSFile):
            def __init__(
                self,
                _vfs: VFS,
                _path: str | apsw.URIFilename | None,
                _flags: list[int],
            ) -> None:
                self._path = rpath.open('rb')

            @override
            def xDeviceCharacteristics(self) -> int:
                return 0

            @override
            def xSectorSize(self) -> int:
                return 0

            @override
            def xFileControl(self, op: int, ptr: int) -> bool:
                if op in (14, 15, 30):
                    return False
                return super().xFileControl(op, ptr)

            @override
            def xFileSize(self) -> int:
                return rpath.stat().st_size

            @override
            def xRead(self, amount: int, offset: int) -> bytes:
                self._path.seek(offset)
                return self._path.read(amount)

            @override
            def xClose(self) -> None:
                self._path.close()

        @override
        def __init__(self, name: str = 'rpathvfs', *_args: object, **_kwargs: object) -> None:
            super().__init__(name, '')

        @override
        def xOpen(
            self,
            name: str | apsw.URIFilename | None,
            flags: list[int],
        ) -> apsw.VFSFile:
            if name is None:
                return super().xOpen(name, flags)

            write_flags = {
                apsw.SQLITE_OPEN_READWRITE,
                apsw.SQLITE_OPEN_CREATE,
                apsw.SQLITE_OPEN_DELETEONCLOSE,
                apsw.SQLITE_OPEN_EXCLUSIVE,
                apsw.SQLITE_OPEN_AUTOPROXY,
                apsw.SQLITE_OPEN_WAL,
                apsw.SQLITE_OPEN_SUPER_JOURNAL,
            }
            if any(x in write_flags for x in flags):
                msg = 'Only read-only access supported'
                raise RuntimeError(msg)

            return self.VFSFile(self, name, flags)

    return VFS()


class Sqlite3Reader:
    """Sqlite3 storage reader."""

    def __init__(self, path: RPath) -> None:
        """Set up storage reader.

        Args:
            path: Paths of storage files.

        """
        self.path = path
        self.dbconn: apsw.Connection | None = None
        self.schema = 0
        self.msgtypes: list[dict[str, str]] = []
        self.connections: list[Connection] = []
        self.metadata = ReaderMetadata(0, 2**63 - 1, 0, 0, None, None, None, None)

    def _open(self) -> None:
        """Detect schema and populate state."""
        if isinstance(self.path, Path):
            vfs = None
        else:  # pragma: no cover
            vfs = 'rpathvfs'
            _vfs = make_vfs(self.path)

        conn = apsw.Connection(
            f'file:{self.path}?immutable=1',
            flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI,
            vfs=vfs,
        )

        cur = conn.cursor()
        _ = cur.execute(
            (
                'SELECT count(*) FROM sqlite_master '
                'WHERE type="table" AND name IN ("messages", "topics")'
            ),
        )
        if (x := cur.fetchone()) is None or x[0] != 2:
            msg = f'Cannot open database {self.path} or database missing tables.'
            raise ReaderError(msg)

        cur = conn.cursor()
        if cur.execute('PRAGMA table_info(schema)').fetchall():
            (schema,) = cast(
                'tuple[int] | None',
                cur.execute('SELECT schema_version FROM schema').fetchone(),
            ) or (-1,)
        elif any(
            x[1] == 'offered_qos_profiles'
            for x in cast('Collection[tuple[str, str]]', cur.execute('PRAGMA table_info(topics)'))
        ):
            schema = 2
        else:
            schema = 1

        if schema >= 4:
            msgtypes: list[dict[str, str]] = [
                {
                    'name': x[0],
                    'encoding': x[1],
                    'msgdef': x[2],
                    'digest': x[3],
                }
                for x in cast(
                    'Collection[tuple[str, str, str, str]]',
                    cur.execute(
                        (
                            'SELECT topic_type, encoding, encoded_message_definition,'
                            ' type_description_hash FROM message_definitions ORDER BY id'
                        ),
                    ),
                )
            ]
        else:
            msgtypes = []

        def get_msgdef(name: str) -> MessageDefinition:
            """Get message definition for name."""
            fmtmap = {
                'ros2msg': MessageDefinitionFormat.MSG,
                'ros2idl': MessageDefinitionFormat.IDL,
            }
            if msgtype := next((x for x in msgtypes if x['name'] == name), None):
                return MessageDefinition(fmtmap[msgtype['encoding']], msgtype['msgdef'])
            return MessageDefinition(MessageDefinitionFormat.NONE, '')

        if schema >= 4:
            connections = [
                Connection(
                    cid,
                    topic,
                    msgtype,
                    get_msgdef(msgtype),
                    digest,
                    msgcount,
                    ConnectionExtRosbag2(serialization_format, parse_qos(offered_qos_profiles)),
                    self,
                )
                for (
                    cid,
                    topic,
                    msgtype,
                    msgcount,
                    serialization_format,
                    offered_qos_profiles,
                    digest,
                ) in cast(
                    'Collection[tuple[int, str, str, int, str, str, str]]',
                    cur.execute(
                        (
                            'SELECT topics.id, name, type, count(messages.id), '
                            'serialization_format, offered_qos_profiles, type_description_hash '
                            'FROM topics LEFT JOIN messages ON topics.id == messages.topic_id '
                            'GROUP BY topics.id ORDER BY topics.id'
                        ),
                    ),
                )
            ]
        elif schema >= 2:
            connections = [
                Connection(
                    cid,
                    topic,
                    msgtype,
                    get_msgdef(msgtype),
                    '',
                    msgcount,
                    ConnectionExtRosbag2(serialization_format, parse_qos(offered_qos_profiles)),
                    self,
                )
                for (
                    cid,
                    topic,
                    msgtype,
                    msgcount,
                    serialization_format,
                    offered_qos_profiles,
                ) in cast(
                    'Collection[tuple[int, str, str, int, str, str]]',
                    cur.execute(
                        (
                            'SELECT topics.id, name, type, count(messages.id), '
                            'serialization_format, offered_qos_profiles '
                            'FROM topics LEFT JOIN messages ON topics.id == messages.topic_id '
                            'GROUP BY topics.id ORDER BY topics.id'
                        ),
                    ),
                )
            ]
        else:
            connections = [
                Connection(
                    cid,
                    topic,
                    msgtype,
                    get_msgdef(msgtype),
                    '',
                    msgcount,
                    ConnectionExtRosbag2(serialization_format, []),
                    self,
                )
                for (
                    cid,
                    topic,
                    msgtype,
                    msgcount,
                    serialization_format,
                ) in cast(
                    'Collection[tuple[int, str, str, int, str]]',
                    cur.execute(
                        (
                            'SELECT topics.id, name, type, count(messages.id), '
                            'serialization_format '
                            'FROM topics LEFT JOIN messages ON topics.id == messages.topic_id '
                            'GROUP BY topics.id ORDER BY topics.id'
                        ),
                    ),
                )
            ]

        ((start_time, end_time, msgcount),) = cast(
            'Collection[tuple[int , int , int] | tuple[None, None, int]]',
            cur.execute(
                'SELECT MIN(timestamp), MAX(timestamp) + 1, COUNT(*) FROM messages',
            ),
        )

        self.schema = schema
        self.msgtypes = msgtypes
        self.connections = connections
        self.metadata = ReaderMetadata(
            end_time - start_time if start_time is not None and end_time is not None else 0,
            start_time if start_time is not None else 2**63 - 1,
            end_time if end_time is not None else 0,
            msgcount,
            None,
            None,
            None,
            None,
        )

    def open(self) -> None:
        """Open storage."""
        if isinstance(self.path, Path):
            vfs = None
        else:  # pragma: no cover
            vfs = 'rpathvfs'
            _vfs = make_vfs(self.path)

        try:
            self.dbconn = apsw.Connection(
                f'file:{self.path}?immutable=1',
                flags=apsw.SQLITE_OPEN_READONLY | apsw.SQLITE_OPEN_URI,
                vfs=vfs,
            )
            self._open()
        except BaseException as err:
            if self.dbconn:
                with suppress(Exception):
                    self.close()
            if isinstance(err, apsw.Error):
                msg = f'Cannot open database {self.path}: {err}'
                raise ReaderError(msg) from err
            raise

    def close(self) -> None:
        """Close storage."""
        dbconn, self.dbconn = self.dbconn, None
        assert dbconn

        self.schema = 0
        self.msgtypes.clear()
        self.connections.clear()
        self.metadata = ReaderMetadata(0, 2**63 - 1, 0, 0, None, None, None, None)
        dbconn.close()

    def messages(
        self,
        connections: Collection[Connection],
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

        """
        assert self.dbconn

        query = [
            'SELECT topics.id,messages.timestamp,messages.data',
            'FROM messages JOIN topics ON messages.topic_id=topics.id',
        ]
        args: list[apsw.Binding] = []
        clause = 'WHERE'

        ids = tuple({x.id for x in connections})
        query.append(f'{clause} topics.id IN ({",".join("?" for _ in ids)})')
        args += ids
        clause = 'AND'

        if start is not None:
            query.append(f'{clause} messages.timestamp >= ?')
            args.append(start)
            clause = 'AND'

        if stop is not None:
            query.append(f'{clause} messages.timestamp < ?')
            args.append(stop)
            clause = 'AND'

        query.append('ORDER BY timestamp')
        querystr = ' '.join(query)

        connmap = {x.id: x for x in self.connections}

        cur = cast('Collection[tuple[int, int, bytes]]', self.dbconn.execute(querystr, args))

        for cid, timestamp, data in cur:
            yield connmap[cid], timestamp, data


class Sqlite3Writer:
    """Sqlite3 Storage Writer."""

    SQLITE_SCHEMA = """
    CREATE TABLE schema(
      schema_version INTEGER PRIMARY KEY,
      ros_distro TEXT NOT NULL
    );
    CREATE TABLE metadata(
      id INTEGER PRIMARY KEY,
      metadata_version INTEGER NOT NULL,
      metadata TEXT NOT NULL
    );
    CREATE TABLE topics(
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      type TEXT NOT NULL,
      serialization_format TEXT NOT NULL,
      offered_qos_profiles TEXT NOT NULL,
      type_description_hash TEXT NOT NULL
    );
    CREATE TABLE message_definitions(
      id INTEGER PRIMARY KEY,
      topic_type TEXT NOT NULL,
      encoding TEXT NOT NULL,
      encoded_message_definition TEXT NOT NULL,
      type_description_hash TEXT NOT NULL
    );
    CREATE TABLE messages(
      id INTEGER PRIMARY KEY,
      topic_id INTEGER NOT NULL,
      timestamp INTEGER NOT NULL,
      data BLOB NOT NULL
    );
    CREATE INDEX timestamp_idx ON messages (timestamp ASC);
    INSERT INTO schema(schema_version, ros_distro) VALUES (4, 'rosbags');
    """

    def __init__(self, path: Path, compression: CompressionMode) -> None:
        """Initialize sqlite3 storage."""
        if compression == CompressionMode.STORAGE:
            msg = 'SQLITE3 writer does not support storage-side compression.'
            raise WriterError(msg)

        self.path = path / f'{path.name}.db3'
        self.conn = sqlite3.connect(f'file:{self.path}', uri=True)
        _ = self.conn.executescript(self.SQLITE_SCHEMA)
        self.cursor = self.conn.cursor()

    def add_msgtype(self, connection: Connection) -> None:
        """Add msgtype.

        Args:
            connection: Connection.

        """
        _ = self.cursor.execute(
            (
                'INSERT INTO message_definitions (topic_type, encoding,'
                ' encoded_message_definition, type_description_hash) VALUES(?, ?, ?, ?)'
            ),
            (
                connection.msgtype,
                'ros2msg' if connection.msgdef.format == MessageDefinitionFormat.MSG else 'ros2idl',
                connection.msgdef.data,
                connection.digest,
            ),
        )

    def add_connection(self, connection: Connection, offered_qos_profiles: str) -> None:
        """Add connection.

        Args:
            connection: Connection.
            offered_qos_profiles: Serialized QoS profiles.

        """
        _ = self.cursor.execute(
            'INSERT INTO topics VALUES(?, ?, ?, ?, ?, ?)',
            (
                connection.id,
                connection.topic,
                connection.msgtype,
                cast('ConnectionExtRosbag2', connection.ext).serialization_format,
                offered_qos_profiles,
                connection.digest,
            ),
        )

    def write(self, connection: Connection, timestamp: int, data: bytes | memoryview) -> None:
        """Write message.

        Args:
            connection: Connection to write message to.
            timestamp: Message timestamp (ns).
            data: Serialized message data.

        """
        _ = self.cursor.execute(
            'INSERT INTO messages (topic_id, timestamp, data) VALUES(?, ?, ?)',
            (connection.id, timestamp, data),
        )

    def close(self, version: int, metadata: str) -> None:
        """Close after writing.

        Closes open database transactions and writes metadata.yaml.

        """
        self.cursor.execute(
            'INSERT INTO metadata(metadata_version, metadata) VALUES(?, ?)',
            (version, metadata),
        )

        self.conn.commit()
        _ = self.conn.execute('PRAGMA optimize')
        self.conn.close()

    def abort(self) -> None:
        """Close without committing pending changes."""
        try:
            self.conn.rollback()
        finally:
            self.conn.close()
