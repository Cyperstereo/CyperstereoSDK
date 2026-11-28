# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Shared interfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Generic, NamedTuple, Protocol, TypeAlias, TypeVar

if TYPE_CHECKING:
    from rosbags.interfaces.typing import Typesdict

    from .typing import Fielddefs

T = TypeVar('T')


class ConnectionExtRosbag1(NamedTuple):
    """Rosbag1 specific connection extensions."""

    callerid: str | None
    latching: int | None


class ConnectionExtRosbag2(NamedTuple):
    """Rosbag2 specific connection extensions."""

    serialization_format: str
    offered_qos_profiles: list[Qos]


class Connection(NamedTuple):
    """Connection information."""

    id: int
    topic: str
    msgtype: str
    msgdef: MessageDefinition
    digest: str
    msgcount: int
    ext: ConnectionExtRosbag1 | ConnectionExtRosbag2
    owner: object


class MessageDefinitionFormat(Enum):
    """Message definition format."""

    NONE = 0
    MSG = 1
    IDL = 2


class MessageDefinition(NamedTuple):
    """Message definition."""

    format: MessageDefinitionFormat
    data: str


class QosDurability(Enum):
    """QoS durability parameter."""

    SYSTEM_DEFAULT = 0
    TRANSIENT_LOCAL = 1
    VOLATILE = 2
    UNKNOWN = 3
    BEST_AVAILABLE = 4


class QosHistory(Enum):
    """QoS history parameter."""

    SYSTEM_DEFAULT = 0
    KEEP_LAST = 1
    KEEP_ALL = 2
    UNKNOWN = 3


class QosLiveliness(Enum):
    """QoS liveliness parameter."""

    SYSTEM_DEFAULT = 0
    AUTOMATIC = 1
    MANUAL_BY_NODE = 2
    MANUAL_BY_TOPIC = 3
    UNKNOWN = 4
    BEST_AVAILABLE = 5


class QosReliability(Enum):
    """QoS reliability parameter."""

    SYSTEM_DEFAULT = 0
    RELIABLE = 1
    BEST_EFFORT = 2
    UNKNOWN = 3
    BEST_AVAILABLE = 4


class QosTime(NamedTuple):
    """Time in seconds and nanoseconds."""

    sec: int
    nsec: int


class Qos(NamedTuple):
    """QoS parameters."""

    history: QosHistory
    depth: int
    reliability: QosReliability
    durability: QosDurability
    deadline: QosTime
    lifespan: QosTime
    liveliness: QosLiveliness
    liveliness_lease_duration: QosTime
    avoid_ros_namespace_conventions: bool


class TopicInfo(NamedTuple):
    """Topic information."""

    msgtype: str | None
    msgdef: MessageDefinition
    msgcount: int
    connections: list[Connection]


class Typestore(Protocol):
    """Type storage."""

    fielddefs: Typesdict
    max_sequence_length: int

    def get_msgdef(self, typename: str) -> Msgdef[object]:
        """Get message definition."""
        raise NotImplementedError  # pragma: no cover


Bitcvt: TypeAlias = Callable[[bytes | memoryview, int, memoryview, int, Typestore], tuple[int, int]]
BitcvtSize: TypeAlias = Callable[[bytes | memoryview, int, None, int, Typestore], tuple[int, int]]

CDRDeser: TypeAlias = Callable[[bytes | memoryview, int, type, Typestore], tuple[object, int]]
CDRSer: TypeAlias = Callable[[memoryview, int, object, Typestore], int]
CDRSerSize: TypeAlias = Callable[[int, object, Typestore], int]


@dataclass(frozen=True)
class Msgdef(Generic[T]):
    """Metadata of a message."""

    name: str
    fields: Fielddefs
    cls: type[object]
    size_cdr: int
    getsize_cdr: CDRSerSize
    serialize_cdr_le: CDRSer
    serialize_cdr_be: CDRSer
    deserialize_cdr_le: CDRDeser
    deserialize_cdr_be: CDRDeser
    size_ros1: int
    getsize_ros1: CDRSerSize
    serialize_ros1: CDRSer
    deserialize_ros1: CDRDeser
    getsize_ros1_to_cdr: BitcvtSize
    ros1_to_cdr: Bitcvt
    getsize_cdr_to_ros1: BitcvtSize
    cdr_to_ros1: Bitcvt


class Nodetype(IntEnum):
    """Parse tree node types.

    The first four match the Valtypes of final message definitions.
    """

    BASE = auto()
    NAME = auto()
    ARRAY = auto()
    SEQUENCE = auto()
