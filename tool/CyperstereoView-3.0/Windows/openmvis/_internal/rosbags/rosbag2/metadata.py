# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Rosbag2 metadata."""

from __future__ import annotations

from io import StringIO
from typing import TYPE_CHECKING, NamedTuple, TypedDict, cast

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from rosbags.interfaces import (
    Qos,
    QosDurability,
    QosHistory,
    QosLiveliness,
    QosReliability,
    QosTime,
)

from .errors import ReaderError

if TYPE_CHECKING:

    class QosTimeDict(TypedDict):
        """Time in seconds and nanoseconds."""

        sec: int
        nsec: int

    class QosDict(TypedDict):
        """QoS parameters."""

        history: str | int
        depth: int
        reliability: str | int
        durability: str | int
        deadline: QosTimeDict
        lifespan: QosTimeDict
        liveliness: str | int
        liveliness_lease_duration: QosTimeDict
        avoid_ros_namespace_conventions: bool


class Duration(TypedDict):
    """Bag duration."""

    nanoseconds: int


class StartingTime(TypedDict):
    """Bag starting time."""

    nanoseconds_since_epoch: int


class TopicMetadata(TypedDict):
    """Topic metadata."""

    name: str
    type: str
    serialization_format: str
    offered_qos_profiles: list[QosDict] | str
    type_description_hash: str


class TopicWithMessageCount(TypedDict):
    """Topic with message count."""

    message_count: int
    topic_metadata: TopicMetadata


class FileInformation(TypedDict):
    """Per file metadata."""

    path: str
    starting_time: StartingTime
    duration: Duration
    message_count: int


class Metadata(TypedDict):
    """Rosbag2 metadata file."""

    version: int
    storage_identifier: str
    relative_file_paths: list[str]
    starting_time: StartingTime
    duration: Duration
    message_count: int
    compression_format: str
    compression_mode: str
    topics_with_message_count: list[TopicWithMessageCount]
    files: list[FileInformation]
    custom_data: dict[str, str] | None
    ros_distro: str


class ReaderMetadata(NamedTuple):
    """Storage Metadata."""

    duration: int
    start_time: int
    end_time: int
    message_count: int
    compression_format: str | None
    compression_mode: str | None
    ros_distro: str | None
    custom: dict[str, str] | None


def parse_qos(dcts: list[QosDict] | str) -> list[Qos]:
    """Parse and normalize QoS parameters."""
    if not dcts:
        return []
    try:
        items = (
            cast('list[QosDict]', YAML(typ='safe').load(dcts)) if isinstance(dcts, str) else dcts
        )
    except YAMLError as exc:
        msg = f'Could not load YAML from offered_qos_profiles: {exc}'
        raise ReaderError(msg) from None

    res: list[Qos] = []
    for item in items:
        try:
            if isinstance((value := item['history']), int):
                history = QosHistory(value)
            else:
                history = QosHistory.__members__[value.upper()]
        except (KeyError, ValueError):
            history = QosHistory(QosHistory.UNKNOWN)

        try:
            if isinstance((value := item['reliability']), int):
                reliability = QosReliability(value)
            else:
                reliability = QosReliability.__members__[value.upper()]
        except (KeyError, ValueError):
            reliability = QosReliability(QosReliability.UNKNOWN)

        try:
            if isinstance((value := item['durability']), int):
                durability = QosDurability(value)
            else:
                durability = QosDurability.__members__[value.upper()]
        except (KeyError, ValueError):
            durability = QosDurability(QosDurability.UNKNOWN)

        try:
            if isinstance((value := item['liveliness']), int):
                liveliness = QosLiveliness(value)
            else:
                liveliness = QosLiveliness.__members__[value.upper()]
        except (KeyError, ValueError):
            liveliness = QosLiveliness(QosLiveliness.UNKNOWN)

        res.append(
            Qos(
                history,
                item['depth'],
                reliability,
                durability,
                QosTime(item['deadline']['sec'], item['deadline']['nsec']),
                QosTime(item['lifespan']['sec'], item['lifespan']['nsec']),
                liveliness,
                QosTime(
                    item['liveliness_lease_duration']['sec'],
                    item['liveliness_lease_duration']['nsec'],
                ),
                item['avoid_ros_namespace_conventions'],
            ),
        )

    return res


def dump_qos_v8(qos: list[Qos]) -> str:
    """Dump qos for v8 metadata."""
    dcts: list[QosDict] = [
        {
            'history': x.history.value,
            'depth': x.depth,
            'reliability': x.reliability.value,
            'durability': x.durability.value,
            'deadline': {'sec': x.deadline.sec, 'nsec': x.deadline.nsec},
            'lifespan': {'sec': x.lifespan.sec, 'nsec': x.lifespan.nsec},
            'liveliness': x.liveliness.value,
            'liveliness_lease_duration': {
                'sec': x.liveliness_lease_duration.sec,
                'nsec': x.liveliness_lease_duration.nsec,
            },
            'avoid_ros_namespace_conventions': x.avoid_ros_namespace_conventions,
        }
        for x in qos
    ]

    if not dcts:
        return ''

    stream = StringIO()
    YAML(typ='safe').dump(dcts, stream)
    return stream.getvalue().strip()


def dump_qos_v9(qos: list[Qos]) -> list[QosDict]:
    """Dump qos for v9 metadata."""
    return [
        {
            'history': x.history.name.lower(),
            'depth': x.depth,
            'reliability': x.reliability.name.lower(),
            'durability': x.durability.name.lower(),
            'deadline': {'sec': x.deadline.sec, 'nsec': x.deadline.nsec},
            'lifespan': {'sec': x.lifespan.sec, 'nsec': x.lifespan.nsec},
            'liveliness': x.liveliness.name.lower(),
            'liveliness_lease_duration': {
                'sec': x.liveliness_lease_duration.sec,
                'nsec': x.liveliness_lease_duration.nsec,
            },
            'avoid_ros_namespace_conventions': x.avoid_ros_namespace_conventions,
        }
        for x in qos
    ]
