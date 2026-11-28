# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
#
# THIS FILE IS GENERATED, DO NOT EDIT
"""Message type definitions."""

# ruff: noqa: N801,N816,RUF100,TC004

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rosbags.interfaces import Nodetype as T

if TYPE_CHECKING:
    from typing import ClassVar

    from rosbags.interfaces.typing import Typesdict


@dataclass
class builtin_interfaces__msg__Duration:
    """Class for builtin_interfaces/msg/Duration."""

    sec: int
    nanosec: int
    __msgtype__: ClassVar[str] = 'builtin_interfaces/msg/Duration'


@dataclass
class builtin_interfaces__msg__Time:
    """Class for builtin_interfaces/msg/Time."""

    sec: int
    nanosec: int
    __msgtype__: ClassVar[str] = 'builtin_interfaces/msg/Time'


FIELDDEFS: Typesdict = {
    'builtin_interfaces/msg/Duration': (
        [],
        [
            ('sec', (T.BASE, ('int32', 0))),
            ('nanosec', (T.BASE, ('uint32', 0))),
        ],
    ),
    'builtin_interfaces/msg/Time': (
        [],
        [
            ('sec', (T.BASE, ('int32', 0))),
            ('nanosec', (T.BASE, ('uint32', 0))),
        ],
    ),
}
