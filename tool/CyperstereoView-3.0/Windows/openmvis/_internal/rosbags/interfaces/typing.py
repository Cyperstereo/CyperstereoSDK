# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Rosbags typing."""

import sys
from pathlib import PurePath
from typing import Any, BinaryIO, Literal, Protocol, TypeAlias, TypeVar

from rosbags.interfaces import Nodetype

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing_extensions import Self

T = TypeVar('T')


Basename: TypeAlias = Literal[
    'bool',
    'byte',
    'char',
    'fixed',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',
    'float32',
    'float64',
    'float128',
    'string',
    'wchar',
    'wstring',
]
Basetype: TypeAlias = tuple[Basename, int]

BaseDesc: TypeAlias = tuple[Literal[Nodetype.BASE], Basetype]
NameDesc: TypeAlias = tuple[Literal[Nodetype.NAME], str]
FieldDesc: TypeAlias = (
    BaseDesc
    | NameDesc
    | tuple[Literal[Nodetype.ARRAY, Nodetype.SEQUENCE], tuple[BaseDesc | NameDesc, int]]
)

ConstValue: TypeAlias = str | bool | int | float
ScalarValue: TypeAlias = float | int | bool | str
Value: TypeAlias = ScalarValue | list[ScalarValue]

Constdefs: TypeAlias = list[tuple[str, Basename, ConstValue]]
Fielddefs: TypeAlias = list[tuple[str, FieldDesc]]
Typesdict: TypeAlias = dict[str, tuple[Constdefs, Fielddefs]]


class StatResult(Protocol):  # pragma: no cover
    """Start result protocol."""

    @property
    def st_size(self) -> int:
        """Proxy."""
        raise NotImplementedError


class RPath(Protocol):  # pragma: no cover
    """Reader path protocol."""

    def exists(self) -> bool:
        """Proxy."""
        ...

    def is_dir(self) -> bool:
        """Proxy."""
        ...

    def open(  # type: ignore[explicit-any]
        self,
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> BinaryIO:
        """Proxy."""
        ...

    def read_text(
        self,
        encoding: str | None = None,
    ) -> str:
        """Proxy."""
        ...

    def stat(self, *, follow_symlinks: bool = True) -> StatResult:
        """Proxy."""
        ...

    def __truediv__(self, key: Self | PurePath | str) -> Self:
        """Proxy."""
        ...

    @property
    def stem(self) -> str:
        """Proxy."""
        raise NotImplementedError

    @property
    def suffix(self) -> str:
        """Proxy."""
        ...
