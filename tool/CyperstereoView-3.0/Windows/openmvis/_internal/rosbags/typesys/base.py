# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""Types and helpers used by message definition converters."""

from __future__ import annotations

import keyword
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

from rosbags.interfaces import Nodetype

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rosbags.interfaces.typing import (
        BaseDesc,
        FieldDesc,
        NameDesc,
        ScalarValue,
        Typesdict,
        Value,
    )


class TypesysError(Exception):
    """Parser error."""


class BaseName(Enum):
    """Base type."""

    BOOL = 'bool'
    BYTE = 'byte'
    CHAR = 'char'
    FIXED = 'fixed'
    INT8 = 'int8'
    INT16 = 'int16'
    INT32 = 'int32'
    INT64 = 'int64'
    UINT8 = 'uint8'
    UINT16 = 'uint16'
    UINT32 = 'uint32'
    UINT64 = 'uint64'
    FLOAT32 = 'float32'
    FLOAT64 = 'float64'
    FLOAT128 = 'float128'
    STRING = 'string'
    WCHAR = 'wchar'
    WSTRING = 'wstring'


BASE_NAMES = {t.value: t for t in BaseName}


class Cardinality(Enum):
    """Type cardinality."""

    SCALAR = 'scalar'
    ARRAY = 'array'
    SEQUENCE = 'sequence'


class BaseType(NamedTuple):
    """Base type."""

    name: BaseName


class NamedType(NamedTuple):
    """Named type."""

    name: str


TypeRef = BaseType | NamedType


class Type(NamedTuple):
    """Named type."""

    ref: TypeRef
    cardinality: Cardinality = Cardinality.SCALAR
    str_size: int = 0
    size: int = 0


class Annotation(NamedTuple):
    """Annotation."""

    name: str


class Constant(NamedTuple):
    """Annotation."""

    typ: BaseName
    name: str
    value: ScalarValue


class Field(NamedTuple):
    """Field."""

    typ: Type
    name: str
    value: Value | None


class Message(NamedTuple):
    """Message."""

    name: str
    constants: list[Constant]
    fields: list[Field]


def normalize_fieldname(name: str) -> str:
    """Normalize field name.

    Avoid collisions with Python keywords.

    Args:
        name: Field name.

    Returns:
        Normalized name.

    """
    if keyword.iskeyword(name):
        return f'{name}_'
    return name


def create_field_desc(typ: Type) -> FieldDesc:
    """Create field type descriptor."""
    inner: BaseDesc | NameDesc = (
        (Nodetype.BASE, (typ.ref.name.value, typ.str_size))
        if isinstance(typ.ref, BaseType)
        else (Nodetype.NAME, typ.ref.name)
    )
    if typ.cardinality == Cardinality.SCALAR:
        return inner

    return (
        Nodetype.ARRAY if typ.cardinality == Cardinality.ARRAY else Nodetype.SEQUENCE,
        (inner, typ.size),
    )


def make_typesdict(msgs: Sequence[Message]) -> Typesdict:
    """Make typesdict from messages."""
    return {
        x.name: (
            [(y.name, y.typ.value, y.value) for y in x.constants],
            [(y.name, create_field_desc(y.typ)) for y in x.fields],
        )
        for x in msgs
    }
