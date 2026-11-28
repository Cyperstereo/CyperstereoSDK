# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""MSG Parser.

Parser for the ROS1 and ROS2 `MSG`_ message definition format. It also
supports concatenated message definitions as found in Rosbag1 connection
information.

.. _MSG: http://wiki.ros.org/msg

"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from .base import (
    BASE_NAMES,
    Annotation,
    BaseType,
    Cardinality,
    Constant,
    Field,
    Message,
    NamedType,
    Type,
    TypesysError,
    make_typesdict,
)
from .peg import Failure, Parser, Success

if TYPE_CHECKING:
    from rosbags.interfaces.typing import (
        ScalarValue,
        Typesdict,
        Value,
    )

    from .peg import Result


def normalize_msgtype(name: str) -> str:
    """Normalize message name.

    Args:
        name: Message name.

    Returns:
        Message name in ROS2 style.

    """
    path = PurePosixPath(name)
    if path.parent.name not in ('msg', 'action'):
        path = path.parent / 'msg' / path.name
    return str(path)


def denormalize_msgtype(name: str) -> str:
    """Undo message name normalization.

    Args:
        name: Normalized message name.

    Returns:
        Message name in ROS1 style.

    """
    assert '/msg/' in name
    return str((path := PurePosixPath(name)).parent.parent / path.name)


def normalize_type_ref(typename: str, typ: Type) -> Type:
    """Normalize type reference.

    Args:
        typename: Type name of field owner.
        typ: Type.

    Returns:
        Normalized type reference.

    """
    if isinstance(typ.ref, BaseType):
        return typ

    name = typ.ref.name
    if '/' not in name:
        name = str(PurePosixPath(typename).parent / name)
    elif '/msg/' not in name and '/action/' not in name:
        name = str((path := PurePosixPath(name)).parent / 'msg' / path.name)

    return Type(NamedType(name), str_size=typ.str_size, cardinality=typ.cardinality, size=typ.size)


class MSGParser(Parser):
    """ROS MSG Parser."""

    Separator = '================================================================================'

    MixedSnake = r'[A-Za-z][A-Za-z0-9_]*'
    LowerSnake = r'[a-z](?:_?[a-z0-9]+)*'
    MixedCamel = r'[A-Za-z](?:_?[A-Za-z0-9]+)*'
    UpperCamel = r'[A-Z](?:_?[A-Za-z0-9]+)*'
    UpperSnake = r'[A-Z](?:_?[A-Z0-9]+)*'

    # ROS2 uses stricter UpperSnake.
    ConstantName = MixedSnake
    # ROS2 uses stricter LowerSnake.
    FieldName = MixedSnake
    # ROS2 uses stricter UpperCamel.
    MessageName = MixedCamel
    PackageName = LowerSnake

    def specification(self) -> list[Message]:
        """Parse message specification."""
        if isinstance(
            res := self.all(
                0,
                (
                    lambda p: self.regex(p, r'\s*', re.MULTILINE),
                    self.message,
                    lambda p: self.many0(
                        p,
                        lambda p: self.all(
                            p,
                            (
                                lambda p: self.literal(p, self.Separator + '\n'),
                                self.message,
                            ),
                        ),
                    ),
                ),
            ),
            Failure,
        ):
            msg = f'Could not parse:\n{self.text!r}\n{res.pos}\n{res.context}\n{res.expected}'
            raise TypesysError(msg)

        if res.pos < len(self.text):
            msg = f'Could not parse:\n{self.text!r}\n{res.pos}\n{self.text[res.pos :]}'
            raise TypesysError(msg)

        _, head, tail = res.value
        msgs = [head, *(x[1] for x in tail)]

        return [
            Message(
                name := normalize_msgtype(x.name),
                x.constants,
                [y._replace(typ=normalize_type_ref(name, y.typ)) for y in x.fields],
            )
            for x in msgs
        ]

    def message(self, pos: int) -> Result[Message]:
        """Parse message."""
        if isinstance(
            res := self.all(
                pos,
                (
                    self.header,
                    lambda p: self.many0(p, self.line),
                ),
            ),
            Failure,
        ):
            return res
        hdr, lines = res.value
        constants = [x for x in lines if isinstance(x, Constant)]
        fields = [x for x in lines if isinstance(x, Field)]
        return Success(res.pos, Message(hdr, constants, fields))

    def header(self, pos: int) -> Result[str]:
        """Parse message header."""
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, 'MSG: '),
                    lambda p: self.regex(p, r'[A-Za-z0-9/_]+'),
                    lambda p: self.regex(p, r'[ \t]*\n'),
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, res.value[1])

    def line(self, pos: int) -> Result[Annotation | Constant | Field | None]:
        """Parse message line."""
        return self.any(
            pos,
            (
                self.annotation,
                self.constant,
                self.field,
                self.end,
            ),
        )

    def annotation(self, pos: int) -> Result[Annotation]:
        """Parse annotation."""
        if isinstance(
            res := self.all(
                self.optspace(pos).pos,
                (
                    lambda p: self.literal(p, '@'),
                    lambda p: self.regex(p, '[a-z]+'),
                    lambda p: self.any(p, (self.space, self.newline)),
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, Annotation(res.value[1]))

    def constant(self, pos: int) -> Result[Constant]:
        """Parse constant."""
        if isinstance(
            res := self.all(
                self.optspace(pos).pos,
                (
                    self.base_type,
                    self.space,
                    lambda p: self.regex(p, self.ConstantName),
                    lambda p: self.regex(p, r'[ \t]*=[ \t]*'),
                    self.value,
                    self.end,
                ),
            ),
            Failure,
        ):
            return res

        typ, _, name, _, value, _ = res.value
        assert isinstance(typ.ref, BaseType)
        assert not isinstance(value, list)
        return Success(res.pos, Constant(typ.ref.name, name, value))

    def field(self, pos: int) -> Result[Field]:
        """Parse field."""
        if isinstance(
            res := self.all(
                self.optspace(pos).pos,
                (
                    self.typ,
                    self.space,
                    lambda p: self.regex(p, self.FieldName),
                    lambda p: self.optional(
                        p,
                        lambda p: self.all(p, (self.space, self.value)),
                    ),
                    self.end,
                ),
            ),
            Failure,
        ):
            return res

        typ, _, name, optvalue, _ = res.value
        return Success(res.pos, Field(typ, name, optvalue[1] if optvalue is not None else None))

    def typ(self, pos: int) -> Result[Type]:
        """Parse type."""
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.any(
                        p,
                        (
                            self.bounded_string,
                            self.base_type,
                            self.message_type,
                        ),
                    ),
                    lambda p: self.optional(
                        p,
                        lambda p: self.all(
                            p,
                            (
                                lambda p: self.literal(p, '['),
                                lambda p: self.optional(
                                    p,
                                    lambda p: self.any(
                                        p,
                                        (
                                            self.decimal_literal,
                                            lambda p: self.all(
                                                p,
                                                (
                                                    lambda p: self.literal(p, '<='),
                                                    self.decimal_literal,
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                                lambda p: self.literal(p, ']'),
                            ),
                        ),
                    ),
                ),
            ),
            Failure,
        ):
            return res

        typ, default = res.value
        if default is not None:
            if default[1] is None:
                typ = typ._replace(cardinality=Cardinality.SEQUENCE)
            elif isinstance(default[1], tuple):
                typ = typ._replace(cardinality=Cardinality.SEQUENCE, size=default[1][1])
            else:
                typ = typ._replace(cardinality=Cardinality.ARRAY, size=default[1])

        return Success(res.pos, typ)

    def message_type(self, pos: int) -> Result[Type]:
        """Parse message type."""
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.many0(
                        p,
                        lambda p: self.all(
                            p,
                            (
                                lambda p: self.regex(p, self.PackageName),
                                lambda p: self.literal(p, '/'),
                            ),
                        ),
                    ),
                    lambda p: self.any(
                        p,
                        (
                            lambda p: self.regex(p, self.MessageName),
                            lambda p: self.literal(p, 'time'),
                            lambda p: self.literal(p, 'duration'),
                        ),
                    ),
                ),
            ),
            Failure,
        ):
            return res

        pkgs, name = res.value
        name = '/'.join((*(x[0] for x in pkgs), name))

        dct: dict[str, str] = {
            'time': 'builtin_interfaces/msg/Time',
            'duration': 'builtin_interfaces/msg/Duration',
            'Header': 'std_msgs/msg/Header',
        }
        name = dct.get(name, name)

        return Success(
            res.pos,
            Type(NamedType(name), Cardinality.SCALAR, 0, 0),
        )

    def bounded_string(self, pos: int) -> Result[Type]:
        """Parse bounded string."""
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, 'string<='),
                    self.decimal_literal,
                ),
            ),
            Failure,
        ):
            return res
        return Success(
            res.pos,
            Type(BaseType(BASE_NAMES['string']), Cardinality.SCALAR, res.value[1], 0),
        )

    def base_type(self, pos: int) -> Result[Type]:
        """Parse base type."""
        if isinstance(
            res := self.regex(pos, r'(bool|byte|char|float(32|64)|u?int(8|16|32|64)|string)\b'),
            Failure,
        ):
            return res
        return Success(res.pos, Type(BaseType(BASE_NAMES[res.value]), Cardinality.SCALAR, 0, 0))

    def value(self, pos: int) -> Result[Value]:
        """Parse value."""
        return self.any(
            pos,
            (self.array_value, self.scalar_value),
        )

    def array_value(self, pos: int) -> Result[list[ScalarValue]]:
        """Parse array value."""
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, '['),
                    self.optspace,
                    self.scalar_value,
                    lambda p: self.many0(
                        p,
                        (
                            lambda p: self.all(
                                p,
                                (
                                    self.optspace,
                                    lambda p: self.literal(p, ','),
                                    self.optspace,
                                    self.scalar_value,
                                ),
                            )
                        ),
                    ),
                    self.optspace,
                    lambda p: self.literal(p, ']'),
                ),
            ),
            Failure,
        ):
            return res

        _, _, head, sep_tail, _, _ = res.value
        items = [head, *(x[3] for x in sep_tail)]
        return Success(res.pos, items)

    def scalar_value(self, pos: int) -> Result[ScalarValue]:
        """Parse scalar value."""
        return self.any(
            pos,
            (
                self.float_literal,
                self.integer_literal,
                self.boolean_literal,
                self.string_literal,
            ),
        )

    def float_literal(self, pos: int) -> Result[float]:
        """Parse float value."""
        if isinstance(
            res := self.regex(
                pos,
                (
                    r'[+-]?([0-9]+\.[0-9]*([eE][+-]?[0-9]+)?)|'
                    r'(\.[0-9]+([eE][+-]?[0-9]+)?)|'
                    r'([+-]?[0-9]+[eE][+-]?[0-9]+)'
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, float(res.value))

    def integer_literal(self, pos: int) -> Result[int]:
        """Parse integer value."""
        return self.any(
            pos,
            (
                self.hexadecimal_literal,
                self.binary_literal,
                self.octal_literal,
                self.decimal_literal,
            ),
        )

    def hexadecimal_literal(self, pos: int) -> Result[int]:
        """Parse hexadecimal integer value."""
        if isinstance(res := self.regex(pos, r'[-+]?0[xX][a-fA-F0-9]+'), Failure):
            return res
        return Success(res.pos, int(res.value, 0))

    def decimal_literal(self, pos: int) -> Result[int]:
        """Parse decimal integer value."""
        if isinstance(res := self.regex(pos, r'[+-]?[0-9]+'), Failure):
            return res
        return Success(res.pos, int(res.value))

    def octal_literal(self, pos: int) -> Result[int]:
        """Parse octal integer value."""
        if isinstance(res := self.regex(pos, r'[-+]?0[0-7]+'), Failure):
            return res
        return Success(res.pos, int(res.value, 8))

    def binary_literal(self, pos: int) -> Result[int]:
        """Parse binary integer value."""
        if isinstance(res := self.regex(pos, r'[-+]?0b[01]+'), Failure):
            return res
        return Success(res.pos, int(res.value, 0))

    def boolean_literal(self, pos: int) -> Result[bool]:
        """Parse boolean value."""
        if isinstance(res := self.regex(pos, r'true|false', re.IGNORECASE), Failure):
            return res
        return Success(res.pos, res.value.lower() == 'true')

    def string_literal(self, pos: int) -> Result[str]:
        """Parse string value."""
        if isinstance(
            res := self.regex(pos, r"""('(?:\\.|[^'\\])*')|("(?:\\.|[^"\\])*")"""),
            Failure,
        ):
            if isinstance(res := self.regex(pos, r'[ \t]*[^# \t\n][^\n]*'), Failure):
                return res
            return Success(res.pos, res.value.strip())
        return Success(res.pos, res.value[1:-1])

    def end(self, pos: int) -> Result[None]:
        """Parse line end."""
        if isinstance(
            res := self.all(
                pos,
                (
                    self.optspace,
                    lambda p: self.optional(p, self.comment),
                    self.newline,
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, None)

    def comment(self, pos: int) -> Result[str]:
        """Parse comment."""
        return self.regex(pos, r'#[^\n]*')

    def newline(self, pos: int) -> Result[str]:
        """Parse newline."""
        return self.regex(pos, r'\n|$')

    def optspace(self, pos: int) -> Result[str]:
        """Parse optional whitespace."""
        return self.regex(pos, r'[ \t]*')

    def space(self, pos: int) -> Result[str]:
        """Parse whitespace."""
        return self.regex(pos, r'[ \t]+')


def get_types_from_msg(text: str, name: str) -> Typesdict:
    """Get type from msg message definition.

    Args:
        text: Message definition.
        name: Message typename.

    Returns:
        list with single message name and parsetree.

    """
    return make_typesdict(MSGParser(f'MSG: {name}\n{text}').specification())
