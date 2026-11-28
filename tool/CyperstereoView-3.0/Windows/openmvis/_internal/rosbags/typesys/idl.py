# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""IDL Parser.

Parsing grammar, parser and conversion functions for message definitions in
`IDL`_ format.

.. _IDL: https://www.omg.org/spec/IDL/About-IDL/

"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

from .base import (
    BASE_NAMES,
    BaseName,
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
    from collections.abc import Sequence
    from typing import TypeAlias

    from rosbags.interfaces.typing import ScalarValue, Typesdict

    from .base import TypeRef
    from .peg import Result, Rule

    Expression: TypeAlias = 'UnaryExpression | BinaryExpression | Reference | Value'


class Annotation(NamedTuple):
    """Annotation."""

    name: str
    arg: list[tuple[str, Expression]] | None


class ConstDecl(NamedTuple):
    """Constant."""

    name: str
    typ: ExpType
    value: Expression


class Enum(NamedTuple):
    """Enum."""

    name: str
    enumerators: list[tuple[str, Expression | None]]


class Member(NamedTuple):
    """Member."""

    name: str
    typ: ExpType


class Struct(NamedTuple):
    """Struct."""

    name: str
    fields: list[Member]


class UnaryExpression(NamedTuple):
    """Unary expression."""

    op: str
    expr: Expression


class BinaryExpression(NamedTuple):
    """Binary expression."""

    op: str
    left: Expression
    right: Expression


class Reference(NamedTuple):
    """Named reference."""

    name: ExpType


class Value(NamedTuple):
    """Literal value."""

    value: ScalarValue


class ExpType(NamedTuple):
    """Named type."""

    ref: TypeRef
    cardinality: Cardinality = Cardinality.SCALAR
    str_size: Expression = Value(0)
    size: Expression = Value(0)


def transform_declarator(
    typ: ExpType,
    decls: list[str | tuple[str, list[Expression]]],
) -> list[Member]:
    """Transform declarator to list of fields."""
    items = []
    for decl in decls:
        if isinstance(decl, str):
            items.append(Member(decl, typ))
        else:
            name, dims = decl
            if len(dims) != 1:
                msg = 'Multidimensional arrays are not supported'
                raise TypesysError(msg)
            items.append(
                Member(
                    name,
                    ExpType(
                        typ.ref,
                        str_size=typ.str_size,
                        cardinality=Cardinality.ARRAY,
                        size=dims[0],
                    ),
                )
            )
    return items


def resolve_typedef(typ: ExpType, scopename: str, typedefs: dict[str, ExpType]) -> ExpType:
    """Resolve typedef."""
    if isinstance(typ.ref, BaseType):
        return typ
    name = typ.ref.name

    def merge_type(src: ExpType, dst: ExpType) -> ExpType:
        if Cardinality.SCALAR not in (src.cardinality, dst.cardinality):
            msg = 'Multidimensional arrays are not supported'
            raise TypesysError(msg)
        return (
            dst
            if src.cardinality == Cardinality.SCALAR
            else ExpType(dst.ref, str_size=dst.str_size, cardinality=src.cardinality, size=src.size)
        )

    if name.startswith('/'):
        name = name[1:]
        if name in typedefs:
            typ = merge_type(typ, typedefs[name])
            return resolve_typedef(typ, name, typedefs)
        return typ._replace(ref=NamedType(name))

    for parent in PurePosixPath(scopename).parents:
        candidate = str(parent / name)
        if candidate in typedefs:
            typ = merge_type(typ, typedefs[candidate])
            return resolve_typedef(typ, candidate, typedefs)
    return typ


def resolve_type(
    typ: ExpType,
    scopename: str,
    types: Sequence[str],
    consts: Sequence[ConstDecl],
) -> Type:
    """Resolve type reference."""
    str_size = eval_expression(
        typ.str_size,
        ExpType(BaseType(BaseName.UINT32)),
        scopename,
        consts,
    ).value
    if not isinstance(str_size, int):
        msg = f'String bound must be an integer, got {str_size!r}'
        raise TypesysError(msg)

    size = eval_expression(
        typ.size,
        ExpType(BaseType(BaseName.UINT32)),
        scopename,
        consts,
    ).value
    if not isinstance(size, int):
        msg = f'Array or sequence bound must be an integer, got {size!r}'
        raise TypesysError(msg)

    if isinstance(typ.ref, BaseType):
        return Type(typ.ref, typ.cardinality, str_size, size)

    name = typ.ref.name
    for parent in PurePosixPath(scopename).parents:
        candidate = str(parent / name)
        if candidate in types:
            name = candidate
            break
    else:
        if '/' not in name:
            name = str(PurePosixPath(scopename).parent / name)

    return Type(NamedType(name), typ.cardinality, str_size, size)


def eval_expression(
    expr: Expression,
    typ: ExpType,
    name: str,
    consts: Sequence[ConstDecl],
) -> Value:
    """Evaluate const expression."""
    if not isinstance(typ.ref, BaseType):
        msg = f'Cannot evaluate constant of non-base type {typ.ref.name!r}'
        raise TypesysError(msg)

    if isinstance(expr, UnaryExpression):
        op = expr.op
        value = eval_expression(expr.expr, typ, name, consts)
        if op == '-':
            if not isinstance(value.value, float | int):
                msg = f'Cannot apply operator {op!r} to {value.value!r}'
                raise TypesysError(msg)
            return Value(-value.value)
        if op == '~':
            if not isinstance(value.value, int):
                msg = f'Cannot apply operator {op!r} to {value.value!r}'
                raise TypesysError(msg)
            return Value(~value.value)
        return value

    if isinstance(expr, BinaryExpression):
        op = expr.op
        lval = eval_expression(expr.left, typ, name, consts).value
        rval = eval_expression(expr.right, typ, name, consts).value
        if type(lval) is not type(rval):
            ltype = type(lval).__name__
            rtype = type(rval).__name__
            msg = f'Cannot mix types {ltype!r} and {rtype!r} in expression'
            raise TypesysError(msg)
        if op in {'*', '/', '+', '-'}:
            if not isinstance(lval, float | int):
                msg = f'Cannot apply operator {op!r} to {lval!r}'
                raise TypesysError(msg)
            rval = cast('float | int', rval)
            if op == '-':
                return Value(lval - rval)
            if op == '+':
                return Value(lval + rval)
            if op == '*':
                return Value(lval * rval)
            assert op == '/'
            if rval == 0:
                msg = 'Division by zero in expression'
                raise TypesysError(msg)
            return Value(lval / rval if type(rval) is float else lval // rval)
        if not isinstance(lval, int):
            msg = f'Cannot apply operator {op!r} to {lval!r}'
            raise TypesysError(msg)
        rval = cast('int', rval)
        if op in {'<<', '>>'} and rval < 0:
            msg = 'Cannot shift by negative count in expression'
            raise TypesysError(msg)
        if op == '%':
            if rval == 0:
                msg = 'Modulo by zero in expression'
                raise TypesysError(msg)
            return Value(lval % rval)
        if op == '<<':
            return Value(lval << rval)
        if op == '>>':
            return Value(lval >> rval)
        if op == '&':
            return Value(lval & rval)
        if op == '|':
            return Value(lval | rval)
        assert op == '^'
        return Value(lval ^ rval)

    if isinstance(expr, Reference):
        refname = cast('NamedType', expr.name.ref).name
        for parent in PurePosixPath(name).parents:
            candidate = str(parent / refname)
            item = next((x for x in consts if x.name == candidate), None)
            if item:
                assert isinstance(item.value, Value)
                return item.value
        msg = f'Cannot resolve constant {refname!r}'
        raise TypesysError(msg)

    return expr


class IDLParser(Parser):
    """IDL Parser."""

    def specification(self) -> list[Message]:
        """Parse specification (1).

        Grammar:
            definition+

        """
        spc = self.optspace(0)
        if isinstance(
            res := self.many1(spc.pos, lambda p: self.all(p, (self.definition, self.optspace))),
            Failure,
        ):
            msg = f'Could not parse:\n{self.text!r}\n{res.pos}\n{res.context}\n{res.expected}'
            raise TypesysError(msg)

        if res.pos != len(self.text):
            context = self.text[res.pos :]
            msg = f'Could not parse, left over input:\n{self.text!r}\n{res.pos}\n{context}'
            raise TypesysError(msg)

        defs = [y for x in res.value for y in x[0]]

        typedefs = {}
        typerefs = []

        consts: list[ConstDecl] = []
        msgs: list[Message] = []
        for item in defs:
            if isinstance(item, Member):
                typedefs[item.name] = item.typ
            elif isinstance(item, ConstDecl):
                consts.append(
                    item._replace(
                        typ=(typ := resolve_typedef(item.typ, item.name, typedefs)),
                        value=eval_expression(item.value, typ, item.name, consts),
                    )
                )
            elif isinstance(item, Struct):
                typerefs.append(item.name)
                msgs.append(
                    Message(
                        item.name,
                        [
                            Constant(
                                cast('BaseType', x.typ.ref).name,
                                x.name.split('/')[-1],
                                cast('Value', x.value).value,
                            )
                            for x in consts
                            if x.name.startswith(f'{item.name}_Constants/')
                        ],
                        [
                            Field(
                                resolve_type(
                                    resolve_typedef(x.typ, item.name, typedefs),
                                    item.name,
                                    typerefs,
                                    consts,
                                ),
                                x.name,
                                None,
                            )
                            for x in item.fields
                        ],
                    )
                )
        return msgs

    def definition(self, pos: int) -> Result[list[Enum | Struct | ConstDecl | Member]]:
        """Parse definition (2).

        Grammar:
            module_dcl ";"
            / const_dcl ";"
            / type_dcl ";"

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.any(
                        p,
                        (
                            self.module_dcl,
                            self.const_dcl,
                            self.type_dcl,
                        ),
                    ),
                    self.optspace,
                    lambda p: self.literal(p, ';'),
                ),
            ),
            Failure,
        ):
            return res
        item = res.value[0]
        xitem: list[Enum | Struct | ConstDecl | Member] = []
        if isinstance(item, list):
            xitem.extend(item)
        else:
            xitem.append(item)
        return Success(res.pos, xitem)

    def module_dcl(self, pos: int) -> Result[list[Enum | Struct | ConstDecl | Member]]:
        """Parse module_dcl (3).

        Grammar:
            annotation_appl* "module" IDENTIFIER "{" definition+ "}"

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.many0(
                        p,
                        lambda p: self.all(p, (self.annotation_appl, self.space)),
                    ),
                    lambda p: self.all(
                        p,
                        (
                            lambda p: self.literal(p, 'module'),
                            self.space,
                            self.identifier,
                            self.optspace,
                            lambda p: self.literal(p, '{'),
                            self.optspace,
                        ),
                    ),
                    lambda p: self.many1(
                        p,
                        lambda p: self.all(
                            p,
                            (
                                self.definition,
                                self.optspace,
                            ),
                        ),
                    ),
                    lambda p: self.literal(p, '}'),
                ),
            ),
            Failure,
        ):
            return res
        _annoatations, (_, _, name, _, _, _), spaced_defs, _ = res.value
        defs = [y for x in spaced_defs for y in x[0]]

        items = [x._replace(name=f'{name}/{x.name}') for x in defs]
        return Success(res.pos, items)

    def scoped_name(self, pos: int) -> Result[ExpType]:
        """Parse scoped_name (4).

        Grammar:
            IDENTIFIER
            / "::" IDENTIFIER
            / scoped_name "::" IDENTIFIER

        """
        if isinstance(
            res := self.any(
                pos,
                (
                    lambda p: self.all(
                        p,
                        (self.identifier, lambda p: self.literal(p, '::'), self.scoped_name),
                    ),
                    lambda p: self.all(
                        p,
                        (lambda p: self.literal(p, '::'), self.scoped_name),
                    ),
                    self.identifier,
                ),
            ),
            Failure,
        ):
            return res
        if isinstance(res.value, str):
            name = res.value
        elif len(res.value) == 2:
            name = f'/{res.value[1].ref.name}'
        else:
            typ = cast('NamedType', res.value[2].ref)
            name = f'{res.value[0]}/{typ.name}'

        if name in {
            'FALSE',
            'TRUE',
            'boolean',
            'char',
            'double',
            'float',
            'int',
            'int16',
            'int32',
            'int64',
            'int8',
            'long',
            'octet',
            'sequence',
            'short',
            'string',
            'uint16',
            'uint32',
            'uint64',
            'uint8',
            'unsigned',
            'wchar',
            'wstring',
        }:
            return Failure(
                res.pos,
                self.text[res.pos : min(len(self.text), res.pos + 32)],
                {'Base type found'},
            )
        return Success(res.pos, ExpType(NamedType(name)))

    def const_dcl(self, pos: int) -> Result[ConstDecl]:
        """Parse const_dcl (5).

        Grammar:
            annotation_appl* "const" const_type IDENTIFIER "=" const_expr

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.many0(
                        p,
                        lambda p: self.all(
                            p,
                            (self.annotation_appl, self.space),
                        ),
                    ),
                    lambda p: self.all(
                        p,
                        (
                            lambda p: self.literal(p, 'const'),
                            self.space,
                            self.const_type,
                            self.space,
                            self.identifier,
                            self.optspace,
                        ),
                    ),
                    lambda p: self.all(
                        p,
                        (
                            lambda p: self.literal(p, '='),
                            self.optspace,
                            self.const_expr,
                        ),
                    ),
                ),
            ),
            Failure,
        ):
            return res
        _annotations, (_, _, typ, _, name, _), (_, _, value) = res.value
        return Success(res.pos, ConstDecl(name, typ, value))

    def const_type(self, pos: int) -> Result[ExpType]:
        """Parse const_type (6).

        Grammar:
            integer_type
            / floating_pt_type
            / "fixed"           (43)
            / "char"            (34)
            / "wchar"           (35)
            / "boolean"         (36)
            / "octet"           (37)
            / string_type
            / wide_string_type
            / scoped_name

        """
        if isinstance(
            res := self.any(
                pos,
                (
                    self.floating_pt_type,
                    self.integer_type,
                    lambda p: self.regex(p, r'(:?fixed|char|wchar|boolean|octet)\b'),
                    self.string_type,
                    self.wide_string_type,
                    self.scoped_name,
                ),
            ),
            Failure,
        ):
            return res
        value = res.value
        if isinstance(value, str):
            name = {
                'boolean': 'bool',
                'octet': 'byte',
            }.get(value, value)
            return Success(res.pos, ExpType(BaseType(BASE_NAMES[name])))
        return Success(res.pos, value)

    def _binop(
        self, pos: int, left: Rule[Expression], op: str, right: Rule[Expression]
    ) -> Result[Expression]:
        """Parse binary expression."""
        if isinstance(
            res := self.all(
                pos,
                (left, self.optspace, lambda p: self.regex(p, op), self.optspace, right),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, BinaryExpression(res.value[2], res.value[0], res.value[4]))

    def _unop(self, pos: int, op: str, rule: Rule[Expression]) -> Result[Expression]:
        """Parse unary expression."""
        if isinstance(
            res := self.all(
                pos,
                (lambda p: self.regex(p, op), self.optspace, rule),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, UnaryExpression(res.value[0], res.value[2]))

    def const_expr(self, pos: int) -> Result[Expression]:
        """Parse const_expr (7).

        Grammar:
            or_expr

        """
        return self.or_expr(pos)

    def or_expr(self, pos: int) -> Result[Expression]:
        """Parse or_expr (8).

        Grammar:
            xor_expr
            / or_expr "|" xor_expr

        """
        return self.any(
            pos,
            (lambda p: self._binop(p, self.xor_expr, r'\|', self.or_expr), self.xor_expr),
        )

    def xor_expr(self, pos: int) -> Result[Expression]:
        """Parse xor_expr (9).

        Grammar:
            and_expr
            / xor_expr "^" and_expr

        """
        return self.any(
            pos,
            (lambda p: self._binop(p, self.and_expr, r'\^', self.xor_expr), self.and_expr),
        )

    def and_expr(self, pos: int) -> Result[Expression]:
        """Parse and_expr (10).

        Grammar:
            shift_expr
            / and_expr "&" shift_expr

        """
        return self.any(
            pos,
            (lambda p: self._binop(p, self.shift_expr, r'&', self.and_expr), self.shift_expr),
        )

    def shift_expr(self, pos: int) -> Result[Expression]:
        """Parse shift_expr (11).

        Grammar:
            add_expr
            / shift_expr ">>" add_expr
            / shift_expr "<<" add_expr

        """
        return self.any(
            pos,
            (lambda p: self._binop(p, self.add_expr, r'>>|<<', self.shift_expr), self.add_expr),
        )

    def add_expr(self, pos: int) -> Result[Expression]:
        """Parse add_expr (12).

        Grammar:
            mult_expr
            / add_expr "&" mult_expr

        """
        return self.any(
            pos,
            (lambda p: self._binop(p, self.mult_expr, r'\+|-', self.add_expr), self.mult_expr),
        )

    def mult_expr(self, pos: int) -> Result[Expression]:
        """Parse mult_expr (13).

        Grammar:
            unary_expr
            / mult_expr "&" unary_expr

        """
        return self.any(
            pos,
            (lambda p: self._binop(p, self.unary_expr, r'\*|/|%', self.mult_expr), self.unary_expr),
        )

    def unary_expr(self, pos: int) -> Result[Expression]:
        """Parse unary_expr (14).

        Grammar:
            unary_op primary_expr   (15)
            / primary_expr

        """
        return self.any(
            pos,
            (lambda p: self._unop(p, r'-|\+|~', self.primary_expr), self.primary_expr),
        )

    def primary_expr(self, pos: int) -> Result[Expression]:
        """Parse primary_expr (16).

        Grammar:
            scoped_name
            / literal
            / "(" const_expr ")"

        """
        if isinstance(nres := self.scoped_name(pos), Success):
            return Success(nres.pos, Reference(nres.value))

        if isinstance(vres := self.literal_(pos), Success):
            return Success(vres.pos, Value(vres.value))

        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, '('),
                    self.optspace,
                    self.const_expr,
                    self.optspace,
                    lambda p: self.literal(p, ')'),
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, res.value[2])

    def literal_(self, pos: int) -> Result[ScalarValue]:
        """Parse literal (17).

        Grammar:
            integer_literal
            / floating_pt_literal
            / fixed_pt_literal
            / character_literal
            / wide_character_literal
            / boolean_literal
            / string_literals
            / wide_string_literals

        """
        return self.any(
            pos,
            (
                self.float_literal,
                self.integer_literal,
                # self.fixed_pt_literal,
                self.character_literal,
                # self.wide_character_literal,
                self.boolean_literal,
                self.strings_literal,
                # self.wide_string_literals,
            ),
        )

    def boolean_literal(self, pos: int) -> Result[bool]:
        """Parse boolean_literal (18).

        Grammar:
            "TRUE"
            / "FALSE"

        """
        if isinstance(
            res := self.any(
                pos,
                (
                    lambda p: self.literal(p, 'TRUE'),
                    lambda p: self.literal(p, 'FALSE'),
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, res.value == 'TRUE')

    def positive_int_const(self, pos: int) -> Result[Expression]:
        """Parse positive_int_const (19).

        Grammar:
            const_expr

        """
        if isinstance(res := self.const_expr(pos), Failure):
            return res
        expr = res.value
        return Success(res.pos, expr)

    def type_dcl(self, pos: int) -> Result[Enum | Struct | list[Member]]:
        """Parse type_decl (20).

        Grammar:
            constr_type_dcl
            / typedef_dcl

        """
        return self.any(
            pos,
            (
                self.constr_type_dcl,
                self.typedef_dcl,
            ),
        )

    def type_spec(self, pos: int) -> Result[ExpType]:
        """Parse type_spec (21, 216).

        Grammar:
            simple_type_spec
            / template_type_spec

        """
        return self.any(
            pos,
            (
                self.simple_type_spec,
                self.template_type_spec,
            ),
        )

    def simple_type_spec(self, pos: int) -> Result[ExpType]:
        """Parse simple_type_spec (22).

        Grammar:
            base_type_spec
            / scoped_name

        """
        return self.any(
            pos,
            (
                self.base_type_spec,
                self.scoped_name,
            ),
        )

    def base_type_spec(self, pos: int) -> Result[ExpType]:
        """Parse base_type_spec (23).

        Grammar:
            integer_type
            / floating_pt_type
            / "char"            (34)
            / "wchar"           (35)
            / "boolean"         (36)
            / "octet"           (37)

        """
        if isinstance(
            res := self.any(
                pos,
                (
                    self.floating_pt_type,
                    self.integer_type,
                    lambda p: self.regex(p, r'(:?char|wchar|boolean|octet)\b'),
                ),
            ),
            Failure,
        ):
            return res
        value = res.value
        if isinstance(value, str):
            name = {
                'boolean': 'bool',
                'octet': 'byte',
            }.get(value, value)
            return Success(res.pos, ExpType(BaseType(BASE_NAMES[name])))
        return Success(res.pos, value)

    def floating_pt_type(self, pos: int) -> Result[ExpType]:
        """Parse floating_pt_type (24).

        Grammar:
            "float"
            / "double"
            / "long" "double"

        """
        if isinstance(res := self.regex(pos, r'(:?long\s+double|double|float)\b'), Failure):
            return res

        name = {
            'float': 'float32',
            'double': 'float64',
            'long double': 'float128',
        }[' '.join(res.value.split())]

        return Success(res.pos, ExpType(BaseType(BASE_NAMES[name])))

    def integer_type(self, pos: int) -> Result[ExpType]:
        """Parse integer_type (25).

        Grammar:
            signed_int
            / unsigned_int

            signed_int                   (26)
               = "short"                 (27)
               / "long"                  (28)
               / "long" "long"           (29)
               / "int8"                  (206, 208)
               / "int16"                 (210)
               / "int32"                 (211)
               / "int64"                 (212)

             unsigned_int                 (30)
               = "unsigned" "short"       (31)
               / "unsigned" "long"        (32)
               / "unsigned" "long" "long" (33)
               / "uint8"                  (207, 209)
               / "uint16"                 (213)
               / "uint32"                 (214)
               / "uint64"                 (215)

        """
        if isinstance(
            res := self.regex(
                pos,
                r'(:?(u?int(64|32|16|8))|((unsigned\s+)?((long\s+)?long|short)))\b',
            ),
            Failure,
        ):
            return res

        oname = ' '.join(res.value.split())
        name = {
            'short': 'int16',
            'long': 'int32',
            'long long': 'int64',
            'unsigned short': 'uint16',
            'unsigned long': 'uint32',
            'unsigned long long': 'uint64',
        }.get(oname, oname)

        return Success(res.pos, ExpType(BaseType(BASE_NAMES[name])))

    def template_type_spec(self, pos: int) -> Result[ExpType]:
        """Parse template_type_spec (38).

        Grammar:
            sequence_type
            / string_type
            / wide_string_type
            / fixed_pt_type

        """
        return self.any(
            pos,
            (
                self.sequence_type,
                self.string_type,
                self.wide_string_type,
                # self.fixed_pt_type,
            ),
        )

    def sequence_type(self, pos: int) -> Result[ExpType]:
        """Parse sequence_type (39).

        Grammar:
            "sequence" "<" type_spec "," positive_int_const ">"
            / "sequence" "<" type_spec ">"

        """
        if isinstance(
            xres := self.all(
                pos,
                (
                    lambda p: self.literal(p, 'sequence'),
                    self.optspace,
                    lambda p: self.literal(p, '<'),
                    lambda p: self.all(
                        p,
                        (
                            self.optspace,
                            self.type_spec,
                            self.optspace,
                            lambda p: self.literal(p, ','),
                            self.optspace,
                            self.positive_int_const,
                        ),
                    ),
                    self.optspace,
                    lambda p: self.literal(p, '>'),
                ),
            ),
            Success,
        ):
            typ = xres.value[3][1]
            bound = xres.value[3][5]
            return Success(
                xres.pos,
                ExpType(
                    typ.ref, str_size=typ.str_size, cardinality=Cardinality.SEQUENCE, size=bound
                ),
            )

        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, 'sequence'),
                    self.optspace,
                    lambda p: self.literal(p, '<'),
                    lambda p: self.all(
                        p,
                        (
                            self.optspace,
                            self.type_spec,
                        ),
                    ),
                    self.optspace,
                    lambda p: self.literal(p, '>'),
                ),
            ),
            Failure,
        ):
            return res
        typ = res.value[3][1]
        return Success(
            res.pos,
            ExpType(typ.ref, str_size=typ.str_size, cardinality=Cardinality.SEQUENCE),
        )

    def _bound_unbound(self, pos: int, typ: Literal['string', 'wstring']) -> Result[ExpType]:
        """Parse bound / unbound items.

        Grammar:
            typ "<" positive_int_const ">"
            / typ

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, typ),
                    self.optspace,
                    lambda p: self.literal(p, '<'),
                    lambda p: self.all(
                        p,
                        (
                            self.optspace,
                            self.positive_int_const,
                            self.optspace,
                        ),
                    ),
                    lambda p: self.literal(p, '>'),
                ),
            ),
            Success,
        ):
            return Success(res.pos, ExpType(BaseType(BASE_NAMES[typ]), str_size=res.value[3][1]))
        if isinstance(xres := self.regex(pos, typ), Failure):
            return res
        return Success(xres.pos, ExpType(BaseType(BASE_NAMES[typ])))

    def string_type(self, pos: int) -> Result[ExpType]:
        """Parse string_type (40).

        Grammar:
            "string" "<" positive_int_const ">"
            / "string"

        """
        return self._bound_unbound(pos, 'string')

    def wide_string_type(self, pos: int) -> Result[ExpType]:
        """Parse wide_string_type (41).

        Grammar:
            "wstring" "<" positive_int_const ">"
            / "wstring"

        """
        return self._bound_unbound(pos, 'wstring')

    def fixed_pt_type(self, pos: int) -> Result[ExpType]:  # pragma: no cover
        """Parse fixed_pt_type (42).

        Grammar:
            "fixed" "<" positive_int_const "," positive_int_const ">"

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, 'fixed'),
                    self.optspace,
                    lambda p: self.literal(p, '<'),
                    lambda p: self.all(
                        p,
                        (
                            self.optspace,
                            self.positive_int_const,
                            self.optspace,
                            lambda p: self.literal(p, ','),
                            self.optspace,
                            self.positive_int_const,
                        ),
                    ),
                    self.optspace,
                    lambda p: self.literal(p, '>'),
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, ExpType(BaseType(BASE_NAMES['fixed'])))

    def constr_type_dcl(self, pos: int) -> Result[Struct | Enum]:
        """Parse constr_type_dcl (44).

        Grammar:
            struct_dcl
            / enum_dcl

        """
        return self.any(pos, (self.struct_dcl, self.enum_dcl))

    def struct_dcl(self, pos: int) -> Result[Struct]:
        """Parse struct_dcl (45).

        Grammar:
            struct_def
            / struct_forward_dcl

        """
        return self.any(pos, (self.struct_def, self.struct_forward_dcl))

    def struct_def(self, pos: int) -> Result[Struct]:
        """Parse struct_def (46).

        Grammar:
            annotation_appl* "struct" IDENTIFIER "{" member+ "}"

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.many0(
                        p,
                        lambda p: self.all(
                            p,
                            (self.annotation_appl, self.space),
                        ),
                    ),
                    lambda p: self.all(
                        p,
                        (
                            lambda p: self.literal(p, 'struct'),
                            self.space,
                            self.identifier,
                            self.optspace,
                        ),
                    ),
                    lambda p: self.all(
                        p,
                        (
                            lambda p: self.literal(p, '{'),
                            self.optspace,
                            lambda p: self.many1(
                                p,
                                lambda p: self.all(
                                    p,
                                    (
                                        self.member,
                                        self.optspace,
                                    ),
                                ),
                            ),
                            self.optspace,
                            lambda p: self.literal(p, '}'),
                        ),
                    ),
                ),
            ),
            Failure,
        ):
            return res
        _annotation, (_, _, name, _), (_, _, spaced_members, _, _) = res.value
        members = [y for x in spaced_members for y in x[0]]
        return Success(res.pos, Struct(name, members))

    def member(self, pos: int) -> Result[list[Member]]:
        """Parse member (47).

        Grammar:
            annotation_appl* type_spec declarators ";"

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.many0(
                        p,
                        lambda p: self.all(
                            p,
                            (self.annotation_appl, self.space),
                        ),
                    ),
                    self.type_spec,
                    self.space,
                    self.declarators,
                    self.optspace,
                    lambda p: self.literal(p, ';'),
                ),
            ),
            Failure,
        ):
            return res

        _annotations, typ, _, decls, _, _ = res.value
        return Success(res.pos, transform_declarator(typ, decls))

    def struct_forward_dcl(self, pos: int) -> Result[Struct]:
        """Parse struct_forward_dcl (48).

        Grammar:
            "struct" IDENTIFIER

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, 'struct'),
                    self.space,
                    self.identifier,
                ),
            ),
            Failure,
        ):
            return res

        return Success(res.pos, Struct(res.value[2], []))

    def enum_dcl(self, pos: int) -> Result[Enum]:
        """Parse enum_dcl (57).

        Grammar:
            annotation_appl* "enum" IDENTIFIER "{" enumerator ("," enumerator)* "}"

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.many0(
                        p,
                        lambda p: self.all(
                            p,
                            (self.annotation_appl, self.space),
                        ),
                    ),
                    lambda p: self.all(
                        p,
                        (
                            lambda p: self.literal(p, 'enum'),
                            self.space,
                            self.identifier,
                            self.optspace,
                        ),
                    ),
                    lambda p: self.all(
                        p,
                        (
                            lambda p: self.literal(p, '{'),
                            self.optspace,
                            self.enumerator,
                            lambda p: self.many0(
                                p,
                                lambda p: self.all(
                                    p,
                                    (
                                        self.optspace,
                                        lambda p: self.literal(p, ','),
                                        self.optspace,
                                        self.enumerator,
                                    ),
                                ),
                            ),
                            self.optspace,
                            lambda p: self.literal(p, '}'),
                        ),
                    ),
                ),
            ),
            Failure,
        ):
            return res
        _annotations, (_, _, name, _), (_, _, head, tail, _, _) = res.value
        return Success(res.pos, Enum(name, [head, *(x[3] for x in tail)]))

    def enumerator(self, pos: int) -> Result[tuple[str, Expression | None]]:
        """Parse enumerator (58).

        Grammar:
            annotation_appl* IDENTIFIER

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.many0(
                        p,
                        lambda p: self.all(
                            p,
                            (self.annotation_appl, self.space),
                        ),
                    ),
                    self.identifier,
                ),
            ),
            Failure,
        ):
            return res
        annotations, name = res.value
        default = next((x.arg[0][1] for x, _ in annotations if x.name == 'value' and x.arg), None)
        return Success(res.pos, (name, default))

    def array_declarator(self, pos: int) -> Result[tuple[str, list[Expression]]]:
        """Parse array_declarator (59).

        Grammar:
            IDENTIFIER fixed_array_size+

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    self.identifier,
                    lambda p: self.many1(
                        p,
                        lambda p: self.all(
                            p,
                            (
                                self.optspace,
                                self.fixed_array_size,
                                self.optspace,
                            ),
                        ),
                    ),
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, (res.value[0], [x[1] for x in res.value[1]]))

    def fixed_array_size(self, pos: int) -> Result[Expression]:
        """Parse fixed_array_size (60).

        Grammar:
            "[" positive_int_const "]"

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, '['),
                    self.optspace,
                    self.positive_int_const,
                    self.optspace,
                    lambda p: self.literal(p, ']'),
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, res.value[2])

    def simple_declarator(self, pos: int) -> Result[str]:
        """Parse simple_declarator (62).

        Grammar:
            IDENTIFIER

        """
        return self.identifier(pos)

    def typedef_dcl(self, pos: int) -> Result[list[Member]]:
        """Parse typedef_dcl (63).

        Grammar:
            "typedef" type_declarator

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, 'typedef'),
                    self.space,
                    self.type_declarator,
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, transform_declarator(*res.value[2]))

    def type_declarator(
        self,
        pos: int,
    ) -> Result[tuple[ExpType, list[str | tuple[str, list[Expression]]]]]:
        """Parse type_declarator (64).

        Grammar:
            (simple_type_spec | template_type_spec | constr_type_dcl) any_declarators

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.any(
                        p,
                        (
                            self.simple_type_spec,
                            self.template_type_spec,
                            self.constr_type_dcl,
                        ),
                    ),
                    self.space,
                    self.any_declarators,
                ),
            ),
            Failure,
        ):
            return res
        typ = res.value[0]
        otyp: ExpType = ExpType(NamedType(typ.name)) if isinstance(typ, Struct | Enum) else typ
        return Success(res.pos, (otyp, res.value[2]))

    def any_declarators(self, pos: int) -> Result[list[str | tuple[str, list[Expression]]]]:
        """Parse any_declarators (65).

        Grammar:
            any_declarator ("," any_declarator)*

        """
        return self.declarators(pos)

    def any_declarator(
        self,
        pos: int,
    ) -> Result[str | tuple[str, list[Expression]]]:  # pragma: no cover
        """Parse any_declarator (66).

        Grammar:
            simple_declarator
            / array_declarator

        """
        return self.declarator(pos)

    def declarators(self, pos: int) -> Result[list[str | tuple[str, list[Expression]]]]:
        """Parse declarators (67).

        Grammar:
            declarator ("," declarator)*

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    self.declarator,
                    lambda p: self.many0(
                        p,
                        lambda p: self.all(
                            p,
                            (
                                self.optspace,
                                lambda p: self.literal(p, ','),
                                self.optspace,
                                self.declarator,
                            ),
                        ),
                    ),
                ),
            ),
            Failure,
        ):
            return res
        head, tail = res.value
        return Success(res.pos, [head, *(x[3] for x in tail)])

    def declarator(self, pos: int) -> Result[str | tuple[str, list[Expression]]]:
        """Parse declarator (68, 217).

        Grammar:
            simple_declarator
            / array_declarator

        """
        return self.any(pos, (self.array_declarator, self.simple_declarator))

    def annotation_appl(self, pos: int) -> Result[Annotation]:
        """Annotation appl (225).

        Grammar:
            "@" scoped_name [ "(" annotation_appl_params ")" ]

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    lambda p: self.literal(p, '@'),
                    self.optspace,
                    self.scoped_name,
                ),
            ),
            Failure,
        ):
            return res
        _, _, name = res.value
        assert isinstance(name.ref, NamedType)

        if isinstance(
            bres := self.all(
                res.pos,
                (
                    self.optspace,
                    lambda p: self.literal(p, '('),
                    self.optspace,
                ),
            ),
            Failure,
        ):
            return Success(res.pos, Annotation(name.ref.name, None))

        if isinstance(params := self.annotation_appl_params(bres.pos), Failure):
            return params

        if isinstance(
            cres := self.all(
                params.pos,
                (
                    self.optspace,
                    lambda p: self.literal(p, ')'),
                ),
            ),
            Failure,
        ):
            return cres
        return Success(cres.pos, Annotation(name.ref.name, params.value))

    def annotation_appl_params(self, pos: int) -> Result[list[tuple[str, Expression]]]:
        """Annotation appl params (226).

        Grammar:
            const_expr
            / annotation_appl_param ("," annotation_appl_param)*

        """
        if isinstance(param := self.annotation_appl_param(pos), Success):
            pos = param.pos
            items: list[tuple[str, Expression]] = [param.value]

            while True:
                if isinstance(
                    xparam := self.all(
                        pos,
                        (
                            self.optspace,
                            lambda p: self.literal(p, ','),
                            self.optspace,
                            self.annotation_appl_param,
                        ),
                    ),
                    Failure,
                ):
                    break
                pos = xparam.pos
                items.append(xparam.value[3])
            return Success(pos, items)

        if isinstance(res := self.const_expr(pos), Failure):
            return res
        return Success(res.pos, [('', res.value)])

    def annotation_appl_param(self, pos: int) -> Result[tuple[str, Expression]]:
        """Annotation appl param (227).

        Grammar:
            IDENTIFIER "=" const_expr

        """
        if isinstance(
            res := self.all(
                pos,
                (
                    self.identifier,
                    self.optspace,
                    lambda p: self.literal(p, '='),
                    self.optspace,
                    self.const_expr,
                ),
            ),
            Failure,
        ):
            return res
        return Success(res.pos, (res.value[0], res.value[4]))

    def float_literal(self, pos: int) -> Result[float]:
        """Float literal."""
        if isinstance(
            res := self.regex(
                pos, r'[-+]?(([0-9]+\.[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)'
            ),
            Failure,
        ):
            return res
        return Success(res.pos, float(res.value))

    def integer_literal(self, pos: int) -> Result[int]:
        """Integer literal."""
        return self.any(pos, (self.hexadecimal_literal, self.octal_literal, self.decimal_literal))

    def hexadecimal_literal(self, pos: int) -> Result[int]:
        """Parse hexadecimal literal."""
        if isinstance(res := self.regex(pos, r'[-+]?0[xX][a-fA-F0-9]+'), Failure):
            return res
        return Success(res.pos, int(res.value, 16))

    def decimal_literal(self, pos: int) -> Result[int]:
        """Parse decimal literal."""
        if isinstance(res := self.regex(pos, r'[+-]?[0-9]+'), Failure):
            return res
        return Success(res.pos, int(res.value))

    def octal_literal(self, pos: int) -> Result[int]:
        """Parse octal literal."""
        if isinstance(res := self.regex(pos, r'[-+]?0[0-7]+'), Failure):
            return res
        return Success(res.pos, int(res.value, 8))

    def character_literal(self, pos: int) -> Result[str]:
        """Parse character literal."""
        if isinstance(res := self.regex(pos, r'\'[a-zA-Z0-9_]\''), Failure):
            return res
        return Success(res.pos, res.value[1])

    def strings_literal(self, pos: int) -> Result[str]:
        """Parse multi string literal."""
        if isinstance(
            res := self.all(
                pos,
                (
                    self.string_literal,
                    lambda p: self.many0(
                        p,
                        lambda p: self.all(
                            p,
                            (
                                self.space,
                                self.string_literal,
                            ),
                        ),
                    ),
                ),
            ),
            Failure,
        ):
            return res

        head, tail = res.value
        return Success(res.pos, ''.join((head, *([x[1] for x in tail]))))

    def string_literal(self, pos: int) -> Result[str]:
        """Parse string value."""
        if isinstance(res := self.regex(pos, r'"(?:\\"|[^"])*"', re.MULTILINE), Failure):
            return res
        return Success(res.pos, res.value[1:-1])

    def identifier(self, pos: int) -> Result[str]:
        """Parse identifier."""
        return self.regex(pos, r'[a-zA-Z][a-zA-Z_0-9_]*')

    def space(self, pos: int) -> Result[str]:
        """Parse whitespace."""
        return self.regex(pos, r'(\s|/[*]([^*]|[*](?!/))*[*]/|//[^\n]*$)+', re.MULTILINE)

    def optspace(self, pos: int) -> Result[str]:
        """Parse optional whitespace."""
        return self.regex(pos, r'(\s|/[*]([^*]|[*](?!/))*[*]/|//[^\n]*$)*', re.MULTILINE)


def get_types_from_idl(text: str) -> Typesdict:
    """Get types from IDL message definition.

    Args:
        text: Message definition.

    Returns:
        Types dictionary.

    """
    return make_typesdict(IDLParser(text).specification())
