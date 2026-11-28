# Copyright 2020-2026 Ternaris
# SPDX-License-Identifier: Apache-2.0
"""PEG Parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar, cast, overload

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, TypeAlias

    Result: TypeAlias = 'Success[T] | Failure'
    Rule: TypeAlias = 'Callable[[int], Result[T]]'

    T1 = TypeVar('T1')
    T2 = TypeVar('T2')
    T3 = TypeVar('T3')
    T4 = TypeVar('T4')
    T5 = TypeVar('T5')
    T6 = TypeVar('T6')

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)


@dataclass(frozen=True)
class Success(Generic[T_co]):
    """Parse success."""

    pos: int
    value: T_co


@dataclass(frozen=True)
class Failure:
    """Parse failure."""

    pos: int
    context: str
    expected: set[str]


class Parser:
    """PEG Parser."""

    def __init__(self, text: str) -> None:
        """Initialize."""
        self.text = text

    def literal(self, pos: int, text: str) -> Result[str]:
        """Match a literal."""
        if self.text.startswith(text, pos):
            return Success(pos + len(text), text)
        return Failure(pos, self.text[pos : min(len(self.text), pos + 32)], {repr(text)})

    def regex(self, pos: int, pattern: str, flags: int = 0) -> Result[str]:
        """Match a regex."""
        rx = re.compile(pattern, flags)
        if res := rx.match(self.text, pos):
            return Success(res.span()[1], res.group())
        return Failure(pos, self.text[pos : min(len(self.text), pos + 32)], {f'/{pattern}/'})

    def optional(self, pos: int, rule: Rule[T]) -> Result[T | None]:
        """Make rule optional."""
        if isinstance(res := rule(pos), Success):
            return cast('Result[T | None]', res)
        return Success(pos, cast('T | None', None))

    def many0(self, pos: int, rule: Rule[T]) -> Success[list[T]]:
        """Apply rule zero or more times."""
        out = []
        while pos < len(self.text):
            if isinstance(res := rule(pos), Success):
                pos = res.pos
                out.append(res.value)
                continue
            break
        return Success(pos, out)

    def many1(self, pos: int, rule: Rule[T]) -> Result[list[T]]:
        """Apply rule one or more times."""
        if isinstance(res := rule(pos), Success):
            head = res
            tail = self.many0(res.pos, rule)
            return Success(tail.pos, [head.value, *tail.value])
        return res

    @overload
    def any(
        self,
        pos: int,
        rules: tuple[Rule[T1], Rule[T2]],
    ) -> Result[T1 | T2]: ...

    @overload
    def any(
        self,
        pos: int,
        rules: tuple[Rule[T1], Rule[T2], Rule[T3]],
    ) -> Result[T1 | T2 | T3]: ...

    @overload
    def any(
        self,
        pos: int,
        rules: tuple[Rule[T1], Rule[T2], Rule[T3], Rule[T4]],
    ) -> Result[T1 | T2 | T3 | T4]: ...

    @overload
    def any(
        self,
        pos: int,
        rules: tuple[Rule[T1], Rule[T2], Rule[T3], Rule[T4], Rule[T5]],
    ) -> Result[T1 | T2 | T3 | T4 | T5]: ...

    @overload
    def any(
        self,
        pos: int,
        rules: tuple[Rule[T1], Rule[T2], Rule[T3], Rule[T4], Rule[T5], Rule[T6]],
    ) -> Result[T1 | T2 | T3 | T4 | T5 | T6]: ...

    def any(self, pos: int, rules: Any) -> Any:  # type: ignore[explicit-any]
        """Match first successful rule."""
        error = Failure(0, '', set())
        for rule in rules:
            if isinstance(res := rule(pos), Success):
                return res

            if res.pos == error.pos:
                error = Failure(error.pos, error.context, error.expected | res.expected)
            elif res.pos > error.pos:
                error = res
        return error

    @overload
    def all(
        self,
        pos: int,
        rules: tuple[Rule[T1], Rule[T2]],
    ) -> Result[tuple[T1, T2]]: ...

    @overload
    def all(
        self,
        pos: int,
        rules: tuple[Rule[T1], Rule[T2], Rule[T3]],
    ) -> Result[tuple[T1, T2, T3]]: ...

    @overload
    def all(
        self,
        pos: int,
        rules: tuple[Rule[T1], Rule[T2], Rule[T3], Rule[T4]],
    ) -> Result[tuple[T1, T2, T3, T4]]: ...

    @overload
    def all(
        self,
        pos: int,
        rules: tuple[Rule[T1], Rule[T2], Rule[T3], Rule[T4], Rule[T5]],
    ) -> Result[tuple[T1, T2, T3, T4, T5]]: ...

    @overload
    def all(
        self,
        pos: int,
        rules: tuple[Rule[T1], Rule[T2], Rule[T3], Rule[T4], Rule[T5], Rule[T6]],
    ) -> Result[tuple[T1, T2, T3, T4, T5, T6]]: ...

    def all(self, pos: int, rules: tuple[Rule[Any], ...]) -> Result[tuple[Any, ...]]:  # type: ignore[explicit-any]
        """Match all rules."""
        items = []
        for rule in rules:
            if isinstance(res := rule(pos), Failure):
                return res

            pos = res.pos
            items.append(res.value)
        return Success(pos, tuple(items))
