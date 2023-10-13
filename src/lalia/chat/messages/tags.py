from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import field
from itertools import chain

from pydantic import field_validator
from pydantic.dataclasses import dataclass


def derive_predicate(operand: Tag | TagPattern) -> Callable[[set[Tag]], bool]:
    match operand:
        case Tag() as tag:
            return lambda tags: tag in tags
        case TagPattern(key=key, value=re.Pattern() as pattern):
            return lambda tags: any(
                key.match(tag.key) and pattern.match(tag.value)
                if isinstance(tag.value, str)
                else False
                for tag in tags
            )
        case TagPattern(key=key, value=TagPattern() as tag_pattern):
            predicate = derive_predicate(tag_pattern)
            return lambda tags: any(
                key.match(tag.key) and predicate({tag.value})
                if isinstance(tag.value, Tag)
                else False
                for tag in tags
            )
    raise TypeError(f"No predicate defined for: '{type(operand)}'")


class _PredicateOperator(ABC):
    def __init__(self, *predicates: Callable[[set[Tag]], bool]):
        self.predicates = predicates

    @abstractmethod
    def __call__(self, tags: set[Tag]) -> bool:
        raise NotImplementedError

    def __and__(self, other: _PredicateOperator | Tag) -> _And:
        match other:
            case Tag():
                return _And(self, derive_predicate(other))
            case _PredicateOperator():
                return _And(self, other)
        raise TypeError(
            f"Unsupported operand type(s) for &: '{type(self)}' and '{type(other)}'"
        )

    def __or__(self, other: _PredicateOperator | Tag) -> _Or:
        match other:
            case Tag():
                return _Or(self, derive_predicate(other))
            case _PredicateOperator():
                return _Or(self, other)
        raise TypeError(
            f"Unsupported operand type(s) for |: '{type(self)}' and '{type(other)}'"
        )


class _And(_PredicateOperator):
    def __call__(self, tags: set[Tag]) -> bool:
        return all(predicate(tags) for predicate in self.predicates)


class _Or(_PredicateOperator):
    def __call__(self, tags: set[Tag]) -> bool:
        return any(predicate(tags) for predicate in self.predicates)


@dataclass(frozen=True)
class Tag:
    key: str
    value: str

    def __and__(self, other: Tag | TagPattern | _PredicateOperator) -> _And:
        match other:
            case _PredicateOperator():
                return _And(derive_predicate(self), other)
            case Tag() | TagPattern():
                return _And(derive_predicate(self), derive_predicate(other))
        raise TypeError(
            f"Unsupported operand type(s) for &: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )

    def __or__(self, other: Tag | TagPattern | _PredicateOperator) -> _Or:
        match other:
            case _PredicateOperator():
                return _Or(derive_predicate(self), other)
            case Tag() | TagPattern():
                return _Or(derive_predicate(self), derive_predicate(other))
        raise TypeError(
            f"Unsupported operand type(s) for |: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )


@dataclass(frozen=True)
class TagPattern:
    key: re.Pattern
    value: re.Pattern

    @field_validator("key", "value", mode="before")
    @classmethod
    def parse_value(cls, val: str | re.Pattern) -> re.Pattern:
        if isinstance(val, str):
            return re.compile(val)
        return val

    def __and__(self, other: Tag | TagPattern | _PredicateOperator) -> _And:
        match other:
            case _PredicateOperator():
                return _And(derive_predicate(self), other)
            case Tag() | TagPattern():
                return _And(derive_predicate(self), derive_predicate(other))
        raise TypeError(
            f"Unsupported operand type(s) for &: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )

    def __or__(self, other: Tag | TagPattern | _PredicateOperator) -> _Or:
        match other:
            case _PredicateOperator():
                return _Or(derive_predicate(self), other)
            case Tag() | TagPattern():
                return _Or(derive_predicate(self), derive_predicate(other))
        raise TypeError(
            f"Unsupported operand type(s) for |: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )
