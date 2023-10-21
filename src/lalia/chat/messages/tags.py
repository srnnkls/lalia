from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from typing import ClassVar, cast

from pydantic import Field, ValidationInfo, field_validator, validate_call
from pydantic.dataclasses import dataclass

from lalia.io.renderers import TagColor, TagRenderer

GROUP_COLORS_BY_KEY = True


class PredicateRegistry:
    _predicates: ClassVar[dict[Tag | TagPattern, Callable[[set[Tag]], bool]]] = {}

    def register_predicate(
        self,
        tag: Tag | TagPattern,
        predicate: Callable[[set[Tag]], bool],
    ) -> Callable[[set[Tag]], bool]:
        if tag not in self._predicates:
            self._predicates[tag] = predicate
        return self._predicates[tag]

    def deregister_predicate(self, tag: Tag | TagPattern):
        if tag in self._predicates:
            self._predicates.pop(tag)

    def derive_predicate(self, operand: Tag | TagPattern) -> Callable[[set[Tag]], bool]:
        """
        A higher-order function that returns a predicate function for a given
        operand. The returned predicate function takes a set of tags and returns
        True if the operand matches any of the tags in the set.
        """
        match operand:
            case Tag() as tag:

                def is_in_tags(tags: set[Tag]) -> bool:
                    return tag in tags

                return self.register_predicate(tag, is_in_tags)
            case TagPattern(
                key=re.Pattern() as key_pattern, value=re.Pattern() as value_pattern
            ) as tag_pattern:

                def matches_any_tag(tags: set[Tag]) -> bool:
                    return any(
                        key_pattern.match(tag.key) and value_pattern.match(tag.value)
                        for tag in tags
                    )

                return self.register_predicate(tag_pattern, matches_any_tag)

        raise TypeError(f"No predicate defined for: '{type(operand)}'")


predicate_registry = PredicateRegistry()


class _PredicateOperator(ABC):
    def __init__(self, *predicates: Callable[[set[Tag]], bool]):
        self.predicates = predicates

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _PredicateOperator):
            return NotImplemented
        return set(self.predicates) == set(other.predicates)

    @abstractmethod
    def __call__(self, tags: set[Tag]) -> bool:
        ...

    def __and__(self, other: _PredicateOperator | Tag) -> _And:
        match other:
            case Tag():
                return _And(self, predicate_registry.derive_predicate(other))
            case _PredicateOperator():
                return _And(self, other)
        raise TypeError(
            f"Unsupported operand type(s) for &: '{type(self)}' and '{type(other)}'"
        )

    def __or__(self, other: _PredicateOperator | Tag) -> _Or:
        match other:
            case Tag():
                return _Or(self, predicate_registry.derive_predicate(other))
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
    color: TagColor | None = Field(validate_default=True, default=None)

    group_colors_by_key: ClassVar[bool] = GROUP_COLORS_BY_KEY

    def __iter__(self) -> Iterator[str]:
        yield from (self.key, self.value)

    @field_validator("color", mode="before")
    @classmethod
    def set_color(cls, color: TagColor, info: ValidationInfo) -> TagColor:
        key = info.data["key"]

        if color is None:
            color = TagRenderer.get_color(key)

        if cls.group_colors_by_key:
            cls.register_key_color(key, color)

        return color

    @classmethod
    def from_dict(cls, tag: dict[str, str]) -> Tag:
        key, value = next(iter(tag.items()))
        return cls(key=key, value=value)

    @classmethod
    @validate_call
    def register_key_color(cls, key: str, color: TagColor):
        TagRenderer.register_key(key, color)

    def __and__(self, other: Tag | TagPattern | _PredicateOperator) -> _And:
        match other:
            case _PredicateOperator():
                return _And(predicate_registry.derive_predicate(self), other)
            case Tag() | TagPattern():
                return _And(
                    predicate_registry.derive_predicate(self),
                    predicate_registry.derive_predicate(other),
                )
        raise TypeError(
            f"Unsupported operand type(s) for &: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )

    def __or__(self, other: Tag | TagPattern | _PredicateOperator) -> _Or:
        match other:
            case _PredicateOperator():
                return _Or(predicate_registry.derive_predicate(self), other)
            case Tag() | TagPattern():
                return _Or(
                    predicate_registry.derive_predicate(self),
                    predicate_registry.derive_predicate(other),
                )
        raise TypeError(
            f"Unsupported operand type(s) for |: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )


@dataclass(frozen=True)
class TagPattern:
    key: re.Pattern | str
    value: re.Pattern | str

    def __iter__(self) -> Iterator[re.Pattern]:
        yield from (cast(re.Pattern, self.key), cast(re.Pattern, self.value))

    @classmethod
    def from_dict(
        cls, tag: dict[str, str] | dict[re.Pattern, re.Pattern]
    ) -> TagPattern:
        key, value = next(iter(tag.items()))
        return cls(key=re.compile(key), value=re.compile(value))

    @field_validator("key", "value", mode="before")
    @classmethod
    def parse_value(cls, val: str | re.Pattern) -> re.Pattern:
        if isinstance(val, str):
            return re.compile(val)
        return val

    def __and__(self, other: Tag | TagPattern | _PredicateOperator) -> _And:
        match other:
            case _PredicateOperator():
                return _And(predicate_registry.derive_predicate(self), other)
            case Tag() | TagPattern():
                return _And(
                    predicate_registry.derive_predicate(self),
                    predicate_registry.derive_predicate(other),
                )
        raise TypeError(
            f"Unsupported operand type(s) for &: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )

    def __or__(self, other: Tag | TagPattern | _PredicateOperator) -> _Or:
        match other:
            case _PredicateOperator():
                return _Or(predicate_registry.derive_predicate(self), other)
            case Tag() | TagPattern():
                return _Or(
                    predicate_registry.derive_predicate(self),
                    predicate_registry.derive_predicate(other),
                )
        raise TypeError(
            f"Unsupported operand type(s) for |: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )
