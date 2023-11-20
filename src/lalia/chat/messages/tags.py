from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from enum import StrEnum
from typing import ClassVar, TypeGuard, cast

from pydantic import Field, ValidationInfo, field_validator, validate_call
from pydantic.dataclasses import dataclass

from lalia.io.renderers import TagColor, TagRenderer


class DefaultTagKeys(StrEnum):
    ERROR = "error"
    FUNCTION = "function"


GROUP_COLORS_BY_KEY = True


def _is_tag_pattern_tuple(
    tags: object,
) -> TypeGuard[tuple[str | re.Pattern, str | re.Pattern]]:
    return isinstance(tags, tuple) and all(
        isinstance(t, str | re.Pattern) for t in tags
    )


def _is_tag_pattern_dict(
    tags: object,
) -> TypeGuard[dict[str | re.Pattern, str | re.Pattern]]:
    return isinstance(tags, dict) and all(
        isinstance(k, str | re.Pattern) and isinstance(v, str | re.Pattern)
        for k, v in tags.items()
    )


def _convert_tag_like_set(
    tags: set[tuple[str | re.Pattern, str | re.Pattern]]
    | set[dict[str | re.Pattern, str | re.Pattern]]
    | set[Tag]
    | set[TagPattern]
) -> set[TagPattern]:
    return {TagPattern.from_tag_like(tag) for tag in tags}


def convert_tag_like(
    tags: Tag
    | TagPattern
    | set[Tag]
    | set[TagPattern]
    | tuple[str | re.Pattern, str | re.Pattern]
    | dict[str | re.Pattern, str | re.Pattern]
    | set[tuple[str | re.Pattern, str | re.Pattern]]
    | set[dict[str | re.Pattern, str | re.Pattern]]
    | Callable[[set[Tag]], bool],
) -> set[TagPattern]:
    match tags:
        case Tag() | TagPattern():
            return {TagPattern.from_tag_like(tags)}
        case set() as tags_:
            return _convert_tag_like_set(tags_)
        case tuple() as tag:
            return {TagPattern.from_tag_like(tag)}
        case dict() as tag:
            return {TagPattern.from_tag_like(tag)}

    raise TypeError(f"Unsupported type for tags: '{type(tags).__name__}'")


class PredicateRegistry:
    _predicates: ClassVar[dict[Tag | TagPattern, Callable[[set[Tag]], bool]]] = {}

    @classmethod
    def register_predicate(
        cls,
        tag: Tag | TagPattern,
        predicate: Callable[[set[Tag]], bool],
    ):
        if tag not in cls._predicates:
            cls._predicates[tag] = predicate
        cls._predicates[tag]

    @classmethod
    def deregister_predicate(cls, tag: Tag | TagPattern):
        if tag in cls._predicates:
            cls._predicates.pop(tag)

    @classmethod
    def derive_predicate(cls, operand: Tag | TagPattern) -> Callable[[set[Tag]], bool]:
        """
        A higher-order function that returns a predicate function for a given
        operand. The returned predicate function takes a set of tags and returns
        True if the operand matches any of the tags in the set.
        """
        match operand:
            case Tag() as tag:

                def is_in_tags(tags: set[Tag]) -> bool:
                    return tag in tags

                if tag not in cls._predicates:
                    cls.register_predicate(tag, is_in_tags)

                return cls._predicates[tag]

            case TagPattern(
                key=re.Pattern() as key_pattern, value=re.Pattern() as value_pattern
            ) as tag_pattern:

                def matches_any_tag(tags: set[Tag]) -> bool:
                    return any(
                        key_pattern.match(tag.key) and value_pattern.match(tag.value)
                        for tag in tags
                    )

                if tag_pattern not in cls._predicates:
                    cls.register_predicate(tag_pattern, matches_any_tag)

                return cls._predicates[tag_pattern]

        raise TypeError(f"No predicate defined for: '{type(operand)}'")


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
                return _And(self, PredicateRegistry.derive_predicate(other))
            case _PredicateOperator():
                return _And(self, other)
        raise TypeError(
            f"Unsupported operand type(s) for &: '{type(self)}' and '{type(other)}'"
        )

    def __or__(self, other: _PredicateOperator | Tag) -> _Or:
        match other:
            case Tag():
                return _Or(self, PredicateRegistry.derive_predicate(other))
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
                return _And(PredicateRegistry.derive_predicate(self), other)
            case Tag() | TagPattern():
                return _And(
                    PredicateRegistry.derive_predicate(self),
                    PredicateRegistry.derive_predicate(other),
                )
        raise TypeError(
            f"Unsupported operand type(s) for &: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )

    def __or__(self, other: Tag | TagPattern | _PredicateOperator) -> _Or:
        match other:
            case _PredicateOperator():
                return _Or(PredicateRegistry.derive_predicate(self), other)
            case Tag() | TagPattern():
                return _Or(
                    PredicateRegistry.derive_predicate(self),
                    PredicateRegistry.derive_predicate(other),
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
    def from_dict(cls, tag: dict[str | re.Pattern, str | re.Pattern]) -> TagPattern:
        key, value = next(iter(tag.items()))
        return cls(key=re.compile(key), value=re.compile(value))

    @classmethod
    def from_tag_like(
        cls,
        tag_like: Tag
        | TagPattern
        | tuple[str | re.Pattern, str | re.Pattern]
        | dict[str | re.Pattern, str | re.Pattern],
    ) -> TagPattern:
        match tag_like:
            case Tag(str() | re.Pattern() as key, str() | re.Pattern() as value):
                return cls(key, value)
            case TagPattern(
                str() | re.Pattern() as key, str() | re.Pattern() as value
            ) as tag_pattern:
                return tag_pattern
            case tuple() as tag_tuple:
                if _is_tag_pattern_tuple(tag_tuple):
                    return cls(*tag_tuple)
                raise TypeError(f"Incompatible tuple for tag_like: '{tag_tuple!r}'")
            case dict() as tag_dict:
                if _is_tag_pattern_dict(tag_dict):
                    return cls.from_dict(tag_dict)
                raise TypeError(f"Incompatible dict for tag_like: '{tag_dict!r}'")

        raise TypeError(f"Unsupported type for tag_like: '{type(tag_like).__name__}'")

    @field_validator("key", "value", mode="before")
    @classmethod
    def parse_value(cls, val: str | re.Pattern) -> re.Pattern:
        if isinstance(val, str):
            return re.compile(val)
        return val

    def __and__(self, other: Tag | TagPattern | _PredicateOperator) -> _And:
        match other:
            case _PredicateOperator():
                return _And(PredicateRegistry.derive_predicate(self), other)
            case Tag() | TagPattern():
                return _And(
                    PredicateRegistry.derive_predicate(self),
                    PredicateRegistry.derive_predicate(other),
                )
        raise TypeError(
            f"Unsupported operand type(s) for &: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )

    def __or__(self, other: Tag | TagPattern | _PredicateOperator) -> _Or:
        match other:
            case _PredicateOperator():
                return _Or(PredicateRegistry.derive_predicate(self), other)
            case Tag() | TagPattern():
                return _Or(
                    PredicateRegistry.derive_predicate(self),
                    PredicateRegistry.derive_predicate(other),
                )
        raise TypeError(
            f"Unsupported operand type(s) for |: '{type(self).__name__}' "
            f"and '{type(other).__name__}'"
        )
