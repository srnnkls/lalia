from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import field
from itertools import chain
from typing import Any

from pydantic import field_serializer, field_validator
from pydantic.dataclasses import dataclass

from lalia.chat.messages.fold_state import FoldState
from lalia.chat.messages.messages import Message
from lalia.chat.messages.tags import (
    PredicateRegistry,
    Tag,
    TagPattern,
    convert_tag_like,
)
from lalia.io.serialization.functions import parse_callable, serialize_callable

DEFAULT_FOLD_TAGS = {TagPattern("error", ".*")}


def derive_tag_predicate(
    tags: (
        Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | tuple[str | re.Pattern, str | re.Pattern]
        | dict[str | re.Pattern, str | re.Pattern]
        | set[tuple[str | re.Pattern, str | re.Pattern]]
        | set[dict[str | re.Pattern, str | re.Pattern]]
        | Callable[[set[Tag]], bool]
    ),
) -> Callable[[set[Tag]], bool]:
    if not callable(tags):
        tags = convert_tag_like(tags)

    match tags:
        case Tag() | TagPattern():
            return PredicateRegistry.derive_predicate(tags)
        case set() as tag_likes:
            return lambda message_tags: any(
                PredicateRegistry.derive_predicate(tag)(message_tags)
                for tag in tag_likes
            )
        case Callable() as predicate:
            return predicate
        case _:
            raise TypeError(f"Unsupported type for tags: '{type(tags).__name__}'")


@dataclass
class Fold:
    predicate: Callable[[set[Tag]], bool]
    state: FoldState = FoldState.UNFOLDED

    def __invert__(self) -> Fold:
        return Fold(self.predicate, ~self.state)


@dataclass
class Folds:
    message_states: list[FoldState] = field(default_factory=list)
    pending_states: list[FoldState] = field(default_factory=list)
    default_fold_tags: set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool] = field(
        default_factory=lambda: DEFAULT_FOLD_TAGS
    )

    @field_serializer("default_fold_tags")
    def serialize_default_fold_tags(
        self, tags: set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool]
    ) -> list[Tag | TagPattern] | dict[str, Any]:
        if not callable(tags):
            return list(tags)
        if callable(callable_ := tags):
            return serialize_callable(callable_)
        raise AssertionError("Unreachable")

    @field_validator("default_fold_tags", mode="before")
    @classmethod
    def _parse_default_fold_tags(
        cls,
        tags: (
            set[Tag]
            | set[TagPattern]
            | Callable[[set[Tag]], bool]
            | list[dict[str, Any]]  # serialized tags
            | dict[str, Any]
        ),  # serialized callable
    ) -> set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool]:
        match tags:
            case set() as tags:
                return tags
            case Callable() as callable_:
                return callable_
            case list() as tags:
                return {TagPattern(**tag) for tag in tags}
            case dict() as callable_:
                return parse_callable(callable_)
            case _:
                raise TypeError(f"Unsupported type for tags: '{type(tags).__name__}'")

    def __post_init__(self):
        self._folds: list[Fold] = []

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Message],
        pending: Sequence[Message],
        default_fold_tags: set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool],
    ):
        return cls(
            [
                (
                    FoldState.FOLDED
                    if derive_tag_predicate(default_fold_tags)(message.tags)
                    else FoldState.UNFOLDED
                )
                for message in messages
            ],
            [
                (
                    FoldState.FOLDED
                    if derive_tag_predicate(default_fold_tags)(message.tags)
                    else FoldState.UNFOLDED
                )
                for message in pending
            ],
            default_fold_tags,
        )

    @property
    def _default_folds(self) -> Fold:
        if self.default_fold_tags:
            return Fold(
                predicate=derive_tag_predicate(self.default_fold_tags),
                state=FoldState.FOLDED,
            )
        else:
            return Fold(
                predicate=lambda tags: False,
                state=FoldState.UNFOLDED,
            )

    def add(self, message: Message):
        self.pending_states.append(self.get_fold_state(message))

    def apply(
        self, messages: Sequence[Message], pending: Sequence[Message]
    ) -> tuple[Iterator[Message], Iterator[Message]]:
        return (
            message
            for message, fold in zip(messages, self.message_states, strict=True)
            if fold is FoldState.UNFOLDED
        ), (
            message
            for message, fold in zip(pending, self.pending_states, strict=True)
            if fold is FoldState.UNFOLDED
        )

    def clear(self, messages: Sequence[Message], pending: Sequence[Message]):
        self._folds.clear()
        self.update(messages, pending)

    def commit(self):
        self.message_states.extend(self.pending_states)
        self.pending_states = []

    @contextmanager
    def expand(
        self,
        tags: (
            Tag
            | TagPattern
            | set[Tag]
            | set[TagPattern]
            | tuple[str | re.Pattern, str | re.Pattern]
            | dict[str | re.Pattern, str | re.Pattern]
            | set[tuple[str | re.Pattern, str | re.Pattern]]
            | set[dict[str | re.Pattern, str | re.Pattern]]
            | Callable[[set[Tag]], bool]
        ),
        messages: Sequence[Message],
        pending: Sequence[Message],
    ):
        self.unfold(tags, messages, pending)
        yield self
        self.fold(tags, messages, pending)

    def get_fold_state(self, message: Message) -> FoldState:
        folds = chain(reversed(self._folds), [self._default_folds])
        return next(
            (fold.state for fold in folds if fold.predicate(message.tags)),
            FoldState.UNFOLDED,
        )

    def fold(
        self,
        tags: (
            Tag
            | TagPattern
            | set[Tag]
            | set[TagPattern]
            | tuple[str | re.Pattern, str | re.Pattern]
            | dict[str | re.Pattern, str | re.Pattern]
            | set[tuple[str | re.Pattern, str | re.Pattern]]
            | set[dict[str | re.Pattern, str | re.Pattern]]
            | Callable[[set[Tag]], bool]
            | None
        ),
        messages: Sequence[Message],
        pending: Sequence[Message],
    ):
        if not tags:
            self._folds.clear()
        else:
            fold = Fold(
                predicate=derive_tag_predicate(tags),
                state=FoldState.FOLDED,
            )
            if fold in self._folds:
                self._folds.remove(fold)
            self._folds.append(fold)

        self.update(messages, pending)

    def revert(self, start: int, end: int):
        self.pending_states = self.message_states[start:end] + self.pending_states
        self.message_states = self.message_states[:start]

    def rollback(self):
        self.pending_states = []

    def unfold(
        self,
        tags: (
            Tag
            | TagPattern
            | set[Tag]
            | set[TagPattern]
            | tuple[str | re.Pattern, str | re.Pattern]
            | dict[str | re.Pattern, str | re.Pattern]
            | set[tuple[str | re.Pattern, str | re.Pattern]]
            | set[dict[str | re.Pattern, str | re.Pattern]]
            | Callable[[set[Tag]], bool]
            | None
        ),
        messages: Sequence[Message],
        pending: Sequence[Message],
    ):
        if not tags:
            self._folds.clear()
        else:
            if not callable(tags):
                tags = convert_tag_like(tags)

            unfold = Fold(
                predicate=derive_tag_predicate(tags),
                state=FoldState.UNFOLDED,
            )
            if ~unfold in self._folds:
                self._folds.remove(~unfold)
            else:
                self._folds.append(unfold)

        self.update(messages, pending)

    def update(self, messages: Sequence[Message], pending: Sequence[Message]):
        def get_fold_states(messages: Sequence[Message]) -> list[FoldState]:
            return [self.get_fold_state(message) for message in messages]

        self.message_states = get_fold_states(messages)
        self.pending_states = get_fold_states(pending)
