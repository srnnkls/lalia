from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import field
from enum import Enum
from itertools import chain

from pydantic.dataclasses import dataclass
from rich.console import Console

from lalia.chat.messages.messages import Message
from lalia.chat.messages.tags import (
    Tag,
    TagPattern,
    predicate_registry,
)
from lalia.io.renderers import MessageBufferRender

DEFAULT_FOLD_TAGS = TagPattern("function", ".*") | TagPattern("error", ".*")

console = Console()


class FoldState(Enum):
    UNFOLDED = 0
    FOLDED = 1

    def __invert__(self) -> FoldState:
        return FoldState.UNFOLDED if self is FoldState.FOLDED else FoldState.FOLDED


@dataclass
class Fold:
    predicate: Callable[[set[Tag]], bool]
    state: FoldState = FoldState.UNFOLDED

    def __invert__(self) -> Fold:
        return Fold(self.predicate, ~self.state)


def _derive_tag_predicate(
    tags: Tag | TagPattern | set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool]
) -> Callable[[set[Tag]], bool]:
    match tags:
        case Tag() | TagPattern():
            return predicate_registry.derive_predicate(tags)
        case set(tags_):
            return lambda message_tags: all(
                predicate_registry.derive_predicate(tag)(message_tags) for tag in tags_
            )
        case Callable() as predicate:
            return predicate
        case _:
            raise TypeError(f"Unsupported type for tags: '{type(tags).__name__}'")


@dataclass
class Folds:
    messages: list[FoldState] = field(default_factory=list)
    pending: list[FoldState] = field(default_factory=list)
    default_fold_tags: set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool] = field(
        default_factory=lambda: DEFAULT_FOLD_TAGS
    )

    def __post_init__(self):
        self._folds: list[Fold] = [self._default_folds]

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Message],
        pending: Sequence[Message],
        default_fold_tags: set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool],
    ):
        return cls(
            [
                FoldState.FOLDED
                if _derive_tag_predicate(default_fold_tags)(message.tags)
                else FoldState.UNFOLDED
                for message in messages
            ],
            [
                FoldState.FOLDED
                if _derive_tag_predicate(default_fold_tags)(message.tags)
                else FoldState.UNFOLDED
                for message in pending
            ],
            default_fold_tags,
        )

    @property
    def _default_folds(self) -> Fold:
        return Fold(
            predicate=_derive_tag_predicate(self.default_fold_tags),
            state=FoldState.FOLDED,
        )

    def add(self, message: Message):
        self.pending.append(self.get_fold_state(message))

    def apply(
        self, messages: Sequence[Message], pending: Sequence[Message]
    ) -> tuple[Iterator[Message], Iterator[Message]]:
        return (
            message
            for message, fold in zip(messages, self.messages, strict=True)
            if not fold
        ), (
            message
            for message, fold in zip(pending, self.pending, strict=True)
            if not fold
        )

    def commit(self):
        self.messages.extend(self.pending)
        self.pending = []

    @contextmanager
    def expand(
        self,
        tags: Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | Callable[[set[Tag]], bool],
        messages: Sequence[Message],
        pending: Sequence[Message],
    ):
        self.unfold(tags, messages, pending)
        yield self
        self.fold(tags, messages, pending)

    def get_fold_state(self, message: Message) -> FoldState:
        return next(
            (
                fold.state
                for fold in reversed(self._folds)
                if fold.predicate(message.tags)
            ),
            FoldState.UNFOLDED,
        )

    def fold(
        self,
        tags: Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | Callable[[set[Tag]], bool]
        | None,
        messages: Sequence[Message],
        pending: Sequence[Message],
    ):
        if tags is None:
            self._folds = [self._default_folds]
        else:
            fold = Fold(
                predicate=_derive_tag_predicate(tags),
                state=FoldState.FOLDED,
            )
            self._folds.append(fold)

        self.update(messages, pending)

    def revert(self, start: int, end: int):
        self.pending = self.messages[start:end] + self.pending
        self.messages = self.messages[:start]

    def rollback(self):
        self.pending = []

    def unfold(
        self,
        tags: Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | Callable[[set[Tag]], bool]
        | None,
        messages: Sequence[Message],
        pending: Sequence[Message],
    ):
        if tags is None:
            self._folds = [self._default_folds]
        else:
            unfold = Fold(
                predicate=_derive_tag_predicate(tags),
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

        self.messages = get_fold_states(messages)
        self.pending = get_fold_states(pending)


class MessageBuffer(Sequence[Message]):
    def __init__(
        self,
        messages: Sequence[Message] = (),
        pending: Sequence[Message] = (),
        *,
        verbose=False,
        default_fold_tags: set[Tag]
        | set[TagPattern]
        | Callable[[set[Tag]], bool] = DEFAULT_FOLD_TAGS,
    ):
        self.messages = list(messages)
        self.pending = list(pending)
        self.default_fold_state = default_fold_tags
        self.folds = Folds.from_messages(
            messages, pending, default_fold_tags=default_fold_tags
        )
        self._transactional_bounds: list[tuple[int, int]] = []
        self.verbose = verbose

    def __getitem__(self, index: int) -> Message:
        return (self.messages + self.pending)[index]

    def __iter__(self) -> Iterator[Message]:
        messages, pending = self.folds.apply(self.messages, self.pending)
        yield from chain(messages, pending)

    def __len__(self) -> int:
        return len(self.messages + self.pending)

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageBufferRender(self.messages, self.pending)._repr_mimebundle_(
            include, exclude, **kwargs
        )

    def add(self, message: Message | None):
        if message is not None:
            self.add_message(message)

    def add_message(self, message: Message):
        if self.verbose:
            console.print(message)
        self.pending.append(message)
        self.folds.add(message)

    def add_messages(self, messages: Sequence[Message]):
        for message in messages:
            self.add_message(message)

    def clear(self):
        self.messages = []
        self.pending = []
        self._transactional_bounds = []
        self.folds = Folds()

    def commit(self):
        self._transactional_bounds.append(
            (len(self.messages), len(self.messages) + len(self.pending))
        )
        self.messages.extend(self.pending)
        self.pending = []
        self.folds.commit()

    @contextmanager
    def expand(
        self,
        tags: Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | Callable[[set[Tag]], bool],
    ):
        with self.folds.expand(tags, self.messages, self.pending):
            yield self

    def filter(
        self,
        predicate: Callable[[Message], bool] = lambda message: True,
        tags: Tag
        | TagPattern
        | Callable[[set[Tag]], bool]
        | set[Tag]
        | dict[str, str] = lambda tags: True,
    ) -> MessageBuffer:
        def filter_messages(messages: Sequence[Message]) -> list[Message]:
            return [
                message
                for message in messages
                if tag_predicate(message.tags) and predicate(message)
            ]

        if isinstance(tags, dict):
            tags = TagPattern.from_dict(tags)

        tag_predicate = _derive_tag_predicate(tags)

        messages = filter(
            lambda message: tag_predicate(message.tags) and predicate(message),
            self.messages,
        )
        messages = filter_messages(self.messages)
        pending = filter_messages(self.pending)

        return MessageBuffer(
            messages,
            pending,
            verbose=self.verbose,
            default_fold_tags=self.folds.default_fold_tags,
        )

    def fold(
        self,
        tags: Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | Callable[[set[Tag]], bool]
        | None = None,
    ):
        self.folds.fold(tags, self.messages, self.pending)

    def unfold(
        self,
        tags: Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | Callable[[set[Tag]], bool]
        | None = None,
    ):
        self.folds.unfold(tags, self.messages, self.pending)

    def rollback(self):
        self.pending = []
        self.folds.rollback()

    def revert(self):
        if self._transactional_bounds:
            start, end = self._transactional_bounds.pop()
            self.pending = self.messages[start:end] + self.pending
            self.folds.revert(start, end)
