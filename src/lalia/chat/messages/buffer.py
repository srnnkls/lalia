from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from itertools import chain

from rich.console import Console

from lalia.chat.messages.folds import DEFAULT_FOLD_TAGS, Folds, derive_tag_predicate
from lalia.chat.messages.messages import Message
from lalia.chat.messages.tags import (
    Tag,
    TagPattern,
)
from lalia.io.logging import get_logger
from lalia.io.renderers import MessageBufferRender

console = Console()

logger = get_logger(__name__)


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
        self._default_fold_tags = default_fold_tags
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
        return MessageBufferRender(
            self.messages,
            self.pending,
            self.folds.message_states,
            self.folds.pending_states,
        )._repr_mimebundle_(include, exclude, **kwargs)

    def add(self, message: Message | None):
        if message is not None:
            self.add_message(message)

    def add_message(self, message: Message):
        logger.debug(message)
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

    @property
    def default_fold_tags(
        self,
    ) -> set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool]:
        return self._default_fold_tags

    @default_fold_tags.setter
    def default_fold_tags(
        self, default_fold_tags: set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool]
    ):
        self._default_fold_tags = default_fold_tags
        self.folds.default_fold_tags = default_fold_tags
        self.folds.update(self.messages, self.pending)

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

        tag_predicate = derive_tag_predicate(tags)

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
