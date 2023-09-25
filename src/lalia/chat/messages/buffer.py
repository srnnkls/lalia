from collections.abc import Iterator, Sequence

from rich.console import Console

from lalia.chat.messages.messages import Message
from lalia.io.renderers import MessageBufferRender

console = Console()


class MessageBuffer(Sequence[Message]):
    def __init__(self, messages: Sequence[Message] = (), *, verbose=False):
        self.messages = list(messages)
        self.pending = []
        self.verbose = verbose

        self._transactional_bounds: list[tuple[int, int]] = []

    def __getitem__(self, index: int) -> Message:
        return (self.messages + self.pending)[index]

    def __iter__(self) -> Iterator[Message]:
        yield from (self.messages + self.pending)

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

    def add_messages(self, messages: Sequence[Message]):
        for message in messages:
            self.add_message(message)

    def clear(self):
        self.messages = []
        self.pending = []

    def commit(self):
        self._transactional_bounds.append((len(self.messages), len(self.pending)))
        self.messages.extend(self.pending)
        self.pending = []

    def rollback(self):
        self.pending = []

    def revert(self, transaction: int = -1):
        if self._transactional_bounds:
            start, end = self._transactional_bounds.pop(transaction)
            self.pending = self.messages[start:end] + self.pending
            self.messages = self.messages[:start]
