from collections.abc import Iterator, Sequence

from lalia.chat.messages.messages import Message
from lalia.io.renderers import MessageBufferRender


class MessageBuffer(Sequence[Message]):
    def __init__(self, messages: Sequence[Message] = ()):
        self.messages = list(messages)
        self.pending = []

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

    def add(self, message: Message):
        self.pending.append(message)

    def commit(self):
        self._transactional_bounds.append((len(self.messages), len(self.pending)))
        self.messages.extend(self.pending)
        self.pending = []

    def rollback(self):
        self.pending = []

    def reset(self):
        self.messages = []
        self.pending = []

    def revert(self):
        if self._transactional_bounds:
            start, end = self._transactional_bounds.pop()
            self.messages = self.messages[:start]
            self.pending = self.pending[:end]
