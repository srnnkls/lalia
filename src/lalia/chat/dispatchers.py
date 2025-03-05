from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field, fields
from typing import TYPE_CHECKING, Protocol, Unpack, runtime_checkable

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from lalia.chat.finish_reason import FinishReason
from lalia.chat.messages import Message
from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.tags import TagPattern
from lalia.llm.llm import ChatCompletionResponse, CompleteKwargs

if TYPE_CHECKING:
    from lalia.chat.session import Session


@runtime_checkable
class LLMCallback(Protocol):
    def __call__(
        self,
        messages: Sequence[Message],
        context: set[TagPattern] | None = None,
        **kwargs: Unpack[CompleteKwargs],
    ) -> ChatCompletionResponse: ...


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DispatchCall:
    callback: LLMCallback
    messages: MessageBuffer
    context: set[TagPattern] = field(default_factory=set)
    kwargs: CompleteKwargs = field(default_factory=CompleteKwargs)
    finish_reason: FinishReason = FinishReason.DELEGATE

    def __iter__(
        self,
    ):
        yield from (getattr(self, field.name) for field in fields(self))


@runtime_checkable
class Dispatcher(Protocol):
    def dispatch(self, session: Session) -> DispatchCall: ...

    def reset(self): ...


@dataclass
class NopDispatcher:
    def dispatch(self, session: Session) -> DispatchCall:
        return DispatchCall(
            callback=session.llm.complete,
            messages=session.messages,
        )

    def reset(self):
        pass
