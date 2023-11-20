from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import field, fields
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from lalia.chat.finish_reason import FinishReason
from lalia.chat.messages import Message
from lalia.chat.messages.buffer import MessageBuffer
from lalia.llm.openai import ChatCompletionResponse, ChatModel, FunctionCallDirective

if TYPE_CHECKING:
    from lalia.chat.session import Session


@runtime_checkable
class LLMCallback(Protocol):
    def __call__(
        self,
        messages: Sequence[Message],
        functions: Sequence[Callable[..., Any]] | None = None,
        function_call: FunctionCallDirective
        | dict[str, str] = FunctionCallDirective.NONE,
        n_choices: int = 1,
        temperature: float | None = None,
        model: ChatModel | None = None,
    ) -> ChatCompletionResponse:
        ...


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DispatchCall:
    callback: LLMCallback
    messages: MessageBuffer
    params: dict[str, Any] = field(default_factory=dict)
    finish_reason: FinishReason = FinishReason.DELEGATE

    def __iter__(self):
        yield from (getattr(self, field.name) for field in fields(self))


@runtime_checkable
class Dispatcher(Protocol):
    def dispatch(self, session: Session) -> DispatchCall:
        ...

    def reset(self):
        ...


class FunctionsDispatcher:
    def dispatch(self, session: Session) -> DispatchCall:
        return DispatchCall(
            callback=session.llm.complete,
            messages=session.messages,
        )

    def reset(self):
        pass
