from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import field, fields
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from lalia.chat.finish_reason import FinishReason
from lalia.chat.messages import Message
from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.tags import TagPattern
from lalia.llm.openai import ChatCompletionResponse, ChatModel, FunctionCallDirective

if TYPE_CHECKING:
    from lalia.chat.session import Session


@runtime_checkable
class LLMCallback(Protocol):
    def __call__(
        self,
        messages: Sequence[Message],
        context: set[TagPattern],
        model: ChatModel,
        functions: Sequence[Callable[..., Any]] = (),
        function_call: (
            FunctionCallDirective | dict[str, str]
        ) = FunctionCallDirective.AUTO,
        logit_bias: dict[str, float] | None = None,
        max_tokens: int | None = None,
        n_choices: int = 1,
        presence_penalty: float | None = None,
        # response_format: ResponseFormat | None = None # NOT SUPPORTED
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        # stream: bool = False, # NOT SUPPORTED
        temperature: float | None = None,
        # tools: Sequence[Tool] | None = None, # NOT SUPPORTED
        # tool_choice: ToolChoice | None = None, # NOT SUPPORTED
        top_p: float | None = None,
        user: str | None = None,
        timeout: int | None = None,
    ) -> ChatCompletionResponse: ...


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class DispatchCall:
    callback: LLMCallback
    messages: MessageBuffer
    context: set[TagPattern] = field(default_factory=set)
    params: dict[str, Any] = field(default_factory=dict)
    finish_reason: FinishReason = FinishReason.DELEGATE

    def __iter__(self):
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
