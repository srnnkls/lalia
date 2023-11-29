from collections.abc import Callable, Iterable, Sequence
from typing import Any, Protocol, runtime_checkable

from lalia.chat.messages import Message
from lalia.llm.openai import ChatCompletionResponse, ChatModel, FunctionCallDirective


@runtime_checkable
class LLM(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        functions: Iterable[Callable[..., Any]] | None = None,
        function_call: FunctionCallDirective
        | dict[str, str] = FunctionCallDirective.AUTO,
        n_choices: int = 1,
        temperature: float | None = None,
        model: ChatModel | None = None,
    ) -> ChatCompletionResponse:
        ...

    def complete_raw(
        self,
        messages: Sequence[dict[str, Any]],
        functions: Iterable[dict[str, Any]] | None = None,
        function_call: FunctionCallDirective
        | dict[str, str] = FunctionCallDirective.AUTO,
        n_choices: int = 1,
        temperature: float | None = None,
        model: ChatModel | None = None,
    ) -> dict[str, Any]:
        ...
