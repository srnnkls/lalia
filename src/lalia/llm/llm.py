from collections.abc import Callable, Sequence
from typing import Any, Protocol, runtime_checkable

from lalia.chat.messages import Message
from lalia.llm.openai import ChatCompletionResponse, ChatModel, FunctionCallDirective


@runtime_checkable
class LLM(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        model: ChatModel | None = None,
        functions: Sequence[Callable[..., Any]] | None = None,
        function_call: FunctionCallDirective
        | dict[str, str] = FunctionCallDirective.AUTO,
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
    ) -> ChatCompletionResponse:
        ...

    def complete_raw(
        self,
        messages: Sequence[dict[str, Any]],
        model: ChatModel | None = None,
        functions: Sequence[dict[str, Any]] | None = None,
        function_call: FunctionCallDirective
        | dict[str, str] = FunctionCallDirective.AUTO,
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
    ) -> dict[str, Any]:
        ...
