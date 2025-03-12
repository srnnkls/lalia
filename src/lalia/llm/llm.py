import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Any, ParamSpec, Protocol, TypeVar, runtime_checkable

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

from lalia.chat.completions import Choice
from lalia.chat.messages import Message
from lalia.chat.messages.tags import TagPattern
from lalia.functions import Function
from lalia.llm.models import ChatModel

P = ParamSpec("P")
R_co = TypeVar("R_co", covariant=True)


class Wrapped(Protocol[P, R_co]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co: ...


class Decorator(Protocol[P, R_co]):
    def __call__(self, func: Wrapped[P, R_co]) -> Wrapped[P, R_co]: ...


class FunctionCallDirective(StrEnum):
    NONE = "none"
    AUTO = "auto"


class FunctionCallByName(TypedDict):
    name: str


class CompleteKwargs(TypedDict, total=False):
    model: ChatModel | None
    functions: Sequence[Function[..., Any]]
    function_call: FunctionCallDirective | FunctionCallByName
    logit_bias: dict[str, float] | None
    max_tokens: int | None
    n_choices: int
    presence_penalty: float | None
    seed: int | None
    stop: str | Sequence[str] | None
    temperature: float | None
    top_p: float | None
    user: str | None
    timeout: int | None


class CallKwargs(TypedDict, total=False):
    """
    A subset of LLMKwargs that can be passed to the call decorator.
    """

    model: ChatModel | None
    logit_bias: dict[str, float] | None
    max_tokens: int | None
    n_choices: int
    presence_penalty: float | None
    seed: int | None
    stop: str | Sequence[str] | None
    temperature: float | None
    top_p: float | None
    user: str | None
    timeout: int | None


class ChatCompletionObject(StrEnum):
    CHAT_COMPLETION = "chat.completion"


@runtime_checkable
class ChatCompletionResponse(Protocol):
    id: str
    object: ChatCompletionObject
    created: datetime
    model: ChatModel
    choices: list[Choice]
    usage: dict[str, Any]


@runtime_checkable
class LLM(Protocol):
    model: ChatModel

    def complete(
        self,
        messages: Sequence[Message],
        context: set[TagPattern] | None = None,
        model: ChatModel | None = None,
        functions: Sequence[Callable[..., Any]] = (),
        function_call: (
            FunctionCallDirective | FunctionCallByName
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

    def call(
        self,
        *,
        prompt: Callable[..., Sequence[Message]],
        context: set[TagPattern] | None = None,
        model: ChatModel | None = None,
        logit_bias: dict[str, float] | None = None,
        max_tokens: int | None = None,
        n_choices: int = 1,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        timeout: int | None = None,
    ) -> Decorator[P, R_co]: ...
