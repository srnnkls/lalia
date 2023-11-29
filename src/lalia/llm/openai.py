from collections.abc import Callable, Sequence
from dataclasses import InitVar, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from openai import OpenAI
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from lalia.chat.completions import Choice
from lalia.chat.messages import Message, SystemMessage, UserMessage, to_raw_messages
from lalia.functions import get_name, get_schema
from lalia.io.logging import get_logger
from lalia.io.parsers import LLMParser, Parser

FAILURE_QUERY = "What went wrong? Do I need to provide more information?"

logger = get_logger(__name__)


class FunctionCallDirective(StrEnum):
    NONE = "none"
    AUTO = "auto"


class ChatCompletionObject(StrEnum):
    CHAT_COMPLETION = "chat.completion"


class ChatModel(StrEnum):
    GPT_3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_4_0613 = "gpt-4-0613"


@dataclass
class ChatCompletionResponse:
    id: str
    object: ChatCompletionObject
    created: datetime
    model: ChatModel
    choices: list[Choice]
    usage: dict[str, int]

    def __post_init__(self):
        if isinstance(self.created, int):
            self.created = datetime.fromtimestamp(self.created, UTC)


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class OpenAIChat:
    model: ChatModel
    api_key: InitVar[str]
    temperature: float = 1.0
    max_retries: int = 5
    parser: InitVar[Parser | None] = None

    failure_messages: list[Message] = field(
        default_factory=lambda: [
            UserMessage(FAILURE_QUERY),
        ]
    )

    def __post_init__(self, api_key: str, parser: Parser | None):
        self._api_key = api_key
        self._responses: list[dict[str, Any]] = []
        self._client = OpenAI(api_key=api_key)

        if parser is None:
            self._parser = LLMParser(
                llms=[self],
            )
        else:
            self._parser = parser

    def _complete_failure(self, messages: Sequence[Message]) -> ChatCompletionResponse:
        messages = list(messages)
        messages.extend(self.failure_messages)
        return self.complete(messages)

    def _complete_invalid_input(
        self, messages: Sequence[Message], e: Exception
    ) -> ChatCompletionResponse:
        messages = list(messages)
        messages.append(
            SystemMessage(
                content=(
                    f"Error: Invalid input: {e}. "
                    "Please try again with valid json as input."
                )
            )
        )
        return self.complete(messages)

    def _parse_function_call_args(
        self,
        response: dict[str, Any],
        functions: Sequence[Callable[..., Any]],
        messages: Sequence[Message] = (),
    ) -> dict[str, Any]:
        if (
            function_call := response["choices"][0]["message"].get("function_call")
        ) is not None:
            name = function_call["name"]
            payload = function_call["arguments"]
            func = next(func for func in functions if get_name(func) == name)
            args, parsing_error_messages = self._parser.parse_function_call_args(
                payload=payload,
                function=func,
                messages=messages,
            )
            function_call["arguments"] = args
            function_call["parsing_error_messages"] = parsing_error_messages
            return response
        else:
            return response

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
        func_schemas = [get_schema(func) for func in functions] if functions else []

        raw_response = self.complete_raw(
            messages=to_raw_messages(messages),
            model=model,
            functions=func_schemas,
            function_call=function_call,
            logit_bias=logit_bias,
            max_tokens=max_tokens,
            n_choices=n_choices,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            user=user,
            timeout=timeout,
        )
        if functions:
            raw_response = self._parse_function_call_args(
                raw_response, functions, messages
            )

        response = ChatCompletionResponse(**raw_response)

        return response

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
        if temperature is None:
            temperature = self.temperature
        if model is None:
            model = self.model

        params = {
            "messages": messages,
            "model": model,
            "logit_bias": logit_bias,
            "max_tokens": max_tokens,
            "n": n_choices,
            "seed": seed,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "timeout": timeout,
        }

        if functions:
            params["functions"] = functions
            params["function_call"] = function_call

        if presence_penalty is not None:
            params["presence_penalty"] = presence_penalty

        if user is not None:
            params["user"] = user

        raw_response = self._client.chat.completions.create(**params).model_dump()

        logger.debug(params)
        logger.debug(raw_response)

        self._responses.append(raw_response)

        return raw_response
