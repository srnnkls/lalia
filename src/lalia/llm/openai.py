from collections.abc import Callable, Iterable, Sequence
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
        functions: Iterable[Callable[..., Any]],
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
        functions: Iterable[Callable[..., Any]] | None = None,
        function_call: FunctionCallDirective
        | dict[str, str] = FunctionCallDirective.AUTO,
        n_choices: int = 1,
        temperature: float | None = None,
        model: ChatModel | None = None,
    ) -> ChatCompletionResponse:
        func_schemas = [get_schema(func) for func in functions] if functions else []

        for _ in range(self.max_retries):
            try:
                raw_response = self.complete_raw(
                    messages=to_raw_messages(messages),
                    functions=func_schemas,
                    function_call=function_call,
                    n_choices=n_choices,
                    temperature=temperature,
                    model=model,
                )
                if functions:
                    raw_response = self._parse_function_call_args(
                        raw_response, functions, messages
                    )

                response = ChatCompletionResponse(**raw_response)

            except Exception as e:
                logger.exception(e)
                continue
            else:
                return response

        return self._complete_failure(messages)

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
        if temperature is None:
            temperature = self.temperature
        if model is None:
            model = self.model

        params = {
            "model": model,
            "messages": messages,
            "n": n_choices,
            "temperature": temperature,
        }

        if functions:
            params["functions"] = functions
            params["function_call"] = function_call

        messages = list(messages)

        raw_response = self._client.chat.completions.create(**params).model_dump()

        logger.debug(params)
        logger.debug(raw_response)

        self._responses.append(raw_response)

        return raw_response
