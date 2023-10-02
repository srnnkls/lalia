import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import InitVar, field
from datetime import UTC, datetime
from enum import StrEnum
from pprint import pprint
from typing import Any

import openai
from pydantic import ConfigDict, TypeAdapter, ValidationError, validate_call
from pydantic.dataclasses import dataclass

from lalia.chat.completions import Choice
from lalia.chat.messages import Message, SystemMessage, UserMessage, to_raw_messages
from lalia.functions import get_schema
from lalia.io.parsers import LLMParser, Parser

FAILURE_QUERY = "What went wrong? Do I need to provide more information?"


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
    debug: bool = False
    parser: InitVar[Parser | None] = None

    failure_messages: list[Message] = field(
        default_factory=lambda: [
            UserMessage(FAILURE_QUERY),
        ]
    )

    def __post_init__(self, api_key: str, parser: Parser | None):
        self._api_key = api_key
        self._responses: list[dict[str, Any]] = []

        if parser is None:
            self._parser = LLMParser(
                llms=[self],
                debug=self.debug,
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
        self, response: dict[str, Any], functions: Iterable[Callable[..., Any]]
    ) -> dict[str, Any]:
        if "function_call" in response["choices"][0]["message"]:
            name = response["choices"][0]["message"]["function_call"]["name"]
            arguments = response["choices"][0]["message"]["function_call"]["arguments"]
            func = next(iter(func for func in functions if func.__name__ == name))
            adapter = TypeAdapter(validate_call(func))
            args, _ = self._parser.parse(arguments, adapter)
            response["choices"][0]["message"]["function_call"]["arguments"] = args
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
            raw_response = self.complete_raw(
                messages=to_raw_messages(messages),
                functions=func_schemas,
                function_call=function_call,
                n_choices=n_choices,
                temperature=temperature,
                model=model,
            )
            if functions:
                raw_response = self._parse_function_call_args(raw_response, functions)
            try:
                response = ChatCompletionResponse(**raw_response)
            except (ValidationError, json.JSONDecodeError) as e:
                self._complete_invalid_input(messages, e)
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
            "api_key": self._api_key,
            "n": n_choices,
            "temperature": temperature,
        }

        if functions:
            params["functions"] = functions
            params["function_call"] = function_call

        messages = list(messages)

        raw_response = openai.ChatCompletion.create(**params).to_dict()  # type: ignore

        if self.debug:
            pprint(raw_response)
        self._responses.append(raw_response)

        return raw_response
