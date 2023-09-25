import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import InitVar, asdict, field
from datetime import UTC, datetime
from enum import StrEnum
from pprint import pprint
from typing import Any

import openai
from pydantic import ValidationError
from pydantic.dataclasses import dataclass

from lalia.chat.completions import Choice
from lalia.chat.messages import BaseMessage, Message, SystemMessage, UserMessage
from lalia.functions import get_schema

FAILURE_QUERY = "What went wrong? Do I need to provide more information?"


def _to_raw_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in asdict(message.to_base_message()).items()
            if value is not None or key == "content"
        }
        for message in messages
    ]


class FunctionCallDirective(StrEnum):
    NONE = "none"
    AUTO = "auto"


class ChatCompletionObject(StrEnum):
    CHAT_COMPLETION = "chat.completion"


class ChatModel(StrEnum):
    GPT3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT4_0613 = "gpt-4-0613"


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


@dataclass(kw_only=True)
class OpenAIChat:
    model: ChatModel
    api_key: InitVar[str]
    temperature: float = 1.0
    max_retries: int = 5
    debug: bool = False

    failure_messages: list[Message] = field(
        default_factory=lambda: [
            UserMessage(FAILURE_QUERY),
        ]
    )

    def __post_init__(self, api_key: str):
        self._api_key = api_key
        self._responses: list[dict[str, Any]] = []

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
                    f"Error: Invalid input: {e} Please try again with valid json as input."
                )
            )
        )
        return self.complete(messages)

    def complete(
        self,
        messages: Sequence[Message],
        functions: Sequence[Callable[..., Any]]
        | Iterable[Callable[..., Any]]
        | None = None,
        function_call: FunctionCallDirective | str = FunctionCallDirective.NONE,
        n_choices: int = 1,
        temperature: float | None = None,
        model: ChatModel | None = None,
    ) -> ChatCompletionResponse:
        if temperature is None:
            temperature = self.temperature
        if model is None:
            model = self.model

        func_schemas = [get_schema(func) for func in functions] if functions else []

        params = {
            "model": model,
            "messages": _to_raw_messages(messages),
            "api_key": self._api_key,
            "n": n_choices,
            "temperature": temperature,
        }

        if func_schemas:
            params["functions"] = func_schemas
            if function_call is FunctionCallDirective.NONE:
                params["function_call"] = FunctionCallDirective.AUTO

        messages = list(messages)
        for _ in range(self.max_retries):
            raw_response = openai.ChatCompletion.create(**params)
            if self.debug:
                pprint(raw_response)
            self._responses.append(raw_response)  # type: ignore
            try:
                response = ChatCompletionResponse(**raw_response)  # type: ignore
            except (ValidationError, json.JSONDecodeError) as e:
                self._complete_invalid_input(messages, raw_response, e)  # type: ignore
                continue
            else:
                return response

        return self._complete_failure(messages)
