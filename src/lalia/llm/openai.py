import json
import os
from collections.abc import Callable, Sequence
from dataclasses import InitVar, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from openai import OpenAI
from pydantic import ConfigDict, Field, TypeAdapter, model_validator
from pydantic.dataclasses import dataclass
from pydantic_core import ArgsKwargs

from lalia.chat.completions import Choice
from lalia.chat.messages import Message, SystemMessage, UserMessage
from lalia.chat.messages.tags import TagPattern
from lalia.chat.roles import Role
from lalia.functions import FunctionSchema, get_name, get_schema
from lalia.io.logging import get_logger
from lalia.io.models.openai import ChatCompletionRequestMessage
from lalia.io.parsers import LLMParser, Parser
from lalia.llm.budgeting.token_counter import (
    calculate_tokens,
    truncate_messages,
)
from lalia.llm.models import ChatModel, FunctionCallDirective

FAILURE_QUERY = "What went wrong? Do I need to provide more information?"

COMPLETION_BUFFER = 450

logger = get_logger(__name__)


def _to_openai_raw_message(message: Message | dict[str, Any]) -> dict[str, Any]:
    if isinstance(message, dict):
        return message

    adapter = TypeAdapter(type(message))
    raw_message = {
        field: value
        for field, value in adapter.dump_python(message, exclude_none=True).items()
        if field in ChatCompletionRequestMessage.model_fields
    }
    match raw_message:
        case {
            "role": Role.ASSISTANT,
            "function_call": {"name": name} as f_call,
        }:
            # TODO: Centralize argument dumping
            raw_message["function_call"] = {
                "name": name,
                "arguments": json.dumps(f_call.get("arguments"), indent=2, default=str),
            }

    return raw_message


def _to_openai_raw_messages(
    messages: Sequence[Message | dict[str, Any]],
) -> list[dict[str, Any]]:
    return [_to_openai_raw_message(message) for message in messages]


def _to_open_ai_raw_function_schema(
    func: Callable[..., Any] | FunctionSchema | dict[str, Any],
) -> dict[str, Any]:
    match func:
        case FunctionSchema() as func_schema:
            return func_schema.to_dict()
        case dict() as raw_func_schema:
            return FunctionSchema(**raw_func_schema).to_dict()
        case Callable():
            return get_schema(func).to_dict()
    raise ValueError(f"Cannot convert {func} to OpenAI function schema.")


def _to_open_ai_raw_function_schemas(
    funcs: (
        Sequence[Callable[..., Any]]
        | Sequence[FunctionSchema]
        | Sequence[dict[str, Any]]
    ),
) -> list[dict[str, Any]]:
    return [_to_open_ai_raw_function_schema(func) for func in funcs]


def _truncate_raw_messages(
    messages: Sequence[dict[str, Any]],
    model: ChatModel,
    functions: Sequence[dict[str, Any]],
    completion_buffer: int,
    exlude_roles: frozenset[str] = frozenset({"system"}),
) -> list[dict[str, Any]]:
    excluded_messages = [
        message for message in messages if message["role"] in exlude_roles
    ]

    excluded_messages_tokens = calculate_tokens(
        messages=excluded_messages,
        model=model,
    )

    to_truncate = [
        message for message in messages if message["role"] not in exlude_roles
    ]

    messages_truncated = [
        *excluded_messages,
        *truncate_messages(
            messages=to_truncate,
            token_threshold=model.context_window - excluded_messages_tokens,
            completion_buffer=completion_buffer,
            functions=functions,
        ),
    ]

    logger.info(
        f"Truncated {len(list(messages))} messages with "
        f"{calculate_tokens(messages, functions, model=model)} tokens to "
        f"{len(messages_truncated)} messages with "
        f"{calculate_tokens(messages_truncated, functions, model=model)} tokens."
    )

    return messages_truncated


@dataclass
class Usage:
    prompt: int
    completion: int
    total: int


class ChatCompletionObject(StrEnum):
    CHAT_COMPLETION = "chat.completion"


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
    api_key: InitVar[str | None] = None
    model: ChatModel = ChatModel.GPT_3_5_TURBO_0613
    temperature: float = 1.0
    max_retries: int = 5
    parser: Parser | None = Field(None, exclude=True)
    failure_messages: list[Message] = field(
        default_factory=lambda: [
            UserMessage(FAILURE_QUERY),
        ]
    )
    completion_buffer: int = COMPLETION_BUFFER

    @model_validator(mode="before")
    @classmethod
    def _set_up_parser(cls, data: ArgsKwargs) -> ArgsKwargs:
        kwargs = {} if not data.kwargs else data.kwargs
        if "parser" not in kwargs:
            parser = LLMParser(
                llms=[
                    cls(
                        *data.args,
                        **kwargs,
                        # the parser's LLM has to be parserless to avoid
                        # infinite recursion
                        parser=None,
                    )
                ]
            )
            kwargs["parser"] = parser
            data = ArgsKwargs(data.args, kwargs)
        return data

    def __post_init__(self, api_key: str | None):
        if api_key is None:
            api_key_env = os.getenv("OPENAI_API_KEY")
            if api_key_env is None:
                raise ValueError(
                    "No OpenAI API key provided, `api_key` or env `OPENAI_API_KEY` must"
                    " be set."
                )
        self._api_key = api_key or api_key_env
        self._responses: list[dict[str, Any]] = []
        self._client = OpenAI(api_key=api_key)

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
        context: set[TagPattern] | None = None,
    ) -> dict[str, Any]:
        if self.parser is None:
            return response
        function_call = response["choices"][0]["message"].get("function_call")
        if function_call is not None:
            name = function_call["name"]
            payload = function_call["arguments"]
            func = next(func for func in functions if get_name(func) == name)
            args, parsing_error_messages = self.parser.parse_function_call_args(
                payload=payload,
                function=func,
                messages=messages,
            )
            function_call["function"] = func
            function_call["arguments"] = args
            function_call["context"] = context or set()
            function_call["parsing_error_messages"] = parsing_error_messages
            return response
        else:
            return response

    def complete(
        self,
        messages: Sequence[Message],
        context: set[TagPattern] | None = None,
        model: ChatModel | None = None,
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
    ) -> ChatCompletionResponse:
        if context is None:
            context = set()
        if model is None:
            model = self.model

        func_schemas = _to_open_ai_raw_function_schemas(functions)

        raw_response = self.complete_raw(
            messages=messages,
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

        if functions and self.parser is not None:
            raw_response = self._parse_function_call_args(
                raw_response, functions, messages, context
            )

        response = ChatCompletionResponse(**raw_response)

        return response

    def complete_raw(
        self,
        messages: Sequence[Message | dict[str, Any]],
        model: ChatModel | None = None,
        functions: Sequence[dict[str, Any]] = (),
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
    ) -> dict[str, Any]:
        if temperature is None:
            temperature = self.temperature
        if model is None:
            model = self.model

        raw_messages = _to_openai_raw_messages(messages)

        # TODO: Pass context tags
        messages_truncated = _truncate_raw_messages(
            messages=raw_messages,
            model=model,
            functions=functions,
            completion_buffer=self.completion_buffer,
        )

        params = {
            "messages": messages_truncated,
            "model": model,
            "max_tokens": max_tokens,
            "n": n_choices,
            "seed": seed,
            "stop": stop,
            "temperature": temperature,
            "top_p": top_p,
            "timeout": timeout,
        }

        if logit_bias is not None:
            params["logit_bias"] = logit_bias

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
