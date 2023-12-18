import json
from collections import deque
from collections.abc import Callable, Sequence
from enum import IntEnum
from typing import Any, overload

import tiktoken

from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.messages import (
    AssistantMessage,
    BaseMessage,
    FunctionMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from lalia.formatting import format_function_as_typescript
from lalia.functions import FunctionCallResult, Result, get_schema
from lalia.functions.types import FunctionSchema
from lalia.llm.models import ChatModel, FunctionCallDirective


class Overhead(IntEnum):
    MESSAGE_NAME = -1
    MESSAGE_INSTANCE = 3
    SYSTEM_ROLE = -4
    ROLE = 1
    FUNCTION_ROLE = -2
    FUNCTION_CALL = 3
    FUNCTION_NAME = 4
    FUNCTION_DEFINITION = 8
    NONE_FUNCTION_CALL = 1
    COMPLETION = 3


def count_tokens_in_string(
    string: str, model_name: ChatModel = ChatModel.GPT_3_5_TURBO_0613
):
    encoding = tiktoken.encoding_for_model(model_name.value)
    token_count = len(encoding.encode(string))
    return token_count


def get_tokens(
    string: str | None,
    overhead: int = 0,
    model_name: ChatModel = ChatModel.GPT_3_5_TURBO_0613,
):
    return (count_tokens_in_string(string, model_name) + overhead) if string else 0


def estimate_tokens_in_messages(
    messages: MessageBuffer | Sequence[Message | dict[str, Any]],
    model_name: ChatModel = ChatModel.GPT_3_5_TURBO_0613,
) -> int:
    message_tokens = []

    for message in messages:
        match message:
            case (
                SystemMessage() | UserMessage() | AssistantMessage() | FunctionMessage()
            ):
                base_message = message.to_base_message()
            case dict():
                base_message = BaseMessage(**message)
            case _:
                raise ValueError(
                    "Input must be either a MessageBuffer or a a sequence of Messages"
                    "or raw dictionaries"
                )

        # role tokens
        message_tokens.append(Overhead.ROLE)

        # SYSTEM role overhead
        if base_message.role == "system":
            message_tokens.append(Overhead.SYSTEM_ROLE)

        # name tokens
        # if there's a name, the role is omitted, role is always required and always 1
        message_tokens.append(
            get_tokens(
                base_message.name,
                overhead=Overhead.MESSAGE_NAME,
                model_name=model_name,
            )
        )

        # content tokens
        message_tokens.append(get_tokens(base_message.content, model_name=model_name))

        if base_message.function_call:
            # function call name
            message_tokens.append(
                get_tokens(base_message.function_call["name"], model_name=model_name)
            )

            # function call arguments
            message_tokens.append(
                get_tokens(
                    json.dumps(base_message.function_call["arguments"]),
                    model_name=model_name,
                )
            )

            # function call overhead
            message_tokens.append(Overhead.FUNCTION_CALL)

        # function role tokens
        message_tokens.append(
            Overhead.FUNCTION_ROLE if base_message.role.name == "function" else 0
        )

        message_tokens.append(Overhead.MESSAGE_INSTANCE)

    message_tokens.append(Overhead.COMPLETION)

    return sum(message_tokens)


def estimate_tokens_in_functions(
    functions: Sequence[
        Callable[..., Result | FunctionCallResult | str] | dict[str, Any]
    ],
    model_name: ChatModel = ChatModel.GPT_3_5_TURBO_0613,
    include_function_return_types: bool = False,
) -> int:
    function_tokens = []
    for function in functions:
        match function:
            case Callable():
                function_schema = get_schema(function)
            case dict():
                function_schema = FunctionSchema(**function)
            case _:
                raise ValueError("Input must be either a Callable or a dictionary")

        typescript_defintion = format_function_as_typescript(
            function_schema, include_function_return_types
        )
        function_tokens.append(get_tokens(typescript_defintion, model_name=model_name))
    function_tokens.append(Overhead.FUNCTION_DEFINITION)
    return sum(function_tokens)


def estimate_token_count(
    messages: MessageBuffer | Sequence[Message | dict[str, Any]],
    functions: Sequence[
        Callable[..., Result | FunctionCallResult | str] | dict[str, Any]
    ] = (),
    function_call: FunctionCallDirective = FunctionCallDirective.AUTO,
    model: ChatModel = ChatModel.GPT_3_5_TURBO_0613,
    include_function_return_types: bool = False,
) -> int:
    tokens = []

    tokens.append(estimate_tokens_in_messages(messages, model))

    if functions:
        tokens.append(
            estimate_tokens_in_functions(
                functions, model, include_function_return_types
            )
        )

    # if there's a system message _and_ functions are present, subtract four tokens
    if functions:
        for message in messages:
            match message:
                case (
                    SystemMessage()
                    | UserMessage()
                    | AssistantMessage()
                    | FunctionMessage()
                ):
                    base_message = message.to_base_message()
                case dict():
                    base_message = BaseMessage(**message)
                case _:
                    raise ValueError(
                        "Input must be either a MessageBuffer or a a sequence of "
                        "Messages or raw dictionaries"
                    )
            if base_message.role.name == "system":
                tokens.append(Overhead.SYSTEM_ROLE)
                break

    # only add specific function call tokens, if its 'auto' add nothing
    if function_call != FunctionCallDirective.AUTO:
        if function_call == FunctionCallDirective.NONE:
            tokens.append(Overhead.NONE_FUNCTION_CALL)
        elif isinstance(function_call, dict) and "name" in function_call:
            # if it's a specific function call, add function name overhead
            tokens.append(get_tokens(function_call.name, Overhead.FUNCTION_NAME, model))

    return sum(tokens)


@overload
def truncate_messages(
    messages: Sequence[dict[str, Any]],
    token_threshold: int,
    completion_buffer: int,
    functions: Sequence[dict[str, Any]] = (),
) -> list[dict[str, Any]]:
    ...


@overload
def truncate_messages(
    messages: MessageBuffer | Sequence[Message],
    token_threshold: int,
    completion_buffer: int,
    functions: Sequence[Callable[..., Result | FunctionCallResult | str]] = (),
) -> list[Message]:
    ...


def truncate_messages(
    messages: MessageBuffer | Sequence[Message] | Sequence[dict[str, Any]],
    token_threshold: int,
    completion_buffer: int,
    functions: Sequence[Callable[..., Any] | dict[str, Any]] = (),
) -> list[Message] | list[dict[str, Any]]:
    max_tokens_usable = token_threshold - completion_buffer
    current_tokens = estimate_token_count(messages, functions)

    truncated_messages = deque(messages)
    while current_tokens > max_tokens_usable:
        if not truncated_messages:
            raise ValueError(
                "All messages truncated. Remove functions or increase token threshold."
            )
        truncated_messages.popleft()
        current_tokens = estimate_token_count(truncated_messages, functions)

    return list(truncated_messages)  # type: ignore
