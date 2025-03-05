import json
import re
from collections.abc import Callable, Iterator, Sequence
from enum import IntEnum
from typing import Any, overload

import tiktoken
from pydantic import TypeAdapter

from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.folds import derive_tag_predicate
from lalia.chat.messages.messages import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from lalia.chat.messages.tags import Tag, TagPattern
from lalia.formatting import OpenAIFunctionFormatter
from lalia.functions import FunctionSchema, get_schema
from lalia.llm.llm import FunctionCallDirective
from lalia.llm.models import ChatModel


class Overhead(IntEnum):
    MESSAGE_NAME = -1
    MESSAGE_INSTANCE = 4
    SYSTEM_ROLE = -4
    FUNCTION_ROLE = -2
    FUNCTION_CALL = 3
    FUNCTION_NAME = 4
    FUNCTION_DEFINITION = 8
    NONE_FUNCTION_CALL = 1
    COMPLETION = 3


def _calculate_tokens_for_function_arguments(
    arguments: dict[str, Any] | None, model: ChatModel = ChatModel.GPT_4O
) -> int:
    serialized_arguments = json.dumps(arguments, default=str)
    return get_tokens(serialized_arguments, model=model)


def _calculate_tokens_in_message(
    message: Message, model: ChatModel = ChatModel.GPT_4O
) -> int:
    message_tokens = []
    match message:
        case SystemMessage():
            message_tokens.append(Overhead.SYSTEM_ROLE)
        case FunctionMessage(name=name):
            message_tokens.append(
                get_tokens(name, overhead=Overhead.MESSAGE_NAME, model=model)
            )
        case AssistantMessage(
            function_call=FunctionCall(name=name, arguments=arguments)
        ):
            message_tokens.append(get_tokens(name, model=model))
            message_tokens.append(
                _calculate_tokens_for_function_arguments(arguments, model=model)
            )
            message_tokens.append(Overhead.FUNCTION_CALL)
        case FunctionMessage():
            message_tokens.append(Overhead.FUNCTION_ROLE)

    message_tokens.append(
        get_tokens(message.content, overhead=Overhead.MESSAGE_INSTANCE, model=model)
    )

    return sum(message_tokens)


def _iterate_tokens_in_messages(
    messages: MessageBuffer | Sequence[Message | dict[str, Any]],
    model: ChatModel = ChatModel.GPT_4O,
) -> Iterator[int]:
    adapter = TypeAdapter(Message)

    for message in messages:
        match message:
            case dict() as raw_message:
                parsed_message = adapter.validate_python(raw_message)
                yield _calculate_tokens_in_message(parsed_message, model)
            case (
                SystemMessage()
                | UserMessage()
                | AssistantMessage()
                | FunctionMessage()
            ):
                yield _calculate_tokens_in_message(message, model)
            case _:
                raise ValueError(
                    "Input must be either a MessageBuffer or a a sequence of "
                    "Messages or raw messages"
                )


def count_tokens_in_string(
    string: str, model: ChatModel = ChatModel.GPT_4O
):
    encoding = tiktoken.encoding_for_model(model.value)
    token_count = len(encoding.encode(string))
    return token_count


def get_tokens(
    string: str | None,
    overhead: int = 0,
    model: ChatModel = ChatModel.GPT_4O,
):
    return (count_tokens_in_string(string, model) + overhead) if string else 0


def calculate_tokens_in_messages(
    messages: MessageBuffer | Sequence[Message | dict[str, Any]],
    model: ChatModel = ChatModel.GPT_4O,
) -> int:
    if not messages:
        return 0
    return sum(_iterate_tokens_in_messages(messages, model)) + Overhead.COMPLETION


def calculate_tokens_in_functions(
    functions: Sequence[Callable[..., Any] | dict[str, Any]],
    model: ChatModel = ChatModel.GPT_4O,
) -> int:
    if not functions:
        return 0
    function_schemas = []
    for function in functions:
        match function:
            case Callable():
                function_schema = get_schema(function)
            case dict():
                function_schema = FunctionSchema(**function)
            case _:
                raise ValueError("Input must be either a Callable or a dictionary")
        function_schemas.append(function_schema)

    formatter = OpenAIFunctionFormatter()
    functions_formatted = formatter.format(function_schemas)
    return get_tokens(functions_formatted, model=model)


def calculate_tokens(
    messages: MessageBuffer | Sequence[Message | dict[str, Any]],
    functions: Sequence[Callable[..., Any] | dict[str, Any]] = (),
    function_call: FunctionCallDirective = FunctionCallDirective.AUTO,
    model: ChatModel = ChatModel.GPT_4O,
) -> int:
    tokens = []

    tokens.append(calculate_tokens_in_messages(messages, model))

    if functions:
        tokens.append(calculate_tokens_in_functions(functions, model))

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
    exclude_tags: (
        Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | tuple[str | re.Pattern, str | re.Pattern]
        | dict[str | re.Pattern, str | re.Pattern]
        | set[tuple[str | re.Pattern, str | re.Pattern]]
        | set[dict[str | re.Pattern, str | re.Pattern]]
        | Callable[[set[Tag]], bool]
    ) = lambda _: False,
) -> list[dict[str, Any]]: ...  # pragma: no cover


@overload
def truncate_messages(
    messages: MessageBuffer | Sequence[Message],
    token_threshold: int,
    completion_buffer: int,
    functions: Sequence[Callable[..., Any]] = (),
    exclude_tags: (
        Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | tuple[str | re.Pattern, str | re.Pattern]
        | dict[str | re.Pattern, str | re.Pattern]
        | set[tuple[str | re.Pattern, str | re.Pattern]]
        | set[dict[str | re.Pattern, str | re.Pattern]]
        | Callable[[set[Tag]], bool]
    ) = lambda _: False,
) -> list[Message]: ...  # pragma: no cover


def truncate_messages(
    messages: MessageBuffer | Sequence[Message] | Sequence[dict[str, Any]],
    token_threshold: int,
    completion_buffer: int,
    functions: Sequence[Callable[..., Any]] | Sequence[dict[str, Any]] = (),
    exclude_tags: (
        Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | tuple[str | re.Pattern, str | re.Pattern]
        | dict[str | re.Pattern, str | re.Pattern]
        | set[tuple[str | re.Pattern, str | re.Pattern]]
        | set[dict[str | re.Pattern, str | re.Pattern]]
        | Callable[[set[Tag]], bool]
    ) = lambda _: False,
) -> list[Message] | list[dict[str, Any]]:
    return truncate_messages_or_buffer(
        messages=messages,
        token_threshold=token_threshold,
        completion_buffer=completion_buffer,
        functions=functions,
        exclude_tags=exclude_tags,
    )


def _get_message_tags(message: Message | dict[str, Any]) -> set[Tag]:
    match message:
        case SystemMessage() | UserMessage() | AssistantMessage() | FunctionMessage():
            return message.tags
        case {"tags": tags}:
            return {Tag(**tag) for tag in tags}
    return set()


def truncate_messages_or_buffer(
    messages: MessageBuffer | Sequence[Message] | Sequence[dict[str, Any]],
    token_threshold: int,
    completion_buffer: int,
    functions: Sequence[Callable[..., Any] | dict[str, Any]] = (),
    exclude_tags: (
        Tag
        | TagPattern
        | set[Tag]
        | set[TagPattern]
        | tuple[str | re.Pattern, str | re.Pattern]
        | dict[str | re.Pattern, str | re.Pattern]
        | set[tuple[str | re.Pattern, str | re.Pattern]]
        | set[dict[str | re.Pattern, str | re.Pattern]]
        | Callable[[set[Tag]], bool]
    ) = lambda _: False,
    model: ChatModel = ChatModel.GPT_4O,
) -> list[Message] | list[dict[str, Any]]:
    adapter = TypeAdapter(Message)

    # inner function to calculate token count for a single message
    def _get_token_count(message: Message | dict[str, Any]):
        match message:
            case dict() as raw_message:
                parsed_message = adapter.validate_python(raw_message)
                return _calculate_tokens_in_message(parsed_message, model)
            case _:
                return _calculate_tokens_in_message(message, model)

    # calculate maximum tokens allowed for messages
    max_usable_tokens = token_threshold - completion_buffer

    # prepare predicate for message exclusion
    exclude_predicate = derive_tag_predicate(exclude_tags)

    # step 1: track indices of messages to exclude or to retain
    excluded_indices = set()
    eligible_indices = []
    for i, message in enumerate(messages):
        if exclude_predicate(_get_message_tags(message)):
            excluded_indices.add(i)
        else:
            eligible_indices.append(i)

    # step 2: calculate base tokens used by excluded messages, functions and completion buffer
    base_tokens = completion_buffer
    if excluded_indices:
        excluded_messages = [messages[i] for i in excluded_indices]
        base_tokens += calculate_tokens_in_messages(excluded_messages)

    if functions:
        base_tokens += calculate_tokens_in_functions(functions)

    # failsafe: if base tokens exceed the threshold, abort truncation
    if base_tokens > max_usable_tokens:
        raise ValueError(
            "Base tokens exceed the available tokens. Aborting truncation."
        )

    # step 3: truncate eligible messages based on token count
    truncated_indices = []
    tokens = base_tokens
    for i in reversed(eligible_indices):
        message_tokens = _get_token_count(messages[i])

        # if the message still fits in the budget, add it to the truncated indices
        if tokens + message_tokens <= max_usable_tokens:
            truncated_indices.append(i)
            tokens += message_tokens
        else:
            break  # Stop adding messages once the token limit is exceeded

    # step 4: recunstruct message buffer
    indices_to_keep = set(truncated_indices) | excluded_indices

    return [message for i, message in enumerate(messages) if i in indices_to_keep]  # type: ignore
