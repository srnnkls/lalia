from __future__ import annotations

import builtins
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from inspect import cleandoc
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

from pydantic import TypeAdapter, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from lalia.chat.completions import Choice
from lalia.chat.messages import FunctionMessage, Message
from lalia.chat.messages.messages import AssistantMessage, FunctionCall
from lalia.chat.messages.tags import Tag
from lalia.llm.llm import FunctionCallByName

if TYPE_CHECKING:
    from lalia.llm import LLM
from lalia.io.logging import get_logger

logger = get_logger(__name__)

yaml = YAML(typ="safe")


VALIDATION_ERROR_DIRECTIVE = cleandoc(
    """
    Error: {error}

    Original payload: {payload!r}

    Are all required parameters provided?
   """
)

DESERIALIZATION_ERROR_DIRECTIVE = cleandoc(
    """
    Error: {error}

    Malformed payload: {payload}
    """
)

DESERIALIZERS = (
    (json.loads, json.JSONDecodeError, {"strict": False}),
    (yaml.load, YAMLError, {}),
)


@runtime_checkable
class Parser(Protocol):
    def parse(
        self,
        payload: str,
        type: builtins.type[T],
        messages: Sequence[Message] = (),
    ) -> tuple[T | None, list[FunctionMessage]]: ...


T = TypeVar("T")


@contextmanager
def disable_parser(llm: LLM) -> Iterator[LLM]:
    if parser := getattr(llm, "parser", None):
        llm.parser = None
        yield llm
        llm.parser = parser
    else:
        yield llm


def _create_error_message(name: str, payload: str, error: Exception) -> FunctionMessage:
    deserializer_errors = tuple(
        deserializer_error for _, deserializer_error, _ in DESERIALIZERS
    )
    common_tags = {
        Tag("error", "function_call"),
        Tag("function", name),
    }
    if isinstance(validation_error := error, ValidationError):
        return FunctionMessage(
            content=VALIDATION_ERROR_DIRECTIVE.format(
                error=validation_error, payload=payload
            ),
            name=name,
            result=None,
            tags={
                Tag("error", "validation"),
                *common_tags,
            },
        )
    if isinstance(deserialization_error := error, (TypeError, *deserializer_errors)):
        return FunctionMessage(
            content=DESERIALIZATION_ERROR_DIRECTIVE.format(
                error=deserialization_error, payload=payload
            ),
            name=name,
            result=None,
            tags={
                Tag("error", "deserialization"),
                *common_tags,
            },
        )
    raise ValueError("Unknown error type.")


class LLMParser:
    deserializers = DESERIALIZERS

    def __init__(
        self,
        llms: Sequence[LLM],
        max_retries: int = 3,
    ):
        self.llms = llms
        self.max_retries = max_retries

    def _complete_invalid_payload(
        self,
        payload: str,
        type: builtins.type[T],
        messages: Sequence[Message],
        llm: LLM,
        exception: Exception,
    ) -> tuple[str, Callable[[dict[str, Any]], dict[str, Any]], FunctionMessage]:
        name = f"{type.__name__}_response"
        error_message = _create_error_message(name, payload, exception)
        logger.debug(error_message)

        def response_wrapper(payload: T):
            """
            Supply a valid JSON payload that corrects the failed input.

            Don't add extra quotes around strings. Don't change the input, just
            correct the input types.
            """
            return payload

        response_wrapper.__annotations__ = {"payload": type}
        response_wrapper.__name__ = name

        def unwrap_response(response: dict[str, Any], /) -> dict[str, Any]:
            return response["payload"]

        with disable_parser(llm):
            response = llm.complete(
                messages=[*messages, error_message],
                functions=[response_wrapper],
                function_call=FunctionCallByName(name=name),
            )

        choice: Choice[str] = next(iter(response.choices))

        arguments, assistant_message = self._handle_choice(choice)

        if arguments is not None:
            return arguments, unwrap_response, error_message

        return self._complete_invalid_payload(
            payload=payload,
            type=type,
            messages=[*messages, error_message, assistant_message],
            llm=llm,
            exception=exception,
        )

    def _deserialize(self, payload: str) -> dict[str, Any]:
        errors = {}

        for deserializer, error, params in self.deserializers:
            try:
                deserialized = deserializer(payload, **params)  # type: ignore
            except (TypeError, error) as e:
                errors[deserializer.__name__] = e
                continue
            else:
                return deserialized

        raise errors["loads"]

    def _handle_choice(self, choice: Choice[T]) -> tuple[T | None, AssistantMessage]:
        match choice.message:
            case AssistantMessage(function_call=FunctionCall(arguments=arguments)):
                return arguments, choice.message
        raise ValueError("No function_call for completion.")

    def _parse_with_retry(
        self,
        payload: str,
        type: builtins.type[T],
        messages: Sequence[Message] = (),
    ) -> tuple[T | None, list[FunctionMessage]]:
        adapter = TypeAdapter[T](type)
        error_messages: list[FunctionMessage] = []
        parsed = None

        def unwrap_response(response: dict[str, Any], /) -> dict[str, Any]:
            return response

        deserializer_errors = tuple(
            serializer_error for _, serializer_error, _ in self.deserializers
        )

        for llm in self.llms:
            for _ in range(self.max_retries):
                try:
                    obj = unwrap_response(self._deserialize(payload))
                    logger.debug(obj)
                    parsed = adapter.validate_python(obj)
                except (ValidationError, TypeError, *deserializer_errors) as e:
                    payload, unwrap_response, error_message = (
                        self._complete_invalid_payload(
                            payload=payload,
                            type=type,
                            messages=[*messages, *error_messages],
                            llm=llm,
                            exception=e,
                        )
                    )
                    error_messages.append(error_message)
                continue

            return parsed, error_messages

        return None, error_messages

    def parse(
        self,
        payload: str,
        type: builtins.type[T],
        messages: Sequence[Message] = (),
    ) -> tuple[T | None, list[FunctionMessage]]:
        return self._parse_with_retry(payload, type, messages)
