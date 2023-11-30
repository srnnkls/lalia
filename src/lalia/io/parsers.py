from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from inspect import cleandoc
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import TypeAdapter, ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from lalia.chat.messages import Message, SystemMessage
from lalia.chat.messages.tags import Tag
from lalia.chat.roles import Role
from lalia.functions import dereference_schema, get_callable, get_schema

if TYPE_CHECKING:
    from lalia.llm import LLM
from lalia.chat.messages import to_raw_messages
from lalia.io.logging import get_logger

logger = get_logger(__name__)

yaml = YAML(typ="safe")

VALIDATION_ERROR_DIRECTIVE = cleandoc(
    """
    Error: {error}

    Invalid payload: {payload}

    Are all required parameters provided?
   """
)

DESERIALIZATION_ERROR_DIRECTIVE = cleandoc(
    """
    Error: {error}

    Malformed input: {payload}
    """
)

DESERIALIZERS = (
    (json.loads, json.JSONDecodeError, {"strict": False}),
    (yaml.load, YAMLError, {}),
)


def _get_func_call_schema(adapter: TypeAdapter) -> dict[str, Any]:
    """
    Wrap a type adapter's json schema in a function call schema.
    """
    schema = adapter.json_schema()

    func_schema = {
        "name": adapter.validator.title,
        "parameters": {"type": "object", "properties": {}},
    }
    func_schema["parameters"] = dereference_schema(schema)

    return func_schema


@runtime_checkable
class Parser(Protocol):
    def parse(
        self,
        payload: str,
        adapter: TypeAdapter,
        messages: Sequence[Message] = (),
    ) -> tuple[dict[str, Any] | None, list[SystemMessage]]:
        ...

    def parse_function_call_args(
        self,
        payload: str,
        function: Callable[..., Any],
        messages: Sequence[Message] = (),
    ) -> tuple[dict[str, Any] | None, list[SystemMessage]]:
        ...


class LLMParser:
    deserializers = DESERIALIZERS

    def __init__(
        self,
        llms: Sequence[LLM],
        max_retries: int = 3,
    ):
        self.llms = llms
        self.max_retries = max_retries

    def _complete_invalid_input(
        self,
        payload: str,
        function_call_schema: dict[str, Any],
        messages: Sequence[dict[str, Any]],
        llm: LLM,
        exception: Exception,
    ) -> tuple[str, SystemMessage]:
        common_tags = {
            Tag("error", "function_call"),
            Tag("function", function_call_schema["name"]),
        }

        match exception:
            case ValidationError():
                error_message = SystemMessage(
                    content=VALIDATION_ERROR_DIRECTIVE.format(
                        error=exception, payload=payload
                    ),
                    tags={
                        Tag("error", "validation"),
                        *common_tags,
                    },
                )
            case json.JSONDecodeError() | YAMLError():
                error_message = SystemMessage(
                    content=DESERIALIZATION_ERROR_DIRECTIVE.format(
                        error=exception, payload=payload
                    ),
                    tags={
                        Tag("error", "deserialization"),
                        *common_tags,
                    },
                )
            case _:
                raise exception

        logger.debug(error_message)

        response = llm.complete_raw(
            messages=[*messages, error_message.to_base_message().to_raw_message()],
            functions=[function_call_schema],
            function_call={"name": function_call_schema["name"]},
        )

        choice = next(iter(response["choices"]))

        arguments, _ = self._handle_choice(choice)

        return arguments, error_message

    def _deserialize(self, payload: str) -> dict[str, Any]:
        errors = {}

        for deserializer, error, params in self.deserializers:
            try:
                deserialized = deserializer(payload, **params)  # type: ignore
            except error as e:
                errors[deserializer.__name__] = e
                continue
            else:
                return deserialized

        raise errors["loads"]

    def _handle_choice(self, choice: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        message = choice["message"]
        if message["role"] == Role.ASSISTANT:
            arguments = message["function_call"]["arguments"]
            return arguments, message
        else:
            raise ValueError("No function_call for completion.")

    def _parse(
        self,
        payload: str,
        adapter: TypeAdapter,
        function_call_schema: dict[str, Any],
        messages: Sequence[Message] = (),
    ) -> tuple[dict[str, Any] | None, list[SystemMessage]]:
        raw_messages = to_raw_messages(messages)
        error_messages: list[SystemMessage] = []
        for llm in self.llms:
            for _ in range(self.max_retries):
                try:
                    obj = self._deserialize(payload)
                    logger.debug(obj)
                    adapter.validate_python(obj)
                except (
                    json.JSONDecodeError,
                    YAMLError,
                    ValidationError,
                ) as e:
                    payload, error_message = self._complete_invalid_input(
                        payload=payload,
                        function_call_schema=function_call_schema,
                        messages=[
                            *raw_messages,
                            *[
                                error_message.to_base_message().to_raw_message()
                                for error_message in error_messages
                            ],
                        ],
                        llm=llm,
                        exception=e,
                    )
                    error_messages.append(error_message)
                    continue

                return obj, error_messages

        return None, error_messages

    def parse(
        self,
        payload: str,
        adapter: TypeAdapter,
        messages: Sequence[Message] = (),
    ) -> tuple[dict[str, Any] | None, list[SystemMessage]]:
        function_call_schema = _get_func_call_schema(adapter)
        return self._parse(payload, adapter, function_call_schema, messages)

    def parse_function_call_args(
        self,
        payload: str,
        function: Callable[..., Any],
        messages: Sequence[Message] = (),
    ) -> tuple[dict[str, Any] | None, list[SystemMessage]]:
        adapter = TypeAdapter(get_callable(function))
        function_call_schema = get_schema(function)
        return self._parse(payload, adapter, function_call_schema, messages)
