from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import InitVar
from inspect import cleandoc
from pprint import pprint
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ValidationError
from pydantic.dataclasses import dataclass
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from lalia.chat.messages import BaseMessage, Message, SystemMessage
from lalia.chat.roles import Role

if TYPE_CHECKING:
    from lalia.llm import LLM
from lalia.chat.messages import to_raw_messages

yaml = YAML(typ="safe")

VALIDARION_FAILURE_DIRECTIVE = cleandoc(
    """
    Error: {error}
    Invalid payload: {payload}

    Are all required parameters provided?

    DON'T change any parameters that are provided and are not
    fail the validation. If you provide corrected input, just
    restate the respective parameters.
    """
)


def get_model_schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.schema()
    func_schema = {
        "name": model.__name__,
        "parameters": {"type": "object", "properties": {}},
    }

    func_schema["parameters"]["properties"] = schema["properties"]

    func_schema["required"] = schema["required"]

    return func_schema


@runtime_checkable
class Parser(Protocol):
    def parse(
        self, payload: str, model: type[BaseModel], messages: Sequence[Message] = ()
    ) -> tuple[dict[str, Any], list[Message]]:
        ...


class LLMParser:
    def __init__(self, llms: Sequence[LLM], max_retries: int = 10, debug: bool = False):
        self.llms = llms
        self.max_retries = max_retries
        self.debug = debug

    def _complete_invalid_input(
        self,
        payload: str,
        model: type[BaseModel],
        messages: Sequence[dict[str, Any]],
        llm: LLM,
        e: Exception,
    ) -> tuple[str, dict[str, Any]]:
        match e:
            case ValidationError():
                failure_message = SystemMessage(
                    content=VALIDARION_FAILURE_DIRECTIVE.format(
                        error=e, payload=payload
                    )
                )
            case json.JSONDecodeError() | YAMLError():
                failure_message = SystemMessage(
                    content=f"Error: Malformed input: {e}. Please try again."
                )
            case _:
                raise e
        messages = list(messages)
        messages.append(failure_message.to_base_message().to_raw_message())
        schema = get_model_schema(model)

        response = llm.complete_raw(
            messages,
            functions=[schema],
            function_call={"name": model.__name__},
        )
        if self.debug:
            pprint(response)

        choice = next(iter(response["choices"]))
        return self._handle_choice(choice)

    def _deserialize(self, payload: str) -> dict[str, Any]:
        parsers = (
            (json.loads, json.JSONDecodeError, {"strict": False}),
            (yaml.load, YAMLError, {}),
        )
        errors = {}
        for parser, error, params in parsers:
            try:
                deserialized = parser(payload, **params)
            except error as e:
                errors[parser.__name__] = e
                continue
            else:
                return deserialized
        raise errors["loads"]

    def _handle_choice(self, choice: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if choice["message"]["role"] == Role.ASSISTANT:
            message = choice["message"]
            arguments = message["function_call"]["arguments"]
            if self.debug:
                pprint({"arguments": arguments})
            return arguments, message
        else:
            raise ValueError("No function_call for completion.")

    def parse(
        self, payload: str, model: type[BaseModel], messages: Sequence[Message] = ()
    ) -> tuple[dict[str, Any], list[Message]]:
        raw_messages = to_raw_messages(messages)
        for llm in self.llms:
            for _ in range(self.max_retries):
                try:
                    obj = self._deserialize(payload)
                    model.parse_obj(obj)
                except (json.JSONDecodeError, YAMLError, ValidationError) as e:
                    payload, message = self._complete_invalid_input(
                        payload, model, raw_messages, llm, e
                    )
                    if raw_messages:
                        raw_messages[-1] = message
                    else:
                        raw_messages.append(message)
                    continue

                return obj, [
                    BaseMessage(**raw_message).parse() for raw_message in raw_messages  # type: ignore
                ]
        raise ValueError("Unable to parse payload.")
