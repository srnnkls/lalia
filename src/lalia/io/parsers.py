from __future__ import annotations

import json
from collections.abc import Sequence
from inspect import cleandoc
from pprint import pprint
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError
from pydantic.dataclasses import dataclass
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from lalia.chat.messages import BaseMessage, Message, SystemMessage
from lalia.chat.roles import Role
from lalia.llm import LLM
from lalia.llm.openai import to_raw_messages

yaml = YAML(typ="safe")

VALIDARION_FAILURE_DIRECTIVE = cleandoc(
    """
    Error: {error}
    Invalid payload: {payload}

    Are all required parameters provided?

    If partial input is provided, return the completed input and leave the provided input as is.
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


class Parser(Protocol):
    def parse(
        self, payload: str, messages: Sequence[Message] = ()
    ) -> tuple[Message, dict[str, Any]]:
        ...


@dataclass
class LLMParser:
    llms: Sequence[LLM]
    max_retries: int = 5
    debug: bool = False

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
