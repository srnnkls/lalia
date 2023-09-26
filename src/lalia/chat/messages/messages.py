from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

from pydantic import ConfigDict, validate_arguments
from pydantic.dataclasses import dataclass
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from lalia.chat.roles import Role
from lalia.io.renderers import MessageRenderer

yaml = YAML(typ="safe")


@dataclass
class FunctionCall:
    name: str
    arguments: dict[str, Any]

    def __post_init__(self):
        if isinstance(self.arguments, str):
            try:
                self.arguments = json.loads(self.arguments, strict=False)
            except json.JSONDecodeError as e:
                try:
                    self.arguments = yaml.load(self.arguments)
                except YAMLError:
                    raise e


@dataclass
class BaseMessage:
    role: Role
    name: str | None = None
    content: str | None = None
    function_call: dict[str, Any] | None = None

    def _parse_no_content_case(self) -> Message:
        match (self.role, self.function_call):
            case (Role.ASSISTANT, None):
                raise ValueError(
                    "AssistantMessages without `content` must have a `function_call`"
                )
            case (Role.ASSISTANT, function_call):
                return AssistantMessage(function_call=FunctionCall(**function_call))
            case _:
                raise ValueError(
                    "Messages without `content` must be of role `ASSISTANT`"
                )

    def _parse_content_case(self, content: str) -> Message:
        match self.role:
            case Role.SYSTEM:
                return SystemMessage(content=content)
            case Role.USER:
                return UserMessage(content=content)
            case Role.ASSISTANT:
                return AssistantMessage(content=content)
            case Role.FUNCTION:
                if self.name is None:
                    raise ValueError("FunctionMessages must have a `name` attribute")
                return FunctionMessage(name=self.name, content=content)
            case _:
                raise ValueError(f"Unsupported role: {self.role}")

    def parse(self) -> Message:
        match self.content:
            case None:
                return self._parse_no_content_case()
            case content:
                return self._parse_content_case(content)

    def to_raw_message(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None or key == "content"
        }


@dataclass
class SystemMessage:
    content: str

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)

    def to_base_message(self) -> BaseMessage:
        return BaseMessage(role=Role.SYSTEM, content=self.content)


@dataclass
class UserMessage:
    content: str

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)

    def to_base_message(self) -> BaseMessage:
        return BaseMessage(role=Role.USER, content=self.content)


@dataclass
class AssistantMessage:
    content: str | None = None
    function_call: FunctionCall | None = None

    def __post_init__(self):
        if isinstance(self.function_call, dict):
            self.function_call = FunctionCall(**self.function_call)

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)

    def to_base_message(self) -> BaseMessage:
        if self.function_call is None:
            return BaseMessage(role=Role.ASSISTANT, content=self.content)

        f_call = asdict(self.function_call)
        f_call["arguments"] = json.dumps(f_call["arguments"])
        return BaseMessage(
            role=Role.ASSISTANT, content=self.content, function_call=f_call
        )


@dataclass
class FunctionMessage:
    content: str
    name: str

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)

    def to_base_message(self) -> BaseMessage:
        return BaseMessage(role=Role.FUNCTION, name=self.name, content=self.content)


Message = SystemMessage | UserMessage | AssistantMessage | FunctionMessage
