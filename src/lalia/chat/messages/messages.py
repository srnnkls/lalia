from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import field
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic.dataclasses import dataclass
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import BeforeValidator
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from lalia.chat.messages.tags import Tag, TagPattern
from lalia.chat.roles import Role
from lalia.functions import Function, FunctionCallResult
from lalia.io.renderers import MessageRenderer, TagColor

yaml = YAML(typ="safe")


def _parse_tag(tag: dict[str, str] | tuple[str, str, str] | Tag) -> Tag:
    match tag:
        case {"key": key, "value": value}:
            return Tag(key, value)
        case (key, value, color):
            return Tag(key, value, color=TagColor(color))
        case Tag() as tag:
            return tag
    raise TypeError(f"Cannot parse tag from {tag!r}")


def _parse_tags(
    tags: list[dict[str, str]] | set[tuple[str, str, str]] | set[Tag],
) -> set[Tag]:
    return {_parse_tag(tag) for tag in tags}


def _serialize_tags(tags: set[Tag | TagPattern]) -> list[Tag | TagPattern]:
    return list(tags)


Tags = Annotated[
    set[Tag], BeforeValidator(_parse_tags), PlainSerializer(_serialize_tags)
]

TagPatterns = Annotated[set[TagPattern], PlainSerializer(_serialize_tags)]


@dataclass
class SystemMessage:
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    tags: Tags = field(default_factory=set)
    role: Literal[Role.SYSTEM] = Role.SYSTEM

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)


@dataclass
class UserMessage:
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    tags: Tags = field(default_factory=set)
    role: Literal[Role.USER] = Role.USER

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)


@dataclass
class FunctionMessage:
    content: str
    name: str
    result: FunctionCallResult | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    tags: Tags = field(default_factory=set)
    role: Literal[Role.FUNCTION] = Role.FUNCTION

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)


@dataclass
class FunctionCall:
    name: str
    arguments: dict[str, Any] | None = None
    function: Function[..., Any] | None = None
    context: TagPatterns = field(default_factory=set)
    parsing_error_messages: list[FunctionMessage] = field(default_factory=list)

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, arguments: str | dict[str, Any]) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                return json.loads(arguments, strict=False)
            except json.JSONDecodeError as e:
                try:
                    return yaml.load(arguments)
                except YAMLError:
                    raise e from None
        return arguments


@dataclass
class AssistantMessage:
    content: str | None = None
    function_call: FunctionCall | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    tags: Tags = field(default_factory=set)
    role: Literal[Role.ASSISTANT] = Role.ASSISTANT

    @field_validator("function_call", mode="before")
    @classmethod
    def _parse_function_call(cls, function_call: dict[str, Any]) -> FunctionCall:
        if isinstance(function_call, dict):
            return FunctionCall(**function_call)
        return function_call

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)


Message = Annotated[
    SystemMessage | UserMessage | FunctionMessage | AssistantMessage,
    Field(discriminator="role"),
]
