from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, field
from datetime import UTC, datetime
from typing import Any

from openai.types.chat.chat_completion_message import FunctionCall as OpenAIFunctionCall
from pydantic import field_validator
from pydantic.dataclasses import dataclass
from pydantic.functional_serializers import model_serializer
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from lalia.chat.messages.tags import Tag, TagPattern
from lalia.chat.roles import Role
from lalia.functions import FunctionCallResult
from lalia.io.models.openai import ChatCompletionRequestMessage
from lalia.io.renderers import MessageRenderer

yaml = YAML(typ="safe")


def _parse_tags(
    tags: list[dict[str, str]] | set[tuple[str, str]] | set[Tag]
) -> set[Tag]:
    match tags:
        case set(tags):
            return {Tag(*tag) for tag in tags if isinstance(tag, tuple)} | {
                tag for tag in tags if isinstance(tag, Tag)
            }
        case list(tags_raw):
            return {Tag.from_dict(tag) for tag in tags_raw}
        case _:
            return set()


def to_raw_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [message.to_base_message().to_raw_message() for message in messages]


@dataclass
class BaseMessage:
    role: Role
    name: str | None = None
    content: str | None = None
    function_call: dict[str, Any] | None = None
    tags: list[dict[str, str]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def _parse_no_content_case(self) -> Message:
        match (self.role, self.function_call):
            case (Role.ASSISTANT, None):
                raise ValueError(
                    "AssistantMessages without `content` must have a `function_call`"
                )
            case (Role.ASSISTANT, function_call):
                return AssistantMessage(function_call=FunctionCall(**function_call))  # type: ignore
            case _:
                raise ValueError(
                    "Messages without `content` must be of role `assistant`"
                )

    def _parse_content_case(self, content: str, tags: list[dict[str, str]]) -> Message:
        match self.role:
            case Role.SYSTEM:
                return SystemMessage(content=content, tags={Tag(**tag) for tag in tags})
            case Role.USER:
                return UserMessage(content=content, tags={Tag(**tag) for tag in tags})
            case Role.ASSISTANT:
                return AssistantMessage(
                    content=content, tags={Tag(**tag) for tag in tags}
                )
            case Role.FUNCTION:
                if self.name is None:
                    raise ValueError("FunctionMessages must have a `name` attribute")
                return FunctionMessage(
                    name=self.name,
                    content=content,
                    result=FunctionCallResult(name=self.name, arguments={}),
                    tags={Tag(**tag) for tag in tags},
                )
            case _:
                raise ValueError(f"Unsupported role: {self.role}")

    def parse(self) -> Message:
        match self.content:
            case None:
                return self._parse_no_content_case()
            case content:
                return self._parse_content_case(content, self.tags)

    def to_raw_message(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if key in set(ChatCompletionRequestMessage.model_fields)
            and (value is not None or key == "content")
        }


@dataclass
class SystemMessage:
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    tags: set[Tag] = field(default_factory=set)

    @model_serializer
    def serialize_message(self) -> dict[str, Any]:
        return asdict(self.to_base_message())

    @field_validator("tags", mode="before")
    @classmethod
    def _parse_tags(
        cls, tags: list[dict[str, str]] | set[tuple[str, str]] | set[Tag]
    ) -> set[Tag]:
        return _parse_tags(tags)

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)

    def to_base_message(self) -> BaseMessage:
        return BaseMessage(
            role=Role.SYSTEM,
            content=self.content,
            timestamp=self.timestamp,
            tags=[{"key": tag.key, "value": tag.value} for tag in self.tags],
        )


@dataclass
class UserMessage:
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    tags: set[Tag] = field(default_factory=set)

    @model_serializer
    def serialize_message(self) -> dict[str, Any]:
        return asdict(self.to_base_message())

    @field_validator("tags", mode="before")
    @classmethod
    def _parse_tags(
        cls, tags: list[dict[str, str]] | set[tuple[str, str]] | set[Tag]
    ) -> set[Tag]:
        return _parse_tags(tags)

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)

    def to_base_message(self) -> BaseMessage:
        return BaseMessage(
            role=Role.USER,
            content=self.content,
            timestamp=self.timestamp,
            tags=[{"key": tag.key, "value": tag.value} for tag in self.tags],
        )


@dataclass
class FunctionCall:
    name: str
    arguments: dict[str, Any] | None
    function: Callable[..., Any] | None = None
    context: set[TagPattern] = field(default_factory=set)
    parsing_error_messages: list[SystemMessage] = field(default_factory=list)

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
                    raise e
        return arguments


@dataclass
class AssistantMessage:
    content: str | None = None
    function_call: FunctionCall | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    tags: set[Tag] = field(default_factory=set)

    @model_serializer
    def serialize_message(self) -> dict[str, Any]:
        return asdict(self.to_base_message())

    @field_validator("function_call", mode="before")
    @classmethod
    def _parse_function_call(cls, function_call: dict[str, Any]) -> FunctionCall:
        if isinstance(function_call, dict):
            return FunctionCall(**function_call)
        return function_call

    @field_validator("tags", mode="before")
    @classmethod
    def _parse_tags(
        cls, tags: list[dict[str, str]] | set[tuple[str, str]] | set[Tag]
    ) -> set[Tag]:
        return _parse_tags(tags)

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)

    def to_base_message(self) -> BaseMessage:
        if self.function_call is None:
            return BaseMessage(
                role=Role.ASSISTANT, content=self.content, timestamp=self.timestamp
            )

        f_call = {
            "name": self.function_call.name,
            "arguments": self.function_call.arguments or {},
        }

        f_call["arguments"] = json.dumps(f_call["arguments"])

        return BaseMessage(
            role=Role.ASSISTANT,
            content=self.content,
            function_call=f_call,
            tags=[{"key": tag.key, "value": tag.value} for tag in self.tags],
            timestamp=self.timestamp,
        )


@dataclass
class FunctionMessage:
    content: str
    name: str
    result: FunctionCallResult | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    tags: set[Tag] = field(default_factory=set)

    @model_serializer
    def serialize_message(self) -> dict[str, Any]:
        return self.to_base_message().to_raw_message()

    @field_validator("tags", mode="before")
    @classmethod
    def _parse_tags(cls, tags: list[dict[str, str]] | set[Tag]) -> set[Tag]:
        return _parse_tags(tags)

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return MessageRenderer(self)._repr_mimebundle_(include, exclude, **kwargs)

    def to_base_message(self) -> BaseMessage:
        return BaseMessage(
            role=Role.FUNCTION,
            name=self.name,
            content=self.content,
            tags=[{"key": tag.key, "value": tag.value} for tag in self.tags],
            timestamp=self.timestamp,
        )


Message = SystemMessage | UserMessage | AssistantMessage | FunctionMessage
