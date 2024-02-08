import re
from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Any, overload

from pydantic import Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass
from tiktoken import encoding_name_for_model, get_encoding

from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.messages import Message
from lalia.chat.messages.tags import Tag, TagPattern
from lalia.llm.budgeting.token_counter import (
    calculate_tokens,
    truncate_messages_or_buffer,
)
from lalia.llm.models import ChatModel


class Encoding(StrEnum):
    CL100K_BASE = "cl100k_base"

    @classmethod
    def from_model(cls, model: ChatModel | str) -> str:
        try:
            encoding_name = encoding_name_for_model(model)
            return cls(encoding_name)
        except KeyError:
            raise ValueError(f"Unsupported model: {model}") from KeyError


@dataclass
class Encoder:
    encoding_name: str = Encoding.CL100K_BASE

    def __post_init__(self):
        self.encoder = get_encoding(self.encoding_name)

    @classmethod
    def from_model(cls, model: ChatModel | str) -> "Encoder":
        return cls(Encoding.from_model(model))

    def encode(self, text: str) -> list[int]:
        try:
            return self.encoder.encode(text)
        except Exception as e:
            raise ValueError(f"Encoding failed with error: {e}") from e

    def decode(self, tokens: list[int]) -> str:
        try:
            return self.encoder.decode(tokens)
        except Exception as e:
            raise ValueError(f"Decoding failed with error: {e}") from e


@dataclass
class Budgeter:
    token_threshold: int = Field(default=0, gt=0)
    completion_buffer: int = Field(default=0, gt=0)
    model: ChatModel | str = ChatModel.GPT_3_5_TURBO_0613

    def __post_init__(self):
        self.encoder = Encoder.from_model(self.model)

    @field_validator("completion_buffer")
    @classmethod
    def check_completion_buffer(cls, buffer: Any, info: ValidationInfo) -> None:
        data = info.data
        if "token_threshold" in data and buffer > data["token_threshold"]:
            raise ValueError("Completion buffer cannot exceed token threshold.")
        else:
            return buffer

    def count_tokens(
        self,
        messages: MessageBuffer | Sequence[Message | dict[str, Any]],
        functions: Sequence[Callable[..., Any] | dict[str, Any]] = (),
    ) -> int:
        return calculate_tokens(messages, functions)

    @overload
    def truncate(
        self,
        messages: Sequence[dict[str, Any]],
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
        ) = lambda _: True,
    ) -> list[dict[str, Any]]: ...

    @overload
    def truncate(
        self,
        messages: MessageBuffer | Sequence[Message],
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
        ) = lambda _: True,
    ) -> list[Message]: ...

    def truncate(
        self,
        messages: MessageBuffer | Sequence[Message] | Sequence[dict[str, Any]],
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
        ) = lambda _: True,
    ) -> list[Message] | list[dict[str, Any]]:
        return truncate_messages_or_buffer(
            messages=messages,
            token_threshold=self.token_threshold,
            completion_buffer=self.completion_buffer,
            functions=functions,
            exclude_tags=exclude_tags,
        )
