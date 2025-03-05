from dataclasses import fields
from typing import Generic, TypeVar

from pydantic.dataclasses import dataclass

from lalia.chat.finish_reason import FinishReason
from lalia.chat.messages import (
    AssistantMessage,
    FunctionMessage,
)

T = TypeVar("T")


@dataclass
class Choice(Generic[T]):
    """
    Wrraps an LLM response message.
    """

    index: int
    message: AssistantMessage[T]
    finish_reason: FinishReason

    def __post_init__(self):
        if isinstance(self.message, dict):
            self.message = AssistantMessage(**self.message)


@dataclass
class Completion:
    """
    Wraps a final message (after processing potential function calls).
    """

    message: AssistantMessage | FunctionMessage
    finish_reason: FinishReason

    def __iter__(self):
        yield from (getattr(self, field.name) for field in fields(self))
