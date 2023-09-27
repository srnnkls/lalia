from dataclasses import fields
from enum import StrEnum

from pydantic.dataclasses import dataclass

from lalia.chat.messages import (
    AssistantMessage,
    FunctionMessage,
    SystemMessage,
)


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    CONTENT_FILTER = "content_filter"
    DELEGATE = "delegate"
    NULL = "null"


@dataclass
class Choice:
    """
    Wrraps an LLM response message.
    """

    index: int
    message: AssistantMessage
    finish_reason: FinishReason

    def __post_init__(self):
        if isinstance(self.message, dict):
            self.message = AssistantMessage(**self.message)


@dataclass
class Completion:
    """
    Wraps a final message (after processing potential function calls).
    """

    message: AssistantMessage | FunctionMessage | SystemMessage
    finish_reason: FinishReason

    def __iter__(self):
        yield from (vars(self)[field.name] for field in fields(self))
