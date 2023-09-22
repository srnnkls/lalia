from dataclasses import fields
from enum import StrEnum

from pydantic.dataclasses import dataclass

from lalia.chat.messages import (
    AssistantMessage,
    BaseMessage,
    FunctionMessage,
    Message,
    SystemMessage,
)


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    CONTENT_FILTER = "content_filter"
    NULL = "null"


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: FinishReason

    def __post_init__(self):
        if isinstance(self.message, dict):
            self.message = BaseMessage(**self.message).parse()


@dataclass
class Completion:
    message: AssistantMessage | FunctionMessage | SystemMessage
    finish_reason: FinishReason

    def __iter__(self):
        yield from (vars(self)[field.name] for field in fields(self))
