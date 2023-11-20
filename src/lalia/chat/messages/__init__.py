from lalia.chat.messages.messages import (
    AssistantMessage,
    BaseMessage,
    FunctionCall,
    FunctionMessage,
    Message,
    SystemMessage,
    UserMessage,
    to_raw_messages,
)
from lalia.chat.messages.tags import DefaultTagKeys, Tag
from lalia.io.renderers import TagColor

Tag.register_key_color(DefaultTagKeys.ERROR, TagColor.RED)
Tag.register_key_color(DefaultTagKeys.FUNCTION, TagColor.MAGENTA)

__all__ = [
    "AssistantMessage",
    "BaseMessage",
    "FunctionCall",
    "FunctionMessage",
    "Message",
    "SystemMessage",
    "UserMessage",
    "to_raw_messages",
]
