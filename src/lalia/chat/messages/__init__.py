from lalia.chat.messages.messages import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from lalia.chat.messages.tags import DefaultTagKeys, Tag
from lalia.io.renderers import TagColor

Tag.register_key_color(DefaultTagKeys.SYSTEM, TagColor.RED)
Tag.register_key_color(DefaultTagKeys.FUNCTION, TagColor.MAGENTA)
Tag.register_key_color(DefaultTagKeys.ERROR, TagColor.BRIGHT_RED)

__all__ = [
    "AssistantMessage",
    "FunctionCall",
    "FunctionMessage",
    "Message",
    "SystemMessage",
    "UserMessage",
]
