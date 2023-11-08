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
from lalia.chat.messages.tags import Tag, TagPattern
from lalia.io.renderers import TagColor

Tag.register_key_color("error", TagColor.RED)
Tag.register_key_color("function", TagColor.GREEN)
