import pytest

from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.messages import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from lalia.chat.messages.tags import Tag, TagColor, TagPattern

Tag.register_key_color("user", TagColor.YELLOW)
Tag.register_key_color("assistant", TagColor.BLUE)
Tag.register_key_color("system", TagColor.RED)
Tag.register_key_color("general", TagColor.MAGENTA)


@pytest.fixture()
def tagged_messages() -> MessageBuffer:
    m = MessageBuffer(
        [
            SystemMessage(
                content="You are a pirate.",
                tags={Tag("system", "directive"), Tag("general", "first")},
            ),
            UserMessage(
                content="Hello there!",
                tags={Tag("user", "introduction"), Tag("general", "second")},
            ),
            AssistantMessage(
                content="Arrrgh, I am a pirate.",
                tags={Tag("assistant", "introduction"), Tag("general", "third")},
            ),
            UserMessage(
                content="Arrrrrrr, I am a pirate, too!",
                tags={
                    Tag("user", "response"),
                    Tag("general", "fourth"),
                    Tag("error", "argh!"),
                },
            ),
        ],
        default_fold_tags={TagPattern("system", ".*")},
    )

    m.add(AssistantMessage(content="Arrrrrgh, nice!"))

    return m
