import os
import subprocess

import pytest

from lalia.chat.messages import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.tags import Tag, TagPattern
from lalia.io.renderers import TagColor

OPENAI_API_KEY_REF = "ahssdo26ixj2gloto2b4z34k7u/OpenAI API Key/OPENAI_API_TOKEN"


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
                tags={Tag("user", "response"), Tag("general", "fourth")},
            ),
        ],
        default_fold_tags={TagPattern("system", ".*")},
    )

    m.add(AssistantMessage(content="Arrrrrgh, nice!"))

    return m


def get_op_secret(ref: str) -> str:
    result = subprocess.run(
        ["/usr/bin/env", "op", "read", f"op://{ref}"],
        capture_output=True,
        text=True,
        check=True,
    )

    if result.stderr:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


@pytest.fixture()
def openai_api_key() -> str:
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]
    return get_op_secret(OPENAI_API_KEY_REF)
