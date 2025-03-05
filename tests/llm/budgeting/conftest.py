import pytest

from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.messages import AssistantMessage, SystemMessage, UserMessage
from lalia.chat.messages.tags import Tag
from lalia.llm.budgeting.budgeter import Encoder

MAX_TOKEN_DEVIATION = 0.1  # relative tolerance


@pytest.fixture(scope="session")
def max_token_deviation():
    return MAX_TOKEN_DEVIATION


@pytest.fixture
def string_fixture():
    """The name is intentional, because something like 'test_string' would
    be interpreted as a test function by pytest..."""
    return "hello world"


@pytest.fixture
def encoder():
    return Encoder()


@pytest.fixture()
def message_buffer():
    return MessageBuffer(
        [
            SystemMessage(content="You are a vet.", tags={Tag("kind", "initial")}),
            UserMessage(
                content="Is it wise to stroke a boar?",
                tags={Tag("kind", "user input"), Tag("intent", "fearful")},
            ),
            AssistantMessage(
                content="No, it is not wise to stroke a boar! Why would you do that?",
                tags={Tag("kind", "response"), Tag("intent", "blaming")},
            ),
        ]
    )


@pytest.fixture()
def tag_collection():
    return {
        "user_input": {Tag(key="kind", value="user input")},
        "response": {Tag(key="kind", value="response")},
        "intent_fearful": {Tag(key="intent", value="fearful")},
        "user_fearful": {
            Tag(key="kind", value="user input"),
            Tag(key="intent", value="fearful"),
        },
    }


@pytest.fixture()
def raw_message_buffer():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
