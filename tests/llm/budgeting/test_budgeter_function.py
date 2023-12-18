import pytest

from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.messages import AssistantMessage, SystemMessage, UserMessage
from lalia.llm.budgeting.token_counter import estimate_token_count, truncate_messages


@pytest.fixture()
def message_buffer():
    return MessageBuffer(
        [
            SystemMessage(content="You are a vet."),
            UserMessage(content="Is it wise to stroke a boar?"),
            AssistantMessage(
                content="No, it is not wise to stroke a boar! Why would you do that?",
            ),
        ]
    )


class TestBudgeterFunction:
    def test_budgeter_function(self, message_buffer):
        truncated_message_buffer = truncate_messages(
            messages=message_buffer,
            token_threshold=30,
            completion_buffer=5,
            functions=[],
        )
        assert len(truncated_message_buffer) == 1
        assert truncated_message_buffer[0].content == message_buffer[2].content
        assert type(truncated_message_buffer[0]) == AssistantMessage
        assert estimate_token_count(truncated_message_buffer) == 25

    def test_truncation_tokens_exceeded(self, message_buffer, foo_function):
        with pytest.raises(
            ValueError,
            match=r"All messages truncated. Remove functions or increase token threshold.",
        ) as _:
            _ = truncate_messages(
                messages=message_buffer,
                token_threshold=30,
                completion_buffer=5,
                functions=[foo_function],
            )
        with pytest.raises(
            ValueError,
            match=r"All messages truncated. Remove functions or increase token threshold.",
        ) as _:
            _ = truncate_messages(
                messages=message_buffer,
                token_threshold=0,
                completion_buffer=5,
                functions=[],
            )
