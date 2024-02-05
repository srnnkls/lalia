import pytest

from lalia.chat.messages.messages import AssistantMessage
from lalia.chat.messages.tags import Tag
from lalia.llm.budgeting.token_counter import estimate_tokens, truncate_messages


class TestBudgeterFunction:
    def test_budgeter_function(self, message_buffer):
        truncated_message_buffer = truncate_messages(
            messages=message_buffer,
            token_threshold=40,
            completion_buffer=5,
        )
        assert len(truncated_message_buffer) == 1
        assert truncated_message_buffer[0].content == message_buffer[2].content
        assert type(truncated_message_buffer[0]) == AssistantMessage
        assert estimate_tokens(truncated_message_buffer) == 25

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

    def test_truncation_with_filter(self, message_buffer):
        truncated_message_buffer = truncate_messages(
            messages=message_buffer,
            token_threshold=20,
            completion_buffer=5,
            exclude_tags=Tag(key="kind", value="initial"),
        )

        assert len(truncated_message_buffer) == 1
        assert all(msg.role == "system" for msg in truncated_message_buffer)
        assert estimate_tokens(truncated_message_buffer) <= 30 - 5

    def test_truncation_without_filter(self, message_buffer):
        # no filtering! so no messages are truncated!
        truncated_message_buffer = truncate_messages(
            messages=message_buffer, token_threshold=4000, completion_buffer=5
        )

        assert list(message_buffer) == truncated_message_buffer
        assert len(truncated_message_buffer) == len(message_buffer)
        assert estimate_tokens(truncated_message_buffer) == 43
