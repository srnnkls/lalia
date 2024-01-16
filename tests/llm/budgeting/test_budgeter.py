import pytest

from lalia.chat.messages.tags import Tag
from lalia.llm.budgeting.budgeter import Budgeter, Encoder
from lalia.llm.budgeting.token_counter import estimate_tokens
from lalia.llm.models import ChatModel


class TestBudgeterInstantiation:
    def test_budgeter_instantiation_default_model(self):
        budgeter = Budgeter(token_threshold=30, completion_buffer=5)
        assert budgeter.model == ChatModel.GPT_3_5_TURBO_0613
        assert isinstance(budgeter.encoder, Encoder)

    def test_budgeter_instantiation_with_model(self):
        budgeter = Budgeter(
            token_threshold=30, completion_buffer=5, model=ChatModel.GPT_3_5_TURBO_0613
        )
        assert budgeter.model == ChatModel.GPT_3_5_TURBO_0613
        assert isinstance(budgeter.encoder, Encoder)

    def test_budgeter_invalid_token_threshold(self):
        with pytest.raises(ValueError):
            Budgeter(token_threshold=-10, completion_buffer=5)

    def test_budgeter_invalid_completion_buffer(self):
        with pytest.raises(ValueError):
            Budgeter(token_threshold=30, completion_buffer=-5)

    def test_budgeter_completion_buffer_exceeds_threshold(self):
        with pytest.raises(ValueError):
            Budgeter(token_threshold=30, completion_buffer=35)


class TestBudgeterClass:
    def test_budgeter_truncation_with_filter(self, message_buffer):
        budgeter = Budgeter(token_threshold=20, completion_buffer=5)

        truncated_messages = budgeter.truncate(
            messages=message_buffer,
            exclude_tags=Tag(key="kind", value="initial"),
        )

        assert len(truncated_messages) == 1
        assert all(msg.to_base_message().role == "system" for msg in truncated_messages)
        assert estimate_tokens(truncated_messages) <= 40 - 5
