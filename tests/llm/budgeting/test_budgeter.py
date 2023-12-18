import pytest

from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.messages import AssistantMessage, SystemMessage, UserMessage
from lalia.llm.budgeting.budgeter import Budgeter, Encoder
from lalia.llm.models import ChatModel


@pytest.fixture
def encoder():
    return Encoder(ChatModel.GPT_3_5_TURBO_0613)


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
