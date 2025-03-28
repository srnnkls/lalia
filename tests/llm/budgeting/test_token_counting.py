import pytest

from lalia.chat.messages.messages import (
    AssistantMessage,
    FunctionCall,
    SystemMessage,
    UserMessage,
)
from lalia.chat.session import Session
from lalia.llm.budgeting.token_counter import (
    count_tokens_in_string,
    calculate_tokens,
    calculate_tokens_in_functions,
    calculate_tokens_in_messages,
    get_tokens,
)
from lalia.llm.openai import ChatModel, OpenAIChat


@pytest.fixture()
def llm(openai_api_key):
    return OpenAIChat(
        model=ChatModel.GPT_4O,
        api_key=openai_api_key,
        temperature=0.0,
    )


@pytest.fixture()
def session(llm):
    return Session(llm=llm, system_message="You are a vet.")


@pytest.fixture()
def session_with_function(llm, foo_function):
    return Session(llm=llm, system_message="You are a vet.", functions=[foo_function])


@pytest.fixture()
def message_buffer():
    return [
        SystemMessage(content="You are a vet."),
        UserMessage(content="Is it wise to stroke a boar?"),
    ]


@pytest.fixture()
def message_buffer_with_function_call(bar_function):
    return [
        SystemMessage(content="You are a vet."),
        UserMessage(content="Is it wise to stroke a boar?"),
        AssistantMessage(
            function_call=FunctionCall(
                name="bar", arguments={"a": 5, "b": "ding"}, function=bar_function
            )
        ),
    ]


class TestUtilityFunctions:
    def test_string_counting(self):
        assert count_tokens_in_string("This is a test.") == 5
        assert count_tokens_in_string("This is a\nmultiline string test.") == 8
        assert (
            count_tokens_in_string("This is a\nmulti paragraph\n\ntest string.") == 10
        )

    def test_token_counting_with_overhead(self):
        assert get_tokens("This is a test.", 5) == 10
        assert get_tokens("This is a test.", -2) == 3


class TestFunctionTokenCounting:
    def test_function_tokens(self, foo_function):
        assert calculate_tokens_in_functions([foo_function]) == 95

    @pytest.mark.openai
    @pytest.mark.exact_tokens
    def test_llm_api_response_to_estimation_exactly(
        self, session_with_function, foo_function
    ):
        session_with_function("Is it wise to stroke a boar?")

        # test the general token estimation function
        assert session_with_function.llm._responses[-1]["usage"][
            "total_tokens"
        ] == calculate_tokens(session_with_function.messages, [foo_function])

        # test seperate message and function token counting
        # to more reliably check for overheads
        assert session_with_function.llm._responses[-1]["usage"][
            "total_tokens"
        ] == calculate_tokens_in_messages(
            session_with_function.messages
        ) + calculate_tokens_in_functions(
            [foo_function]
        )

    @pytest.mark.openai
    def test_llm_api_response_to_estimation_approximately(
        self, session_with_function, foo_function, max_token_deviation
    ):
        session_with_function("Is it wise to stroke a boar?")

        # test the general token estimation function
        assert session_with_function.llm._responses[-1]["usage"][
            "total_tokens"
        ] == pytest.approx(
            calculate_tokens(session_with_function.messages, [foo_function]),
            rel=max_token_deviation,
        )

        # test seperate message and function token counting
        # to more reliably check for overheads
        assert session_with_function.llm._responses[-1]["usage"][
            "total_tokens"
        ] == pytest.approx(
            calculate_tokens_in_messages(session_with_function.messages)
            + calculate_tokens_in_functions([foo_function]),
            rel=max_token_deviation,
        )


class TestMessageTokenCounting:
    def test_message_buffers(self, message_buffer, message_buffer_with_function_call):
        assert calculate_tokens_in_messages(message_buffer) == 21
        assert calculate_tokens_in_messages(message_buffer_with_function_call) == 37

    @pytest.mark.openai
    @pytest.mark.exact_tokens
    def test_llm_api_response_to_estimation_exactly(self, session):
        session("Is it wise to stroke a boar?")
        assert session.llm._responses[-1]["usage"][
            "total_tokens"
        ] == calculate_tokens_in_messages(session.messages)

        session("What do Ants eat?")
        assert session.llm._responses[-1]["usage"][
            "total_tokens"
        ] == calculate_tokens_in_messages(session.messages)

        session("What colors can a cow have?")
        assert session.llm._responses[-1]["usage"][
            "total_tokens"
        ] == calculate_tokens_in_messages(session.messages)

    @pytest.mark.openai
    def test_llm_api_response_to_estimation_approximately(
        self, session, max_token_deviation
    ):
        session("Is it wise to stroke a boar?")
        assert session.llm._responses[-1]["usage"]["total_tokens"] == pytest.approx(
            calculate_tokens_in_messages(session.messages), rel=max_token_deviation
        )

        session("What do Ants eat?")
        assert session.llm._responses[-1]["usage"]["total_tokens"] == pytest.approx(
            calculate_tokens_in_messages(session.messages), rel=max_token_deviation
        )

        session("What colors can a cow have?")
        assert session.llm._responses[-1]["usage"]["total_tokens"] == pytest.approx(
            calculate_tokens_in_messages(session.messages), rel=max_token_deviation
        )


class TestTokenCounting:
    def test_token_counting(
        self, message_buffer, message_buffer_with_function_call, foo_function
    ):
        assert calculate_tokens(message_buffer, [foo_function]) == 116
        assert calculate_tokens(message_buffer_with_function_call) == 37
        assert (
            calculate_tokens(message_buffer_with_function_call, [foo_function]) == 132
        )
