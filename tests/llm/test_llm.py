from inspect import cleandoc

import pytest
from hypothesis import given
from hypothesis import strategies as st

from lalia.chat.completions import Choice
from lalia.chat.messages.messages import AssistantMessage, UserMessage
from lalia.llm.openai import ChatCompletionResponse, FunctionCallDirective


@pytest.fixture()
def functions():
    def drive_crazy(to_drive_crazy: str) -> str:
        return f"{to_drive_crazy.capitalize()} drives me crazy!"

    return [drive_crazy]


def test_llm_complete_no_function(fake_llm, ai_quotes):
    completion = fake_llm.complete([])
    assert isinstance(completion, ChatCompletionResponse)
    message = completion.choices[0].message
    assert isinstance(message.content, str)
    assert message.content in (cleandoc(quote) for quote in ai_quotes)


def test_llm_complete_function(fake_llm, functions):
    completion = fake_llm.complete(
        [],
        functions=functions,
    )
    assert isinstance(completion, ChatCompletionResponse)
    message = completion.choices[0].message
    assert message.content is None
    assert message.function_call is not None
    assert message.function_call.name == "drive_crazy"
    assert message.function_call.arguments is not None
    assert "to_drive_crazy" in message.function_call.arguments


def test_complete(fake_llm_type, functions):
    fake_llm_strategy = st.from_type(fake_llm_type)
    content_strategy = st.text(min_size=10)
    messages_strategy = st.lists(
        st.builds(UserMessage, content=content_strategy), min_size=1
    )
    functions_strategy = st.just(functions)
    function_call_strategy = st.sampled_from(FunctionCallDirective)
    n_choices_strategy = st.just(1)
    temperature_strategy = st.none()
    model_strategy = st.none()
    func_call_args_strategy = st.data()  # dynamically drawn based on functions

    @given(
        fake_llm=fake_llm_strategy,
        messages=messages_strategy,
        functions=functions_strategy,
        function_call=function_call_strategy,
        n_choices=n_choices_strategy,
        temperature=temperature_strategy,
        model=model_strategy,
        func_call_args=func_call_args_strategy,
    )
    def test_complete_with_strategies(
        fake_llm,
        messages,
        functions,
        function_call,
        n_choices,
        temperature,
        model,
        func_call_args,
    ):
        response = fake_llm.complete(
            messages,
            functions,
            function_call,
            n_choices,
            temperature,
            model,
            _hypothesis_func_call_args=func_call_args,
        )

        assert response is not None
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) == n_choices
        assert all(isinstance(choice, Choice) for choice in response.choices)
        assert all(
            isinstance(choice.message, AssistantMessage) for choice in response.choices
        )
        assert all(
            isinstance(choice.message.content, str)
            for choice in response.choices
            if choice.message.content is not None
        )
        # stopping here as testing the fake class isn't too relevant

    test_complete_with_strategies()
