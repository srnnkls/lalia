import random
import warnings
from collections.abc import Callable, Sequence
from inspect import cleandoc
from typing import Any, cast

import pytest
from hypothesis.errors import NonInteractiveExampleWarning
from hypothesis_jsonschema import from_schema
from pydantic.dataclasses import dataclass

from lalia.chat.finish_reason import FinishReason
from lalia.chat.messages.messages import Message
from lalia.functions import get_schema
from lalia.llm.openai import ChatCompletionResponse, ChatModel, FunctionCallDirective


def get_restricted_schema(func):
    schema = get_schema(func)
    restricted_schema = schema.to_dict()["parameters"]
    restricted_schema["additionalProperties"] = False
    return restricted_schema


def generate_function_args(func: Callable[..., Any]) -> dict[str, Any]:
    schema = get_restricted_schema(func)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        return cast(dict[str, Any], from_schema(schema).example())


def get_function_args_strategy(func):
    schema = get_restricted_schema(func)
    return from_schema(schema)


AI_QUOTES = (
    (
        """
        The question of whether a computer can think is no more interesting than the
        question of whether a submarine can swim.
        Edsger W. Dijkstra
        """
    ),
    (
        """
        Machines take me by surprise with great frequency.
        Alan Turing
        """
    ),
    (
        """
        The science of today is the technology of tomorrow.
        Edward Teller
        """
    ),
    (
        """
        You can have data without information, but you cannot have information without
        data.
        Daniel Keys Moran
        """
    ),
    (
        """
        The best way to predict the future is to invent it.
        Alan Kay
        """
    ),
    (
        """
        AI doesn’t have to be evil to destroy humanity—if AI has a goal, and humanity
        just happens to be in the way, it will destroy humanity as a matter of course
        without even thinking about it, no hard feelings.
        Eliezer Yudkowsky
        """
    ),
    (
        """
        A year spent in artificial intelligence is enough to make one believe in God.
        Alan Perlis
        """
    ),
    (
        """
        We cannot have a society in which, if two individuals wish to communicate, the
        only way that can happen is if it's financed by a third person who wishes to
        manipulate them.
        Jaron Lanier
        """
    ),
    (
        """
        Our intelligence is what makes us human, and AI is an extension of that quality.
        Yann LeCun
        """
    ),
    (
        """
        I've always been against trying to implement serious neuroscience insights in
        neural nets. It's wonderful what you can do without understanding anything.
        Jürgen Schmidhuber
        """
    ),
    (
        """
        Deep learning is a superpower. With it, you can make a computer see, synthesize
        novel art, translate languages, render a medical diagnosis, or build pieces of a
        car that can drive itself. If that isn’t a superpower, I don’t know what is.
        Andrew Ng
        """
    ),
    (
        """
        Every time I fire a linguist, the performance of our speech recognition system
        goes up.
        Frederick Jelinek
        """
    ),
    (
        """
        You can't cram the meaning of a whole %&!$# sentence into a single $&!#/ vector!
        Raymond J. Mooney
        """
    ),
    (
        """
        While as human beings, we know it’s not reasonable to extrapolate linearly
        forever, the machine hasn’t figured that out yet
.       Ian Goodfellow
        """
    ),
)


def get_function_to_call(
    functions: Sequence[Callable[..., Any]],
    function_call: FunctionCallDirective | dict[str, str],
) -> Callable[..., Any]:
    if function_call == FunctionCallDirective.AUTO:
        return random.choice(functions)  # noqa
    elif isinstance(function_call, dict):
        return next(
            func for func in functions if func.__name__ == function_call["name"]
        )
    raise ValueError(f"Invalid function call directive: {function_call}")


@dataclass
class FakeLLM:
    """
    Fake LLM class for testing purposes.
    """

    model: str
    api_key: str
    temperature: float
    max_retries: int

    def complete(
        self,
        messages: Sequence[Message],
        functions: Sequence[Callable[..., Any]] | None = None,
        function_call: FunctionCallDirective
        | dict[str, str] = FunctionCallDirective.AUTO,
        n_choices: int = 1,
        temperature: float | None = None,
        model: ChatModel | None = None,
        *,
        _hypothesis_func_call_args=None,
    ) -> ChatCompletionResponse:
        if functions and function_call is not FunctionCallDirective.NONE:
            func = get_function_to_call(functions, function_call)
            if _hypothesis_func_call_args is None:
                args = generate_function_args(func)
            else:
                args_strategy = get_function_args_strategy(func)
                args = _hypothesis_func_call_args.draw(args_strategy)

            function_call_response = {
                "name": func.__name__,
                "arguments": args,
                "function": func,
            }
            content_response = None
            finish_reason = FinishReason.FUNCTION_CALL
        else:
            function_call_response = None
            content_response = cleandoc(random.choice(AI_QUOTES))  # noqa
            finish_reason = FinishReason.STOP

        response = {
            "id": "fake_id",
            "object": "chat.completion",
            "created": 0,
            "model": ChatModel.GPT_4_0613,
            "system_fingerprint": "fp_3875483275fake",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {
                        "content": content_response,
                        "function_call": function_call_response,
                    },
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
        return ChatCompletionResponse(**response)


@pytest.fixture()
def fake_llm():
    return FakeLLM(
        model="fake_model",
        api_key="fake_api_key",
        temperature=0.5,
        max_retries=5,
    )


@pytest.fixture()
def ai_quotes():
    return AI_QUOTES


@pytest.fixture()
def fake_llm_type():
    return FakeLLM


@pytest.fixture()
def get_restricted_json_schema():
    return get_restricted_schema
