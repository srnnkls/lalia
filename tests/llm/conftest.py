import json
import random
import warnings
from collections.abc import Sequence
from inspect import cleandoc
from typing import Any, cast

import hypothesis.strategies as st
import pytest
from hypothesis.errors import (
    HypothesisSideeffectWarning,
    NonInteractiveExampleWarning,
)
from openai.types.chat import ChatCompletion

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=HypothesisSideeffectWarning)
    from hypothesis_jsonschema import from_schema
from dataclasses import dataclass, field

from lalia.chat.finish_reason import FinishReason
from lalia.llm.llm import FunctionCallByName
from lalia.llm.openai import ChatModel, FunctionCallDirective, OpenAIChat


def get_restricted_schema(schema: dict[str, Any]) -> dict[str, Any]:
    restricted_schema = schema.copy()["parameters"]
    restricted_schema["additionalProperties"] = False
    return restricted_schema


def generate_function_args(schema: dict[str, Any]) -> dict[str, Any]:
    schema = get_restricted_schema(schema)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NonInteractiveExampleWarning)
        return cast(dict[str, Any], from_schema(schema).example())


def get_function_args_strategy(schema):
    schema = get_restricted_schema(schema)
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
    functions: Sequence[dict[str, Any]],
    function_call: FunctionCallDirective | FunctionCallByName,
) -> dict[str, Any]:
    if function_call == FunctionCallDirective.AUTO:
        return random.choice(functions)  # noqa
    elif isinstance(function_call, dict):
        return next(func for func in functions if func["name"] == function_call["name"])
    raise ValueError(f"Invalid function call directive: {function_call}")


@dataclass
class FakeOpenAICompletions:
    _hypothesis_func_call_args: st.SearchStrategy | None = None

    def create(
        self,
        messages: Sequence[dict[str, Any]],
        model: ChatModel | None = None,
        functions: Sequence[dict[str, Any]] = (),
        function_call: (
            FunctionCallDirective | FunctionCallByName
        ) = FunctionCallDirective.AUTO,
        logit_bias: dict[str, float] | None = None,
        max_tokens: int | None = None,
        n: int = 1,
        presence_penalty: float | None = None,
        # response_format: ResponseFormat | None = None # NOT SUPPORTED
        seed: int | None = None,
        stop: str | Sequence[str] | None = None,
        # stream: bool = False, # NOT SUPPORTED
        temperature: float | None = None,
        # tools: Sequence[Tool] | None = None, # NOT SUPPORTED
        # tool_choice: ToolChoice | None = None, # NOT SUPPORTED
        top_p: float | None = None,
        user: str | None = None,
        timeout: int | None = None,
    ) -> ChatCompletion:
        if functions and function_call is not FunctionCallDirective.NONE:
            func = get_function_to_call(functions, function_call)
            if self._hypothesis_func_call_args is None:
                args = generate_function_args(func)
            else:
                args_strategy = get_function_args_strategy(func)
                args = self._hypothesis_func_call_args.draw(args_strategy)  # pyright: ignore

            function_call_response = {
                "name": func["name"],
                "arguments": json.dumps(args),
            }
            content_response = None
            finish_reason = FinishReason.FUNCTION_CALL
        else:
            function_call_response = None
            content_response = cleandoc(random.choice(AI_QUOTES))  # noqa
            finish_reason = FinishReason.STOP

        return ChatCompletion(
            **{
                "id": "fake_id",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "system_fingerprint": "fp_3875483275fake",
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "index": 0,
                        "message": {
                            "content": content_response,
                            "function_call": function_call_response,
                            "role": "assistant",
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            }
        )


@dataclass
class FakeOpenAIChat:
    completions: FakeOpenAICompletions = field(default_factory=FakeOpenAICompletions)


@dataclass
class FakeOpenAIClient:
    chat: FakeOpenAIChat = field(default_factory=FakeOpenAIChat)
    hypothesis_function_call_args: st.SearchStrategy | None = None

    def __post_init__(self):
        if self.hypothesis_function_call_args is not None:
            self.chat = FakeOpenAIChat(
                completions=FakeOpenAICompletions(self.hypothesis_function_call_args)
            )


@pytest.fixture()
def fake_llm(fake_openai_client):
    fake_openai = OpenAIChat(api_key="fake_api_key")
    fake_openai._client = fake_openai_client
    return fake_openai


@pytest.fixture()
def fake_openai_client():
    return FakeOpenAIClient()


@pytest.fixture()
def fake_openai_client_type():
    return FakeOpenAIClient


@pytest.fixture()
def ai_quotes():
    return AI_QUOTES


@pytest.fixture()
def get_restricted_json_schema():
    return get_restricted_schema
