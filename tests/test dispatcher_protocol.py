from __future__ import annotations

from collections import deque

from cobi.utils.auth.secrets import get_openai_token
from lalia.chat.dispatcher import DispatchCall
from lalia.chat.session import Session
from lalia.llm.openai import ChatModel, OpenAIChat


class SequentialDispatcher:
    """
    Dispatches LLM function calls sequentially based on the sequence of
    functions provided to the Session.

    Calls every function exactly once.
    """

    def __init__(self):
        self._stack = deque()

    def dispatch(self, session: Session) -> DispatchCall:
        if not self._stack:
            self._stack.extend(session.functions)
        params = {}

        func = self._stack.popleft().__name__
        params["func"] = func

        return DispatchCall(
            callback=session.llm.complete,
            messages=session.messages,
            params=params,
        )


def first(first_step: str) -> str:
    """
    We create a Gin Fizz. In three steps.

    Provide the first step.
    """
    return first_step


def second(second_step: str) -> str:
    """
    We create a Gin Fizz. In three steps.

    Provide the second step.
    """
    return second_step


def third(third_step: str) -> str:
    """
    We create a Gin Fizz. In three steps.

    Provide the third and last step.
    """
    return third_step


session = Session(
    llm=OpenAIChat(
        model=ChatModel.GPT3_5_TURBO_0613,
        api_key=get_openai_token(),
    ),
    system_message="You are a Bartender. Use the provided functions to answer your questions.",
    functions=[first, second, third],
    autocommit=False,
    verbose=True,
)

session("How to create a Gin Fizz?")
