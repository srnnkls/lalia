from __future__ import annotations

from collections import deque

from lalia.chat.completions import FinishReason
from lalia.chat.dispatchers import DispatchCall
from lalia.chat.messages import SystemMessage, UserMessage
from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.messages import UserMessage
from lalia.chat.session import Session
from lalia.llm.openai import ChatModel, FunctionCallDirective, OpenAIChat


class SequentialDispatcher:
    """
    Dispatches LLM function calls sequentially based on the sequence of
    functions provided to the Session.

    Calls every function exactly once.
    """

    def __init__(self):
        self._stack = deque()
        self._called = deque()

    def dispatch(self, session: Session) -> DispatchCall:
        user_input = next(
            message.content
            for message in reversed(session.messages)
            if isinstance(message, UserMessage)
        )
        if "Vesper Martini" in user_input:
            print(user_input)
            messages = list(session.messages)
            messages.append(SystemMessage("Answer with a James Bond quote."))
            return DispatchCall(
                callback=session.llm.complete,
                messages=MessageBuffer(messages),
                params={"function_call": FunctionCallDirective.NONE},
                finish_reason=FinishReason.DELEGATE,
            )

        if not self._stack and not self._called:
            self._stack.extend(list(session.functions))

        params = {}

        if self._stack:
            finish_reason = FinishReason.FUNCTION_CALL

            func = self._stack.popleft()
            self._called.append(func)
            params["function_call"] = {"name": func.__name__}
        else:
            finish_reason = FinishReason.DELEGATE
            params["function_call"] = FunctionCallDirective.NONE

        return DispatchCall(
            callback=session.llm.complete,
            messages=session.messages,
            params=params,
            finish_reason=finish_reason,
        )

    def reset(self) -> None:
        self._stack.extendleft(self._called)
        self._called.clear()


def first(first_step: str) -> str:
    """
    Create a dring in three steps.

    Provide the first step of a cocktail recipe.
    """
    return first_step


def second(second_step: str) -> str:
    """
    Create a drink in three steps.

    Provide the second step of a cocktail recipe.
    """
    return second_step


def third(third_step: str) -> str:
    """
    Create a drink in three steps.

    Provide the third step of a cocktail recipe.
    """
    return third_step


def test_sequential_dispatcher():
    bartender = Session(
        llm=OpenAIChat(
            model=ChatModel.GPT_3_5_TURBO_0613,
            api_key="NOT NEEDED",
        ),
        system_message=(
            "You are a bartender. Use the provided functions to answer your questions. "
            "The number of steps should match the number of function calls."
        ),
        functions=[first, second, third],
        autocommit=False,
        verbose=True,
        dispatcher=SequentialDispatcher(),
    )
    bartender.add(UserMessage("I want a Gin Fizz."))

    for func in bartender.functions:
        dispatch_call = bartender.dispatcher.dispatch(bartender)
        assert dispatch_call.params["function_call"]["name"] == func.__name__

    bartender.add(UserMessage("I want a Vesper Martini."))
    dispatch_call = bartender.dispatcher.dispatch(bartender)
    assert dispatch_call.params["function_call"] == FunctionCallDirective.NONE
