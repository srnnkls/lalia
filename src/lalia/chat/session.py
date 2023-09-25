from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import field
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from rich.console import Console

from lalia.chat import dispatcher
from lalia.chat.completions import Completion, FinishReason
from lalia.chat.messages import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from lalia.chat.messages.buffer import MessageBuffer
from lalia.functions import Error, execute_function_call
from lalia.llm import LLM
from lalia.llm.openai import Choice

console = Console()


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class Session:
    llm: LLM
    system_message: SystemMessage | str = field(
        default_factory=lambda: SystemMessage(content="")
    )
    init_messages: Sequence[Message] = field(default_factory=list)
    functions: Sequence[Callable[..., Any]] | Iterable[Callable[..., Any]] = field(
        default_factory=set
    )
    dispatcher: dispatcher.Dispatcher = field(
        default_factory=dispatcher.FunctionsDispatcher
    )
    autocommit: bool = True
    memory: int = 100
    max_iterations: int = 10
    verbose: bool = False
    debug: bool = False

    def __post_init__(self):
        if isinstance(self.system_message, str):
            self.system_message = SystemMessage(content=self.system_message)

        self.messages = MessageBuffer(
            [
                self.system_message,
                *self.init_messages,
            ],
            verbose=self.verbose,
        )

    def __call__(self, user_input: str = "") -> Message:
        self.messages.add(UserMessage(content=user_input))
        for _ in range(self.max_iterations):
            assistant_message, finish_reason = self.complete()
            if finish_reason is FinishReason.STOP:
                if self.autocommit:
                    self.messages.commit()
                return assistant_message
        return self._complete_failure()

    def _complete_failure(self, message: Message | None = None) -> Message:
        self.messages.add(message)
        choice, *_ = self.llm._complete_failure(self.messages).choices
        failure_message, *_ = self._handle_choice(choice)
        self.messages.rollback()
        self.messages.add(failure_message)
        if self.autocommit:
            self.messages.commit()
        return failure_message

    def _handle_choice(self, choice: Choice) -> list[Message]:
        match choice.message:
            case AssistantMessage(content, FunctionCall(name, arguments)):
                result_message = self._handle_function_call(name, arguments)
                return [choice.message, result_message]
            case AssistantMessage(content, None):
                return [choice.message]
            case AssistantMessage(None, None):
                raise ValueError(
                    "AssistantMessages without `content` must have a `function_call`"
                )
            case _:
                raise ValueError(f"Unsupported message type: {type(choice.message)}")

    def _handle_function_call(self, name: str, arguments: dict[str, Any]) -> Message:
        function_names = [func.__name__ for func in self.functions]
        if name in function_names:
            func = next(func for func in self.functions if func.__name__ == name)
            results = execute_function_call(func, arguments)
            if self.debug:
                console.print(results)

            match results.error, results.result:
                case None, result:
                    return FunctionMessage(name=name, content=str(result))
                case Error(message), None:
                    return SystemMessage(content=f"Error: {message}")
                case _:
                    raise ValueError("Either `error` or `result` must be `None`")

        return SystemMessage(content=f)

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return self.messages._repr_mimebundle_(include, exclude, **kwargs)

    def add(self, message: Message) -> None:
        self.messages.add(message)

    def commit(self) -> None:
        self.messages.commit()

    def rollback(self) -> None:
        self.messages.rollback()

    def clear(self) -> None:
        self.messages.clear()

    def reset(self) -> None:
        if isinstance(self.system_message, str):
            self.system_message = SystemMessage(content=self.system_message)
        self.messages.clear()
        self.messages.messages = [self.system_message, *self.init_messages]

    def revert(self, transaction: int = -1) -> None:
        self.messages.revert(transaction)

    def complete(self, message: Message | None = None) -> Completion:
        return next(iter(self.complete_choices(message, n_choices=1)))

    def complete_choices(
        self, message: Message | None = None, n_choices=1
    ) -> list[Completion]:
        self.messages.add(message)

        llm_complete, messages, params = self.dispatcher.dispatch(self)

        response = llm_complete(
            messages=messages,
            functions=self.functions,
            n_choices=n_choices,
            **params,
        )

        if self.debug:
            console.print(response)

        completions = []
        completion_messages = []

        for choice in response.choices:
            completion_messages.extend(self._handle_choice(choice))
            completions.append(
                Completion(completion_messages[-1], choice.finish_reason)
            )

        self.messages.add_messages(completion_messages)

        return completions
