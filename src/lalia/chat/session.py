from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import field
from pprint import pprint
from typing import Any

from pydantic.dataclasses import dataclass

from lalia.chat.completions import Completion, FinishReason
from lalia.chat.messages import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from lalia.functions import Error, execute_function_call
from lalia.io.renderers import ConversationRenderer
from lalia.llm import LLM
from lalia.llm.openai import Choice


@dataclass
class Session:
    llm: LLM
    system_message: str
    messages: Sequence[Message] = field(default_factory=list)
    functions: Sequence[Callable[..., Any]] | Iterable[Callable[..., Any]] = field(
        default_factory=set
    )
    memory: int = 100
    max_iterations: int = 10
    debug: bool = False

    def __call__(self, input: str) -> Message:
        messages = list(self.messages)
        messages.append(UserMessage(content=input))
        for _ in range(self.max_iterations):
            assistant_message, finish_reason = self.complete(messages)
            messages = list(self.messages)
            if finish_reason is FinishReason.STOP:
                return assistant_message
        return self._complete_failure(messages)

    def __post_init__(self):
        self.messages = [SystemMessage(content=self.system_message), *self.messages]

    def _complete_failure(self, messages: Sequence[Message]) -> Message:
        choice = next(iter(self.llm._complete_failure(messages).choices))
        return next(iter(self._handle_choice(messages, choice)))

    def _handle_choice(
        self, messages: Sequence[Message], choice: Choice
    ) -> list[Message]:
        messages = list(messages)

        match choice.message:
            case AssistantMessage(content, FunctionCall(name, arguments)):
                if self.debug:
                    pprint(
                        {
                            "finish_reason": choice.finish_reason,
                            "name": name,
                            "arguments": arguments,
                        }
                    )
                result_message = self._handle_function_call(name, arguments)
                messages.extend([choice.message, result_message])
            case AssistantMessage(None, None):
                raise ValueError(
                    "AssistantMessages without `content` must have a `function_call`"
                )
            case AssistantMessage(content, None):
                messages.append(choice.message)
            case _:
                raise ValueError(f"Unsupported message type: {type(choice.message)}")
        return messages

    def _handle_function_call(self, name: str, arguments: dict[str, Any]) -> Message:
        function_names = [func.__name__ for func in self.functions]
        if name in function_names:
            func = next(func for func in self.functions if func.__name__ == name)
            results = execute_function_call(func, arguments)

            match results.error, results.result:
                case None, result:
                    return FunctionMessage(name=name, content=str(result))
                case Error(message), None:
                    return SystemMessage(content=f"Error: {message}")
                case _:
                    raise ValueError("Either `error` or `result` must be `None`")

        return SystemMessage(content=f"Error: Function `{name}` not found")

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return ConversationRenderer(self.messages)._repr_mimebundle_(
            include, exclude, **kwargs
        )

    def complete(self, messages: Sequence[Message] = ()) -> Completion:
        return next(iter(self.complete_choices(messages, n_choices=1)))

    def complete_choices(
        self, messages: Sequence[Message] = (), n_choices=2
    ) -> list[Completion]:
        if not messages:
            messages = list(self.messages)

        response = self.llm.complete(
            messages=messages,
            n_choices=n_choices,
            functions=self.functions,
        )

        completion_messages = []
        completions = []

        for choice in response.choices:
            completion_messages.extend(self._handle_choice(messages, choice))
            completions.append(
                Completion(completion_messages[-1], choice.finish_reason)
            )

        self.messages = completion_messages

        return completions
