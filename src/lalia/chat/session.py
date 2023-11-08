from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import field
from typing import Any

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from rich.console import Console

from lalia.chat import dispatchers
from lalia.chat.completions import Completion
from lalia.chat.finish_reason import FinishReason
from lalia.chat.messages import (
    AssistantMessage,
    FunctionCall,
    FunctionMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from lalia.chat.messages.buffer import MessageBuffer
from lalia.chat.messages.tags import Tag
from lalia.functions import Error, execute_function_call, get_name
from lalia.io.logging import get_logger
from lalia.llm import LLM
from lalia.llm.openai import Choice

console = Console()

logger = get_logger(__name__)


@dataclass(kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class Session:
    llm: LLM
    system_message: SystemMessage | str = field(
        default_factory=lambda: SystemMessage(content="")
    )
    init_messages: Sequence[Message] = field(default_factory=list)
    functions: Sequence[Callable[..., Any]] = ()
    dispatcher: dispatchers.Dispatcher = field(
        default_factory=dispatchers.FunctionsDispatcher
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
        try:
            self.messages.add(UserMessage(content=user_input))
            for _ in range(self.max_iterations):
                assistant_message, finish_reason = self.complete()
                if finish_reason is FinishReason.STOP:
                    return assistant_message
            return self._complete_failure()
        except (Exception, KeyboardInterrupt) as e:
            self._handle_exception(e)
            raise e

    def _complete_failure(self, message: Message | None = None) -> Message:
        self.messages.add(message)
        choice, *_ = self.llm._complete_failure(self.messages).choices
        (failure_message, *_), _ = self._handle_choice(choice)
        self.rollback()
        self.messages.add(failure_message)
        if self.autocommit:
            self.messages.commit()
        return failure_message

    def _complete_choices(
        self, message: Message | None = None, n_choices=1
    ) -> list[Completion]:
        self.messages.add(message)

        (
            llm_complete,
            messages,
            params,
            dispatcher_finish_reason,
        ) = self.dispatcher.dispatch(self)

        response = llm_complete(
            messages=messages,
            functions=self.functions,
            n_choices=n_choices,
            **params,
        )

        logger.debug(response)

        completions = []
        completion_messages = []

        for choice in response.choices:
            messages, finish_reason = self._handle_choice(choice)
            completion_messages.extend(messages)

            if dispatcher_finish_reason is not FinishReason.DELEGATE:
                finish_reason = dispatcher_finish_reason

            if finish_reason is FinishReason.STOP:
                if self.autocommit:
                    self.messages.commit()
                self.dispatcher.reset()

            completions.append(Completion(completion_messages[-1], finish_reason))

        self.messages.add_messages(completion_messages)

        return completions

    def _complete_function_call_error(
        self,
        message: SystemMessage,
    ) -> tuple[list[Message], FinishReason]:
        """
        Completes an errornous function call and isolates the error message.
        """
        self.messages.add(message)
        with self.messages.expand(message.tags):
            response = self.llm.complete(self.messages, self.functions, n_choices=1)
        choice, *_ = response.choices
        return self._handle_choice(choice)

    def _handle_choice(self, choice: Choice) -> tuple[list[Message], FinishReason]:
        logger.debug(choice)
        match choice.message:
            case AssistantMessage(_, FunctionCall(name, arguments)):
                result_message_or_error, finish_reason = self._handle_function_call(
                    name, arguments
                )
                if isinstance(result_message_or_error, Error):
                    error_message = SystemMessage(
                        content=result_message_or_error.message,
                        tags={
                            Tag("error", "function_call"),
                            Tag("function", name),
                        },
                    )
                    return self._complete_function_call_error(error_message)
                if finish_reason is FinishReason.DELEGATE:
                    finish_reason = choice.finish_reason
                return [choice.message, result_message_or_error], finish_reason
            case AssistantMessage(_, None):
                return [choice.message], choice.finish_reason
            case AssistantMessage(None, None):
                raise ValueError(
                    "AssistantMessages without `content` must have a `function_call`"
                )
            case _:
                raise ValueError(f"Unsupported message type: {type(choice.message)}")

    def _handle_exception(self, exception: BaseException):
        try:
            self.rollback()
        except Exception as rollback_exception:
            raise rollback_exception from exception
        finally:
            raise exception

    def _handle_function_call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[Message | Error, FinishReason]:
        function_names = [get_name(func) for func in self.functions]
        if name in function_names:
            func = next(func for func in self.functions if get_name(func) == name)
            result = execute_function_call(func, arguments)
            if self.debug:
                console.print(result)

            match result.value, result.finish_reason, result.error:
                case value, finish_reason, None:
                    return (
                        FunctionMessage(
                            name=name,
                            content=json.dumps(value, indent=2, default=str),
                            result=result,
                            tags={
                                Tag("function", name),
                            },
                        ),
                        finish_reason,
                    )
                case None, finish_reason, error:
                    return error, finish_reason
                case _:
                    raise ValueError("Either `error` or `result` must be `None`")

        return (
            SystemMessage(content=f"Error: Function `{name}` not found."),
            FinishReason.DELEGATE,
        )

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return self.messages._repr_mimebundle_(include, exclude, **kwargs)

    def add(self, message: Message):
        self.messages.add(message)

    def complete(self, message: Message | None = None) -> Completion:
        return next(iter(self.complete_choices(message, n_choices=1)))

    def complete_choices(
        self, message: Message | None = None, n_choices=1
    ) -> list[Completion]:
        try:
            return self._complete_choices(message, n_choices)
        except Exception as e:
            self._handle_exception(e)
            raise e

    def commit(self):
        self.messages.commit()

    def reset(self):
        if isinstance(self.system_message, str):
            self.system_message = SystemMessage(content=self.system_message)
        self.messages.clear()
        self.messages.messages = [self.system_message, *self.init_messages]
        self.dispatcher.reset()

    def revert(self):
        self.messages.revert()

    def rollback(self):
        self.messages.rollback()
        self.dispatcher.reset()
