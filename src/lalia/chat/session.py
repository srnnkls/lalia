from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import field
from typing import Any
from uuid import uuid4

from pydantic import UUID4, ConfigDict, Field, field_serializer, field_validator
from pydantic.dataclasses import dataclass

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
from lalia.chat.messages.buffer import DEFAULT_FOLD_TAGS, MessageBuffer
from lalia.chat.messages.tags import Tag, TagPattern
from lalia.functions import (
    Error,
    FunctionCallResult,
    execute_function_call,
    get_name,
)
from lalia.io.logging import get_logger
from lalia.io.serialization.functions import (
    CallableRegistry,
    parse_callables,
    serialize_callables,
)
from lalia.io.storage import DictStorageBackend, StorageBackend
from lalia.llm import LLM
from lalia.llm.openai import Choice

logger = get_logger(__name__)

MAX_FUNCTION_CALL_RETRY_FAILURE_MESSAGE_TEMPLATE = (
    "Error: Calling of function `{name}` failed after"
    " {max_function_call_retries} retries."
)

ARGUMENT_PARSING_FAILURE_MESSAGE_TEMPLATE = (
    "Error: Parsing of function_call arguments for {name} failed."
)

FAILURE_QUERY = "What went wrong? Do I need to provide more information?"


@dataclass(
    kw_only=True,
    config=ConfigDict(arbitrary_types_allowed=True),
)
class Session:
    llm: LLM
    session_id: UUID4 = field(default_factory=uuid4)
    system_message: SystemMessage | str = field(
        default_factory=lambda: SystemMessage(content="")
    )
    init_messages: Sequence[Message] = field(default_factory=list)
<<<<<<< src/lalia/chat/session.py
    messages: MessageBuffer = field(default_factory=MessageBuffer)
=======
    default_fold_tags: set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool] = field(
        default_factory=lambda: DEFAULT_FOLD_TAGS
    )
>>>>>>> src/lalia/chat/session.py
    functions: Sequence[Callable[..., Any]] = ()
    failure_messages: Sequence[Message] = field(
        default_factory=lambda: [UserMessage(content=FAILURE_QUERY)]
    )
    dispatcher: dispatchers.Dispatcher = field(
        default_factory=dispatchers.NopDispatcher
    )
    storage_backend: StorageBackend[UUID4] = Field(
        DictStorageBackend[UUID4](), exclude=True
    )
    autocommit: bool = True
    memory: int = 100
    max_iterations: int = 10
    max_function_call_attempts: int = 5
    rollback_on_error: bool = True
    verbose: bool = False

    @field_serializer("functions")
    def serialize_functions(
        self, functions: Sequence[Callable[..., Any]]
    ) -> list[dict[str, Any]]:
        return serialize_callables(functions)

    @field_validator("functions", mode="before")
    @classmethod
    def parse_functions(
        cls, functions: Sequence[Callable[..., Any]] | Sequence[dict[str, Any]]
    ) -> list[Callable[..., Any]]:
        return parse_callables(functions)

    @field_validator("system_message", mode="before")
    @classmethod
    def parse_system_message(cls, message: str | SystemMessage) -> SystemMessage:
        return SystemMessage(content=message) if isinstance(message, str) else message

    @classmethod
    def from_storage(cls, session_id: UUID4, llm: LLM, **kwargs) -> Session:
        instance = cls(llm=llm)
        instance.load(id_=session_id, **kwargs)
        return instance

    def __post_init__(self):
        if isinstance(self.system_message, str):
            self.system_message = SystemMessage(content=self.system_message)

        for func in self.functions:
            CallableRegistry.register_callable(func)

        self.messages = MessageBuffer(
            [
                self.system_message,
                *self.init_messages,
            ],
            verbose=self.verbose,
            default_fold_tags=self.default_fold_tags,
        )

    def __call__(self, user_input: str = "") -> Message:
        try:
            self.messages.add(UserMessage(content=user_input))
            for _ in range(self.max_iterations):
                completion_message, finish_reason = self.complete()
                match completion_message, finish_reason:
                    case completion_message, FinishReason.STOP:
                        return completion_message
                    case failure_message, FinishReason.ERROR:
                        return self._complete_failure(failure_message).message
            return self._complete_failure().message
        except (Exception, KeyboardInterrupt) as e:
            self._handle_exception(e)
            raise AssertionError("Unreachable") from e

    def _complete_failure(self, message: Message | None = None) -> Completion:
        # TODO: Accept `Error` isstead of `Message`?
        self.messages.add(message)
        self.messages.add_messages(self.failure_messages)

        with self.messages.expand({TagPattern("error", ".*")}) as messages:
            choice, *_ = self.llm.complete(messages).choices

        assistant_message, finish_reson = self._handle_choice(choice)

        if self.autocommit:
            self.messages.commit()

        return Completion(assistant_message, finish_reson)

    def _complete_choices(
        self, message: Message | None = None, n_choices=1
    ) -> list[Completion]:
        self.messages.add(message)

        (
            llm_complete,
            messages,
            context,
            params,
            dispatcher_finish_reason,
        ) = self.dispatcher.dispatch(self)

        with messages.expand(context) as messages:
            response = llm_complete(
                messages=messages,
                functions=self.functions,
                n_choices=n_choices,
                **params,
            )

        logger.debug(response)

        completions = []

        for choice in response.choices:
            completion_message, finish_reason = self._handle_choice(choice)

            if dispatcher_finish_reason is not FinishReason.DELEGATE:
                finish_reason = dispatcher_finish_reason

            if finish_reason is FinishReason.STOP:
                if self.autocommit:
                    self.messages.commit()
                self.dispatcher.reset()

            completions.append(Completion(completion_message, finish_reason))

        return completions

    def _complete_function_call_error(
        self,
        error_message: FunctionMessage,
    ) -> AssistantMessage:
        """
        Completes an erroneous function call and adds a tagged error message to the
        session's messages.
        """

        self.messages.add(error_message)

        with self.messages.expand(error_message.tags) as messages:
            logger.debug(list(messages))
            response = self.llm.complete(
                messages=messages,
                functions=self.functions,
                function_call={"name": error_message.name},
                n_choices=1,
            )

        choice, *_ = response.choices
        function_call_message = choice.message

        return function_call_message

    def _handle_function_call_message(
        self, function_call_message: AssistantMessage
    ) -> tuple[FunctionMessage, FinishReason]:
        for _ in range(self.max_function_call_attempts):
            match function_call_message.function_call:
                case FunctionCall(
                    name=name,
                    arguments=arguments,
                    parsing_error_messages=parsing_error_messages,
                ):
                    function_call_message.tags.add(Tag("function", name))

                    if parsing_error_messages:
                        logger.debug(parsing_error_messages)
                        self.messages.add_messages(parsing_error_messages)

                    self.messages.add(function_call_message)

                    if arguments is None:
                        function_call_message.tags.add(Tag("error", "function_call"))
                        return self._handle_function_call_failure(
                            failure_content=(
                                ARGUMENT_PARSING_FAILURE_MESSAGE_TEMPLATE.format(
                                    name=name
                                )
                            ),
                            name=name,
                        )

                    function_message, finish_reason = self._handle_function_call(
                        name, arguments
                    )

                    if finish_reason is FinishReason.ERROR:
                        function_call_message.tags.add(Tag("error", "function_call"))
                        function_call_message = self._complete_function_call_error(
                            function_message
                        )
                        continue

                    return function_message, finish_reason

                case None:
                    raise ValueError("No function_call supplied")

                case _ as message:
                    raise ValueError(
                        f"Cannot get arguments from message type: {type(message)}"
                    )

        return self._handle_function_call_failure(
            failure_content=(
                MAX_FUNCTION_CALL_RETRY_FAILURE_MESSAGE_TEMPLATE.format(
                    name=name,  # type: ignore
                    max_function_call_retries=self.max_function_call_attempts,
                )
            ),
            name=name,  # type: ignore
        )

    def _handle_choice(
        self, choice: Choice
    ) -> tuple[AssistantMessage | FunctionMessage, FinishReason]:
        logger.debug(choice)
        match choice.message:
            case AssistantMessage(None, None):
                raise ValueError(
                    "AssistantMessages without `content` must have a `function_call`"
                )
            case AssistantMessage(_, FunctionCall()) as function_call_message:
                function_message, finish_reason = self._handle_function_call_message(
                    function_call_message
                )
                if finish_reason is FinishReason.DELEGATE:
                    finish_reason = choice.finish_reason
                if finish_reason is not FinishReason.ERROR:
                    self.messages.add(function_message)
                return function_message, finish_reason
            case AssistantMessage(_, None) as assistant_message:
                self.messages.add(assistant_message)
                return assistant_message, choice.finish_reason
            case _:
                raise ValueError(f"Unsupported message type: {type(choice.message)}")

    def _handle_exception(self, exception: BaseException):
        try:
            if self.rollback_on_error:
                self.rollback()
        except Exception as rollback_exception:
            raise rollback_exception from exception
        finally:
            raise exception

    def _handle_function_call(
        self, name: str, arguments: dict[str, Any]
    ) -> tuple[FunctionMessage, FinishReason]:
        """
        Executes a function call and returns the result as a FunctionMessage.

        Executions run in a retry loop to handle errors.
        """
        function_names = [get_name(func) for func in self.functions]

        if name in function_names:
            func = next(func for func in self.functions if get_name(func) == name)
        else:
            raise ValueError(f"Function `{name}` not found.")

        result = execute_function_call(func, arguments)
        logger.debug(result)

        match result:
            case FunctionCallResult(
                value=value, finish_reason=finish_reason, error=None
            ):
                if isinstance(value, str):
                    value_content = value
                else:
                    value_content = json.dumps(value, indent=2, default=str)
                return (
                    FunctionMessage(
                        name=name,
                        content=value_content,
                        result=result,
                        tags={
                            Tag("function", name),
                        },
                    ),
                    finish_reason,
                )
            case FunctionCallResult(value=None, error=Error() as error):
                return self._handle_function_call_failure(
                    failure_content=(f"Error: {error.message}"),
                    name=name,
                )

            case _:
                raise ValueError("Either `error` or `value` must be `None`")

    def _handle_function_call_failure(
        self, failure_content: str, name: str
    ) -> tuple[FunctionMessage, FinishReason]:
        return (
            FunctionMessage(
                name=name,
                content=failure_content,
                result=None,
                tags={
                    Tag("function", name),
                    Tag("error", "function_call"),
                },
            ),
            FinishReason.ERROR,
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
            raise AssertionError("Unreachable") from e

    def commit(self):
        self.messages.commit()

    def load(self, session_id: UUID4, **kwargs):
        arguments = self.storage_backend.load(session_id)
        # currently, llms are not serialized
        if "llm" not in kwargs:
            arguments["llm"] = self.llm
        arguments.update(kwargs)
        validated_arguments = vars(type(self)(**arguments))
        vars(self).update(validated_arguments)

    def reset(self):
        self.messages.clear()
        self.messages.add(self.system_message)  # type: ignore
        self.messages.add_messages(self.init_messages)

        self.dispatcher.reset()

    def revert(self):
        self.messages.revert()

    def rollback(self):
        self.messages.rollback()

        self.dispatcher.reset()

    def save(self):
        self.storage_backend.save(self, self.session_id)
