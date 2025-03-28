from __future__ import annotations

import functools
import json
from collections.abc import Callable, Sequence
from dataclasses import field
from typing import Any, Unpack
from uuid import uuid4

from pydantic import (
    UUID4,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    validate_call,
)
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
    Function,
    FunctionCallResult,
    execute_function_call,
    get_name,
)
from lalia.io.logging import get_logger
from lalia.io.progress import NopProgressHandler, ProgressManager
from lalia.io.progress.session import (
    ExecutingProgress,
    GeneratingProgress,
    SessionProgressState,
)
from lalia.io.serialization.functions import (
    CallableRegistry,
    serialize_callable,
    serialize_callables,
)
from lalia.io.storage import DictStorageBackend, StorageBackend
from lalia.llm.llm import LLM, CallKwargs, CompleteKwargs, P, R_co, Wrapped
from lalia.llm.openai import Choice, Usage

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
    system_message: SystemMessage | None = None
    init_messages: Sequence[Message] = field(default_factory=list)
    default_fold_tags: set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool] = field(
        default_factory=lambda: DEFAULT_FOLD_TAGS
    )
    messages: MessageBuffer = field(default_factory=MessageBuffer)
    functions: Sequence[Function[..., Any]] = ()
    failure_messages: Sequence[Message] = field(
        default_factory=lambda: [UserMessage(content=FAILURE_QUERY)]
    )
    dispatcher: dispatchers.Dispatcher = field(
        default_factory=dispatchers.NopDispatcher
    )
    storage_backend: StorageBackend = Field(default=DictStorageBackend(), exclude=True)
    progress_manager: ProgressManager = field(
        default_factory=lambda: ProgressManager(
            initial_state=SessionProgressState.IDLE,
            handler=NopProgressHandler(),
        )
    )
    autocommit: bool = True
    memory: int = 100
    max_iterations: int = 10
    max_function_call_attempts: int = 5
    rollback_on_error: bool = True
    verbose: bool = False

    @field_validator("system_message", mode="before")
    @classmethod
    def validate_system_message(
        cls, system_message: SystemMessage | str | None
    ) -> SystemMessage | None:
        if isinstance(system_message, str):
            return SystemMessage(content=system_message)
        return system_message

    @field_serializer("functions")
    def serialize_functions(
        self, functions: Sequence[Callable[..., Any]]
    ) -> list[dict[str, Any]]:
        return serialize_callables(functions)

    @field_serializer("default_fold_tags")
    def serialize_default_fold_tags(
        self, tags: set[Tag] | set[TagPattern] | Callable[[set[Tag]], bool]
    ) -> list[Tag | TagPattern] | dict[str, Any]:
        if not callable(tags):
            return list(tags)
        if callable(callable_ := tags):
            return serialize_callable(callable_)
        raise AssertionError("Unreachable")

    @classmethod
    def from_storage(
        cls,
        session_id: UUID4,
        storage_backend: StorageBackend,
        llm: LLM,
        **kwargs,
    ) -> Session:
        instance = cls(llm=llm, storage_backend=storage_backend)
        instance.load(session_id=session_id, **kwargs)
        return instance

    def __post_init__(self):
        if isinstance(self.system_message, str):
            self.system_message = SystemMessage(content=self.system_message)

        for func in self.functions:
            CallableRegistry.register_callable(func)

        if not self.messages:
            self.messages = MessageBuffer(
                verbose=self.verbose,
                default_fold_tags=self.default_fold_tags,
            )

        if self.system_message is not None:
            self.messages.add(self.system_message)

        if self.init_messages:
            self.messages.add_messages(self.init_messages)

    @validate_call
    def __call__(
        self, user_input: str | UserMessage | None = None
    ) -> AssistantMessage | FunctionMessage:
        if isinstance(user_input, str):
            message = UserMessage(content=user_input)
        else:
            message = user_input
        completion = self.complete_flow(message)
        return completion.message

    def _complete_choices(
        self,
        message: Message | None = None,
        context: set[TagPattern] | None = None,
        **llm_kwargs: Unpack[CompleteKwargs],
    ) -> list[Completion]:
        self.messages.add(message)

        call = self.dispatcher.dispatch(self)

        kwargs: CompleteKwargs = {**call.kwargs, **llm_kwargs}

        if "functions" not in kwargs:
            kwargs["functions"] = self.functions

        if context:
            context = context | call.context
        else:
            context = call.context

        with call.messages.expand(context) as messages:
            progress = GeneratingProgress(
                function=kwargs.get("function_call", {}).get("name")
            )
            self.progress_manager.emit(progress)
            response = call.callback(
                messages=messages,
                context=context,
                **kwargs,
            )

        logger.debug(response)

        completions = []

        for choice in response.choices:
            completion_message, finish_reason = self._handle_choice(choice)

            if call.finish_reason is not FinishReason.DELEGATE:
                finish_reason = call.finish_reason

            completions.append(Completion(completion_message, finish_reason))

            if finish_reason is FinishReason.STOP:
                if self.autocommit:
                    self.messages.commit()
                self.dispatcher.reset()

        return completions

    def _complete_failure(self, message: Message | None = None) -> Completion:
        # TODO: Accept `Error` isstead of `Message`?
        self.messages.add(message)
        self.messages.add_messages(self.failure_messages)

        with self.messages.expand({TagPattern("error", ".*")}) as messages:
            choice, *_ = self.llm.complete(messages).choices

        assistant_message, finish_reason = self._handle_choice(choice)

        if self.autocommit:
            self.messages.commit()

        return Completion(assistant_message, finish_reason)

    def _complete_function_call_error(
        self,
        error_message: FunctionMessage,
        context: set[TagPattern],
        function: Callable[..., Any],
    ) -> AssistantMessage:
        """
        Completes an erroneous function call and adds a tagged error message to the
        session's messages.
        """

        self.messages.add(error_message)

        error_context = {TagPattern.from_tag_like(tag) for tag in error_message.tags}
        with self.messages.expand(context | error_context) as messages:
            logger.debug(list(messages))
            response = self.llm.complete(
                messages=messages,
                context=context,
                functions=[function],
                function_call={"name": get_name(function)},
                n_choices=1,
            )

        choice, *_ = response.choices
        function_call_message = choice.message

        return function_call_message

    def _handle_function_call_message(
        self, function_call_message: AssistantMessage
    ) -> tuple[FunctionMessage, FinishReason]:
        for i in range(1, self.max_function_call_attempts + 1):
            match function_call_message.function_call:
                # TODO: Refactor and appropriately match based on strucutre and
                # class patterns
                case FunctionCall(
                    name=name,
                    function=function,
                    arguments=arguments,
                    parsing_error_messages=parsing_error_messages,
                ):
                    progress = ExecutingProgress(
                        iteration=i,
                        function=name,
                        arguments=arguments,
                    )
                    self.progress_manager.emit(progress)

                    function_call_message.tags.add(Tag("function", name))

                    if parsing_error_messages:
                        logger.debug(parsing_error_messages)
                        self.messages.add_messages(parsing_error_messages)

                    self.messages.add(function_call_message)

                    if arguments is None or function is None:
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
                        name, function, arguments
                    )

                    if finish_reason is FinishReason.FUNCTION_CALL_FAILURE:
                        return function_message, finish_reason

                    if finish_reason is FinishReason.FUNCTION_CALL_ERROR:
                        function_call_message.tags.add(Tag("error", "function_call"))
                        function_call_message = self._complete_function_call_error(
                            error_message=function_message,
                            context=function_call_message.function_call.context,
                            function=function,
                        )
                        continue

                    return function_message, finish_reason

                case None:
                    raise ValueError("No function_call supplied")

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
        self, name: str, function: Callable[..., Any], arguments: dict[str, Any]
    ) -> tuple[FunctionMessage, FinishReason]:
        """
        Executes a function call and returns the result as a FunctionMessage.

        Executions run in a retry loop to handle errors.
        """

        result = execute_function_call(function, arguments)
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
            case FunctionCallResult(
                value=None,
                error=Error() as error,
                finish_reason=finish_reason,
            ):
                return (
                    FunctionMessage(
                        name=name,
                        content=f"Error: {error.message}",
                        result=None,
                        tags={
                            Tag("function", name),
                            Tag("error", "function_call"),
                        },
                    ),
                    finish_reason,
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
            FinishReason.FAILURE,
        )

    def _repr_mimebundle_(
        self, include: Sequence[str], exclude: Sequence[str], **kwargs
    ) -> dict[str, str]:
        return self.messages._repr_mimebundle_(include, exclude, **kwargs)

    def add(self, message: Message):
        self.messages.add(message)

    def call(
        self,
        *,
        prompt: Callable[..., Sequence[Message]],
        **call_kwargs: Unpack[CallKwargs],
    ):
        def merge_prompt(
            *prompt_args: P.args, **prompt_kwargs: P.kwargs
        ) -> Sequence[Message]:
            return [*self.messages.messages, *prompt(*prompt_args, **prompt_kwargs)]

        def decorator(func: Wrapped[P, R_co]) -> Wrapped[P, R_co]:
            @functools.wraps(func)
            def wrapper(*prompt_args: P.args, **prompt_kwargs: P.kwargs) -> R_co:
                function_call = self.llm.call(prompt=merge_prompt, **call_kwargs)(func)

                response = function_call(*prompt_args, **prompt_kwargs)
                self.messages.add_messages(
                    [
                        *prompt(*prompt_args, **prompt_kwargs),
                        AssistantMessage(content=str(response)),
                    ]
                )

                if self.autocommit:
                    self.messages.commit()
                self.dispatcher.reset()

                return response

            return wrapper

        return decorator

    @validate_call
    def complete(
        self,
        message: Message | None = None,
        context: set[TagPattern] | None = None,
        **complete_kwargs: Unpack[CompleteKwargs],
    ) -> Completion:
        return next(iter(self.complete_choices(message, context, **complete_kwargs)))

    @validate_call
    def complete_choices(
        self,
        message: Message | None = None,
        context: set[TagPattern] | None = None,
        **complete_kwargs: Unpack[CompleteKwargs],
    ) -> list[Completion]:
        try:
            return self._complete_choices(message, context, **complete_kwargs)
        except Exception as e:
            self._handle_exception(e)

    @validate_call
    def complete_flow(
        self,
        message: Message | None = None,
        context: set[TagPattern] | None = None,
    ) -> Completion:
        self.messages.add(message)
        try:
            for _ in range(self.max_iterations):
                completion = self.complete(context=context)
                match completion.finish_reason:
                    case FinishReason.STOP:
                        return completion
            return self._complete_failure()
        except (Exception, KeyboardInterrupt) as e:
            self._handle_exception(e)

    def commit(self):
        self.messages.commit()

    def load(self, session_id: UUID4, **kwargs):
        arguments = self.storage_backend.load(session_id)
        # currently, llms are not serialized
        if "llm" not in kwargs:
            arguments["llm"] = self.llm
        if "storage_backend" not in kwargs:
            arguments["storage_backend"] = self.storage_backend
        arguments.update(kwargs)
        validated_arguments = vars(type(self)(**arguments))
        vars(self).update(validated_arguments)

    def reset(self):
        self.messages.clear()
        self.messages.add(self.system_message)
        self.messages.add_messages(self.init_messages)
        self.messages.commit()

        self.dispatcher.reset()

    def revert(self):
        self.messages.revert()

    def rollback(self):
        self.messages.rollback()

        self.dispatcher.reset()

    def save(self):
        self.storage_backend.save(self, self.session_id)

    @property
    def tokens_used(self) -> Usage:
        usages = (response["usage"].values() for response in self.llm._responses)
        prompt_tokens, completion_tokens, choice_tokens = zip(*usages, strict=True)
        return Usage(
            prompt=sum(prompt_tokens),
            completion=sum(completion_tokens),
            total=sum(choice_tokens),
        )
