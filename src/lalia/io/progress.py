import sys
from dataclasses import InitVar, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Generic, Protocol, TextIO, TypeVar, runtime_checkable

from pydantic import ConfigDict, create_model
from pydantic.dataclasses import dataclass
from rich.pretty import pretty_repr

MESSAGE_TEMPLATE = "{timestamp:%Y-%m-%d %H:%M:%S} {msg}"

StateType = TypeVar("StateType", bound=Enum)


class ProgressState(StrEnum):
    IDLE = "idle"  # Session is waiting for user input
    GENERATING = "generating"  # LLM is generating response
    EXECUTING = "executing"  # Session is executing a function
    ABORTED = "aborted"  # aborted by user
    SUSPENDED = "suspended"  # suspended by function


@runtime_checkable
class Progress(Protocol, Generic[StateType]):
    state: StateType
    iteration: int | None
    function: str | None
    arguments: dict[str, Any] | None


@runtime_checkable
class ProgressFormatter(Protocol):
    def format(self, progress: Progress) -> str: ...


@runtime_checkable
class ProgressHandler(Protocol):
    def emit(self, progress: Progress): ...


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ProgressManager(Generic[StateType]):
    handler: ProgressHandler
    initial_state: InitVar[StateType]

    def __post_init__(self, initial_state: StateType):
        self.state = initial_state

    def abort(self):
        raise NotImplementedError

    def emit(self, progress: Progress):
        self.state = progress.state
        self.handler.emit(progress)

    def resume(self):
        raise NotImplementedError

    def suspend(self):
        raise NotImplementedError


@dataclass
class StreamProgressFormatter:
    msg_template: str = MESSAGE_TEMPLATE

    def format(self, progress: Progress) -> str:
        match progress:
            case Progress(
                state=ProgressState.EXECUTING,
                iteration=int() as iteration,
                function=str() as function,
                arguments=dict() as arguments,
            ):
                model = create_model(
                    function,
                    **{name: (type(arg), ...) for name, arg in arguments.items()},  # type: ignore
                )  # type: ignore
                instance = model(**arguments)

                msg = f"Executing {pretty_repr(instance)}...\nIteration: {iteration}\n"
            case Progress(state=ProgressState.GENERATING, function=function):
                if function:
                    msg = f"LLM is generating parameters for function: {function}...\n"
                msg = "LLM is generating response...\n"
            case ProgressState.IDLE, None:
                msg = "Waiting for user input...\n"
            case _:
                msg = f"Progress: {progress.state!r}\n"

        return self.msg_template.format(timestamp=datetime.now(UTC), msg=msg)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class StreamProgressHandler:
    formatter: ProgressFormatter = field(default_factory=StreamProgressFormatter)
    stream: TextIO = sys.stdout

    def emit(self, progress: Progress):
        progress_formatted = self.formatter.format(progress)
        self.stream.write(progress_formatted)

    def flush(self):
        if self.stream and hasattr(self.stream, "flush"):
            self.stream.flush()


@dataclass
class NopProgressHandler:
    def emit(self, progress: Progress):
        pass
