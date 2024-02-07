import sys
from dataclasses import InitVar, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any, Generic, Protocol, TextIO, TypeVar, runtime_checkable

from pydantic import ConfigDict, create_model
from pydantic.dataclasses import dataclass
from rich.pretty import pretty_repr

MESSAGE_TEMPLATE = "{timestamp:%Y-%m-%d %H:%M:%S} {msg}"

ProgressStateType = TypeVar("ProgressStateType", bound=Enum)


class ProgressState(StrEnum):
    IDLE = "idle"  # Session is waiting for user input
    GENERATING = "generating"  # LLM is generating response
    EXECUTING = "executing"  # Session is executing a function


@runtime_checkable
class Progress(Protocol, Generic[ProgressStateType]):
    state: ProgressStateType
    iteration: int | None
    functions: list[str] | None
    arguments: dict[str, Any] | None


ProgressType_contra = TypeVar("ProgressType_contra", contravariant=True, bound=Progress)


@runtime_checkable
class ProgressFormatter(Protocol):
    def format(self, progress: Progress) -> str: ...


@runtime_checkable
class ProgressHandler(Protocol, Generic[ProgressType_contra]):
    def emit(self, progress: ProgressType_contra): ...


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ProgressManager(Generic[ProgressStateType]):
    handler: ProgressHandler
    initial_state: InitVar[ProgressStateType]

    def __post_init__(self, initial_state: ProgressStateType):
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
                functions=[function],
                arguments=dict() as arguments,
            ):
                model = create_model(
                    function,
                    **{name: (type(arg), ...) for name, arg in arguments.items()},  # type: ignore
                )  # type: ignore
                instance = model(**arguments)

                msg = f"Executing {pretty_repr(instance)}...\nIteration: {iteration}\n"
            case Progress(
                state=ProgressState.GENERATING, functions=list() as functions
            ):
                if len(functions) == 1:
                    msg = (
                        "LLM is generating parameters for function: "
                        f"{functions[0]}...\n"
                    )
                msg = "LLM is generating response...\n"
            case _:
                msg = f"Progress: {progress.state!r}\n"

        return self.msg_template.format(timestamp=datetime.now(UTC), msg=msg)


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class StreamProgressHandler(Generic[ProgressType_contra]):
    formatter: ProgressFormatter = field(default_factory=StreamProgressFormatter)
    stream: TextIO = sys.stdout

    def emit(self, progress: ProgressType_contra):
        progress_formatted = self.formatter.format(progress)
        self.stream.write(progress_formatted)

    def flush(self):
        if self.stream and hasattr(self.stream, "flush"):
            self.stream.flush()


@dataclass
class NopProgressHandler(Generic[ProgressType_contra]):
    def emit(self, progress: ProgressType_contra):
        pass
