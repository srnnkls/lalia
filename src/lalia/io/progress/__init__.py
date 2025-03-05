import sys
from dataclasses import InitVar, field
from datetime import UTC, datetime
from typing import Protocol, TextIO, runtime_checkable

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

MESSAGE_TEMPLATE = "{timestamp:%Y-%m-%d %H:%M:%S} {msg}"

DEFAULT_INITIAL_STATE = "idle"


@runtime_checkable
class Representable(Protocol):
    def __repr__(self) -> str: ...


ProgressState = Representable


@runtime_checkable
class Progress(Protocol):
    @property
    def state(self) -> ProgressState: ...


@runtime_checkable
class ProgressFormatter(Protocol):
    def format(self, progress: Progress) -> str: ...


@runtime_checkable
class ProgressHandler(Protocol):
    def emit(self, progress: Progress): ...


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ProgressManager:
    handler: ProgressHandler
    initial_state: InitVar[ProgressState] = DEFAULT_INITIAL_STATE

    def __post_init__(self, initial_state: ProgressState):
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

    def format(self, progress: Progress, end: str = "\n") -> str:
        msg = f"Progress: {progress.state!r}"

        return self.msg_template.format(timestamp=datetime.now(UTC), msg=f"{msg}{end}")


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
