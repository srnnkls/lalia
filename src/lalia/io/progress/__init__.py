import sys
from dataclasses import InitVar, field
from datetime import UTC, datetime
from typing import Any, Protocol, TextIO, runtime_checkable

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from rich import print

MESSAGE_TEMPLATE = "{timestamp:%Y-%m-%d %H:%M:%S} {msg}"


# Probably not needed, as all objects can be considered Representable
# i.e. doesn;t tell more that Any
@runtime_checkable
class Representable(Protocol):
    def __repr__(self) -> str: ...


ProgressState = Representable


@runtime_checkable
class Progress(Protocol):
    state: ProgressState


@runtime_checkable
class ProgressFormatter(Protocol):
    def format(self, progress: Progress) -> str: ...


@runtime_checkable
class ProgressHandler(Protocol):
    def emit(self, progress: Progress): ...


@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class ProgressManager:
    handler: ProgressHandler
    initial_state: InitVar[ProgressState]

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
        print(progress)
        self.stream.write(progress_formatted)

    def flush(self):
        if self.stream and hasattr(self.stream, "flush"):
            self.stream.flush()


@dataclass
class NopProgressHandler:
    def emit(self, progress: Progress):
        pass
