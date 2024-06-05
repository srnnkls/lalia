from dataclasses import field
from datetime import UTC, datetime
from enum import StrEnum
from typing import (
    Annotated,
    Any,
    Literal,
)

from pydantic import Field, create_model
from pydantic.dataclasses import dataclass
from rich.pretty import pretty_repr

from lalia.io.progress import MESSAGE_TEMPLATE, Progress


class SessionProgressState(StrEnum):
    IDLE = "idle"  # Session is waiting for user input
    GENERATING = "generating"  # LLM is generating response
    EXECUTING = "executing"  # Session is executing a function


@dataclass
class IdlingProgress:
    state: Literal[SessionProgressState.IDLE] = SessionProgressState.IDLE


@dataclass
class GeneratingProgress:
    functions: list[str] = field(default_factory=list)
    state: Literal[SessionProgressState.GENERATING] = SessionProgressState.GENERATING


@dataclass
class ExecutingProgress:
    function: str
    arguments: dict[str, Any] | None = None
    iteration: int = 1
    state: Literal[SessionProgressState.EXECUTING] = SessionProgressState.EXECUTING


SessionProgress = Annotated[
    IdlingProgress | GeneratingProgress | ExecutingProgress,
    Field(discriminator="state"),
]


@dataclass
class SessionStreamProgressFormatter:
    msg_template: str = MESSAGE_TEMPLATE

    def format(self, progress: SessionProgress, end: str = "\n") -> str:
        match progress:
            case ExecutingProgress(
                iteration=int() as iteration,
                function=function_name,
                arguments=dict() as arguments,
            ):
                model = create_model(
                    function_name,
                    **{name: (type(arg), ...) for name, arg in arguments.items()},  # type: ignore
                )  # type: ignore
                instance = model(**arguments)

                msg = f"Executing {pretty_repr(instance)}...\nIteration: {iteration}"
            case GeneratingProgress(functions=[function]):
                msg = f"LLM is generating parameters for function: {function}..."
            case GeneratingProgress():
                msg = "LLM is generating response..."
            case Progress(state=state):
                msg = f"Progress: {state!r}"
            case _:
                raise ValueError(f"Invalid progress state: {progress.state!r}")

        return self.msg_template.format(timestamp=datetime.now(UTC), msg=f"{msg}{end}")
