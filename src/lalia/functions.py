from __future__ import annotations

import inspect
from collections.abc import Callable
from types import BuiltinFunctionType, FunctionType
from typing import (
    Annotated,
    Any,
    Generic,
    ParamSpec,
    TypeVar,
    get_origin,
    get_type_hints,
)

from jsonref import replace_refs
from pydantic import TypeAdapter, ValidationError, validate_call
from pydantic.dataclasses import dataclass

from lalia.chat.finish_reason import FinishReason
from lalia.io.logging import get_logger

# need to import Prop so FunctionSchema is correctly defined for pydantic
from lalia.io.serialization.json_schema import ObjectProp, Prop  # noqa

logger = get_logger(__name__)


T = TypeVar("T")
A = TypeVar("A")
P = ParamSpec("P")


@dataclass
class Error:
    message: str


@dataclass
class Result:
    """
    An anonymous result type that wraps a result or an error.
    """

    value: Any | None = None
    error: Error | None = None
    finish_reason: FinishReason = FinishReason.DELEGATE


@dataclass
class FunctionCallResult(Generic[A, T]):
    """
    A result type that is a superset of `Result` containg additional metadata.
    """

    name: str
    arguments: dict[str, A]
    value: T | None = None
    error: Error | None = None
    finish_reason: FinishReason = FinishReason.DELEGATE

    def to_string(self) -> str:
        match self.error, self.value:
            case None, result:
                return str(result)
            case Error(message), None:
                return f"Error: {message}"
            case _:
                raise ValueError("Either `error` or `result` must be `None`")


@dataclass
class FunctionSchema:
    """Describes a function schema, including its parameters."""

    name: str
    parameters: ObjectProp | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return TypeAdapter(type(self)).dump_python(
            self, exclude_none=True, by_alias=True
        )


def is_callable_instance(callable_: object) -> bool:
    if not callable(callable_):
        return False
    return not isinstance(callable_, FunctionType | BuiltinFunctionType)


def dereference_schema(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in replace_refs(schema, proxies=False).items()  # type: ignore
        if key != "$defs"
    }


def get_name(callable_: Callable[P, Any]) -> str:
    if is_callable_instance(callable_):
        return getattr(callable_, "name", type(callable_).__name__)
    return callable_.__name__


def get_callable(callable_: Callable[P, T]) -> Callable[P, T]:
    if is_callable_instance(callable_):
        return callable_.__call__
    return callable_


def get_schema(callable_: Callable[..., Any]) -> FunctionSchema:
    if is_callable_instance(callable_):
        func = callable_.__call__
        name = getattr(callable_, "name", type(callable_).__name__)
        doc = func.__doc__ if func.__doc__ else type(callable_).__doc__

    elif callable(callable_):
        func = callable_
        name = func.__name__
        doc = func.__doc__

    else:
        raise ValueError(f"Not a callable: {callable_}")

    adapter = TypeAdapter(validate_call(func))
    func_schema = dereference_schema(adapter.json_schema())

    type_hints = get_type_hints(func, include_extras=True)

    for prop_name, _ in func_schema["properties"].items():
        annotation = type_hints.get(prop_name)
        if annotation and get_origin(annotation) is Annotated:
            description = next(iter(annotation.__metadata__), "")
        else:
            description = ""

        func_schema["properties"][prop_name]["description"] = description

    return FunctionSchema(
        name=name,
        description=inspect.cleandoc(doc) if doc else "",
        parameters=ObjectProp(**func_schema),
    )


def execute_function_call(
    func: Callable[..., T], arguments: dict[str, A]
) -> FunctionCallResult[A, T]:
    func_with_validation = validate_call(get_callable(func))
    try:
        result = func_with_validation(**arguments)
    except (TypeError, ValidationError) as e:
        logger.debug(e)
        return FunctionCallResult(
            name=get_name(func),
            arguments=arguments,
            error=Error(f"Invalid arguments. Please check the provided arguments: {e}"),
        )

    match result:
        case FunctionCallResult():
            return result
        case Result(value, error, finish_reason):
            if error is not None:
                logger.debug(error)
            return FunctionCallResult(
                name=get_name(func),
                arguments=arguments,
                value=value,
                error=error,
                finish_reason=finish_reason,
            )
        case str():
            if result.startswith("Error:"):
                error = Error(result.removeprefix("Error:").strip())
                logger.debug(error)

                return FunctionCallResult(
                    name=get_name(func),
                    arguments=arguments,
                    error=error,
                )
            else:
                return FunctionCallResult(
                    name=get_name(func),
                    arguments=arguments,
                    value=result,
                )
        case _:
            return FunctionCallResult(
                name=get_name(func),
                arguments=arguments,
                value=result,
            )
