import inspect
from collections.abc import Callable
from types import BuiltinFunctionType, FunctionType
from typing import Annotated, Any, get_origin, get_type_hints

from pydantic import TypeAdapter, ValidationError, validate_call
from pydantic.dataclasses import dataclass

from lalia.chat.finish_reason import FinishReason


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
class FunctionCallResult:
    """
    A result type that is a superset of `Result` containg additional metadata.
    """

    name: str
    parameters: dict[str, Any]
    value: Any | None = None
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


def is_callable_instance(x):
    return not isinstance(x, FunctionType | BuiltinFunctionType) and callable(x)


def get_name(callable_: Callable[..., Any]) -> str:
    if is_callable_instance(callable_):
        return getattr(callable_, "name", type(callable_).__name__)
    return callable_.__name__


def get_callable(callable_: Callable[..., Any]) -> Callable[..., Any]:
    if is_callable_instance(callable_):
        return callable_.__call__
    return callable_


def get_schema(callable_: Callable[..., Any]) -> dict[str, Any]:
    if is_callable_instance(callable_):
        func = callable_.__call__
        name = getattr(callable_, "name", type(callable_).__name__)
        doc = func.__doc__ if func.__doc__ else type(callable_).__doc__
        parameters = inspect.signature(func).parameters

    elif callable(callable_):
        func = callable_
        name = func.__name__
        doc = func.__doc__
        parameters = inspect.signature(func).parameters

    else:
        raise ValueError(f"Not a callable: {callable_}")

    adapter = TypeAdapter(validate_call(func))
    func_schema = adapter.json_schema()

    schema = {
        "name": name,
        "description": inspect.cleandoc(doc) if doc else None,
        "parameters": {"type": "object", "properties": {}},
    }

    schema["parameters"]["properties"] = {
        param: data
        for param, data in func_schema["properties"].items()
        if param in parameters
    }

    for param, data in schema["parameters"]["properties"].items():
        annotation = get_type_hints(func, include_extras=True)[param]
        if get_origin(annotation) is Annotated:
            data["description"] = next(iter(annotation.__metadata__), "")

    schema["required"] = [
        name for name in func_schema["required"] if name in parameters
    ]

    return schema


def execute_function_call(
    func: Callable[..., Any], arguments: dict[str, Any]
) -> FunctionCallResult:
    wrapped = validate_call(get_callable(func))
    try:
        result = wrapped(**arguments)
    except (TypeError, ValidationError) as e:
        return FunctionCallResult(
            name=get_name(func),
            parameters=arguments,
            error=Error(f"Invalid arguments. Please check the provided arguments: {e}"),
        )
    if isinstance(result, FunctionCallResult):
        return result

    if isinstance(result, Result):
        return FunctionCallResult(
            name=get_name(func),
            parameters=arguments,
            value=result.value,
            error=result.error,
            finish_reason=result.finish_reason,
        )

    if isinstance(result, str) and result.startswith("Error:"):
        return FunctionCallResult(
            name=get_name(func),
            parameters=arguments,
            error=Error(
                result.removeprefix("Error:").strip(" "),
            ),
        )
    return FunctionCallResult(
        name=get_name(func),
        parameters=arguments,
        value=result,
    )
