import inspect
from collections.abc import Callable
from types import BuiltinFunctionType, FunctionType
from typing import Annotated, Any, get_origin, get_type_hints

from pydantic import TypeAdapter, ValidationError, validate_call
from pydantic.dataclasses import dataclass

from lalia.chat.completions import FinishReason


@dataclass
class Error:
    message: str


@dataclass
class Result:
    """
    An anonymous result type that wraps a result or an error.
    """

    result: Any | None = None
    error: Error | None = None
    finish_reason: FinishReason = FinishReason.DELEGATE


@dataclass
class FunctionCallResult:
    """
    A result type that is a superset of `Result` containg additional metadata.
    """

    name: str
    parameters: dict[str, Any]
    result: Any | None = None
    error: Error | None = None
    finish_reason: FinishReason = FinishReason.DELEGATE

    def to_string(self) -> str:
        match self.error, self.result:
            case None, result:
                return str(result)
            case Error(message), None:
                return f"Error: {message}"
            case _:
                raise ValueError("Either `error` or `result` must be `None`")


def is_callable_instance(x):
    return not isinstance(x, FunctionType | BuiltinFunctionType) and callable(x)


def get_schema(callable_: Callable[..., Any]) -> dict[str, Any]:
    if is_callable_instance(callable_):
        func = callable_.__call__
        name = type(callable_).__name__
        doc = func.__doc__ if func.__doc__ else type(callable_).__doc__
        parameters = dict(list(inspect.signature(func).parameters.items())[1:])

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
    wrapped = validate_call(func)
    try:
        results = wrapped(**arguments)
    except (TypeError, ValidationError) as e:
        return FunctionCallResult(
            name=func.__name__,
            parameters=arguments,
            error=Error(f"Invalid arguments. Please check the provided arguments: {e}"),
        )
    if isinstance(results, FunctionCallResult):
        return results

    if isinstance(results, Result):
        return FunctionCallResult(
            name=func.__name__,
            parameters=arguments,
            result=results.result,
            error=results.error,
            finish_reason=results.finish_reason,
        )

    if isinstance(results, str) and results.startswith("Error:"):
        return FunctionCallResult(
            name=func.__name__,
            parameters=arguments,
            error=Error(
                results.removeprefix("Error:").strip(" "),
            ),
        )
    return FunctionCallResult(
        name=func.__name__,
        parameters=arguments,
        result=results,
    )
