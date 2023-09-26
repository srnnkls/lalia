import inspect
from collections.abc import Callable, Sequence
from types import BuiltinFunctionType, FunctionType
from typing import Annotated, Any, get_origin, get_type_hints

from pydantic import ValidationError, validate_arguments
from pydantic.dataclasses import dataclass

from lalia.chat.messages import Message


@dataclass
class Error:
    message: str


@dataclass
class FunctionCallResult:
    name: str
    parameters: dict[str, Any]
    result: dict[str, Any] | None = None
    error: Error | None = None

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
    elif callable(callable_):
        func = callable_
        name = func.__name__
        doc = func.__doc__
    else:
        raise ValueError(f"Not a callable: {callable_}")

    pydantic_schema = validate_arguments(func).model.schema()
    parameters = inspect.signature(func).parameters

    schema = {
        "name": name,
        "description": inspect.cleandoc(doc) if doc else None,
        "parameters": {"type": "object", "properties": {}},
    }

    schema["parameters"]["properties"] = {
        param: data
        for param, data in pydantic_schema["properties"].items()
        if param in parameters
    }

    for param, data in schema["parameters"]["properties"].items():
        annotation = get_type_hints(func, include_extras=True)[param]
        if get_origin(annotation) is Annotated:
            data["description"] = next(iter(annotation.__metadata__), "")

    schema["required"] = pydantic_schema["required"]

    return schema


def execute_function_call(
    func: Callable[..., Any], arguments: dict[str, Any]
) -> FunctionCallResult:
    validator = validate_arguments(func).model
    try:
        validator.parse_obj(arguments)
    except ValidationError as e:
        return FunctionCallResult(
            name=func.__name__,
            parameters=arguments,
            error=Error(f"Invalid arguments. Please check the function signature: {e}"),
        )
    results = func(**arguments)
    if isinstance(results, FunctionCallResult):
        return results

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
        result={"result": results},
    )
