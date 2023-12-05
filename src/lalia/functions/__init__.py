from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import (
    Annotated,
    Any,
    ParamSpec,
    TypeVar,
    get_origin,
    get_type_hints,
)

from jsonref import replace_refs
from pydantic import TypeAdapter, ValidationError, validate_call

from lalia.functions.types import (
    BaseProp,
    Error,
    FunctionCallResult,
    FunctionSchema,
    ObjectProp,
    Result,
    ReturnType,
)
from lalia.functions.utils import is_callable_instance
from lalia.io.logging import get_logger

logger = get_logger(__name__)


T = TypeVar("T")
A = TypeVar("A")
P = ParamSpec("P")


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


def get_schema(
    callable_: Callable[..., Any], include_return_types: bool = False
) -> FunctionSchema:
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

    properties = {}
    required_params = []

    for param_name, param_info in func_schema["properties"].items():
        if "required" in func_schema and param_name in func_schema["required"]:
            required_params.append(param_name)

        type_hints = get_type_hints(func, include_extras=True)
        annotation = type_hints.get(param_name)
        if annotation and get_origin(annotation) is Annotated:
            description = next(iter(annotation.__metadata__), "")
        else:
            description = ""

        properties[param_name] = BaseProp.parse_type_and_description(param_info)
        properties[param_name].description = description

    # TODO: ObjectProp needs to get VariantProp too
    function_parameters = ObjectProp(
        properties=properties,
        required=required_params if required_params else None,
    )

    if include_return_types:
        return_type_hint = (
            get_type_hints(func).get("return", None) if include_return_types else "any"
        )
        return_type = ReturnType(type=return_type_hint)
    else:
        return_type = None

    function_schema = FunctionSchema(
        name=name,
        description=inspect.cleandoc(doc) if doc else "",
        parameters=function_parameters,
        return_type=return_type,
    )

    return function_schema


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
