import re
from collections.abc import Callable
from textwrap import indent as tw_indent
from types import UnionType
from typing import Any, get_origin

from lalia.formatting.templates import (
    DESCRIPTION_TEMPLATE,
    FUNCTION_TEMPLATE,
    PARAMETER_TEMPLATE,
)
from lalia.functions import get_schema
from lalia.functions.types import (
    ArrayProp,
    BaseProp,
    BoolProp,
    FunctionSchema,
    NullProp,
    NumberProp,
    ObjectProp,
    PropItem,
    ReturnType,
    StringProp,
    TypeScriptTypes,
    VariantProp,
)


def format_description(description: str | None) -> str:
    """
    Returns a formatted string for the description, suitable for inclusion in
    TypeScript code.
    """
    if description:
        paragraphs = re.split(r"\n+", description.replace("\r", ""))
        single_line_description = " ".join(
            paragraph.strip() for paragraph in paragraphs if paragraph.strip()
        )
        return DESCRIPTION_TEMPLATE.format(description=single_line_description)
    return ""


def format_parameter_type(param: PropItem | None) -> str:
    """
    Determines the TypeScript type representation of a given parameter.
    """
    match param:
        case StringProp() as string_prop:
            if string_prop.enum:
                return " | ".join(f'"{v}"' for v in string_prop.enum)
            return "string"
        case NumberProp() as number_prop:
            if number_prop.enum:
                return " | ".join(f"{v}" for v in number_prop.enum)
            return "number"
        case BoolProp():
            return "boolean"
        case NullProp():
            return "null"
        case ArrayProp(items=BaseProp() as items):
            return f"{format_parameter_type(items)}[]"
        case ObjectProp(properties=dict() as properties):  # noqa: F841
            return "object"
        case _:
            return "any"


def format_return_type(return_type: ReturnType | Any) -> str:
    """
    Maps a Python return type to its TypeScript equivalent.
    """
    match return_type:
        case ReturnType(type=type):
            if get_origin(type) is UnionType:
                union_types = return_type.type.__args__
                ts_types = [TypeScriptTypes.from_python_type(t) for t in union_types]
                return " | ".join(ts_types)
            else:
                return TypeScriptTypes.from_python_type(return_type.type)
        case _:
            return "any"


def format_parameter(name: str, param: PropItem, indent="") -> str:
    """
    Formats a function parameter, including its type and default value
    as its TypeScript representation.
    """
    description = format_description(param.description)
    default = f" = {param.default!r}" if param.default is not None else ""
    optional = "?" if default else ""

    match param:
        case ObjectProp() as prop:
            properties = [
                f"{k}: {format_parameter_type(v)}" for k, v in prop.properties.items()
            ]
            type_str = "{ " + ", ".join(properties) + " }"
        case VariantProp() as prop:
            type_str = prop.type
        case _:
            type_str = format_parameter_type(param)

    return tw_indent(
        PARAMETER_TEMPLATE.format(
            description=description,
            name=name,
            optional=optional,
            type_=type_str,
            default=default,
        ),
        indent,
    )


def format_function_model(
    function_model: FunctionSchema, include_return_type: bool = False
) -> str:
    """
    Generates the TypeScript representation of a function model.
    """
    description = format_description(function_model.description)
    parameters = [
        format_parameter(k, v, indent="")
        for k, v in function_model.parameters.properties.items()
    ]
    parameters_str = tw_indent("\n".join(parameters), "    ")
    return_type_str = (
        format_return_type(function_model.return_type) if include_return_type else "any"
    )
    return FUNCTION_TEMPLATE.format(
        description=description,
        name=function_model.name,
        parameters=parameters_str,
        return_type=return_type_str,
    )


def format_function_as_typescript(
    function_to_convert: Callable, include_return_type: bool = False
):
    """Converts a Python function callable into its TypeScript type signature
    representation."""
    function_info = get_schema(function_to_convert)

    return format_function_model(function_info, include_return_type)
