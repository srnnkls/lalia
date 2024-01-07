import re
from textwrap import indent as tw_indent
from typing import Any

from lalia.formatting.templates import (
    DESCRIPTION_TEMPLATE,
    FUNCTION_TEMPLATE,
    PARAMETER_TEMPLATE,
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


def format_parameter_type(param: dict[str, Any]) -> str:
    """
    Determines the TypeScript type representation of a given parameter.
    """
    match param["type"]:
        case "string" | "string" | "boolean" | "null" as type_:
            if param.get("enum"):
                return " | ".join(f'"{v}"' for v in param["enum"])
            return type_
        case "array":
            return f"{format_parameter_type(param['items'])}[]"
        case "object":
            return "object"
        case _:
            return "any"


def format_parameter(name: str, param: dict[str, Any], indent: str) -> str:
    """
    Formats a function parameter, including its type and default value
    as its TypeScript representation.
    """
    description = format_description(param.get("description"))
    default = f" = {param.get('default', '')!r}"
    optional = "?" if default else ""

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


def format_function_model(function_model: dict[str, Any]) -> str:
    """
    Generates the TypeScript representation of a function model.
    """
    description = format_description(function_model["description"])
    parameters = [
        format_parameter(k, v, indent="")
        for k, v in function_model["parameters"]["properties"].items()
    ]
    parameters_str = tw_indent("\n".join(parameters), "    ")
    return FUNCTION_TEMPLATE.format(
        description=description,
        name=function_model["name"],
        parameters=parameters_str,
        return_type="any",
    )


def format_function_as_typescript(function_to_convert: dict[str, Any]) -> str:
    """Converts a FunctionSchema into its TypeScript type signature
    representation."""
    return format_function_model(function_to_convert)
