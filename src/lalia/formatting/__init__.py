import re
from enum import StrEnum
from textwrap import indent as tw_indent

from lalia.formatting.templates import (
    DESCRIPTION_TEMPLATE,
    FUNCTION_TEMPLATE,
    PARAMETER_TEMPLATE,
)
from lalia.functions import FunctionSchema
from lalia.io.serialization.json_schema import (
    AllOfProp,
    AnyOfProp,
    ArrayProp,
    BooleanProp,
    IntegerProp,
    JsonSchemaType,
    NotProp,
    NullProp,
    NumberProp,
    ObjectProp,
    OneOfProp,
    Prop,
    StringProp,
)


class TypeScriptType(StrEnum):
    NUMBER = "number"  # JSON Schema number maps to TypeScript number
    INTEGER = "number"  # JSON Schema integer also maps to TypeScript number
    STRING = "string"
    BOOLEAN = "boolean"
    ARRAY = "any[]"
    OBJECT = "object"
    NULL = "null"
    # Add other types as needed


def json_schema_type_to_ts(type_: JsonSchemaType) -> str:
    """
    Maps a JsonSchemaType to its TypeScript equivalent.
    """
    return TypeScriptType[type_.name]


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


def format_parameter_type(param: Prop | None) -> str:
    """
    Determines the TypeScript type representation of a given parameter.
    """
    match param:
        case StringProp() as prop:
            if prop.enum:
                # string enums in TS have quoted values
                return " | ".join(f'"{v}"' for v in prop.enum)
            return json_schema_type_to_ts(prop.type_)
        case IntegerProp() | NumberProp() as prop:
            if prop.enum:
                # number enums in TS do not have quoted values
                return " | ".join(f"{v}" for v in prop.enum)
            return json_schema_type_to_ts(prop.type_)
        case BooleanProp() | NullProp() as prop:
            return json_schema_type_to_ts(prop.type_)
        case ArrayProp(items=items):
            return f"{format_parameter_type(items)}[]"
        case (
            OneOfProp(one_of=items) | AnyOfProp(any_of=items) | AllOfProp(all_of=items)
        ):
            return " | ".join(format_parameter_type(item) for item in items)
        case ObjectProp() as prop:
            return json_schema_type_to_ts(prop.type_)
        case NotProp():
            return "any"
        case _:
            return "any"


def format_optional(prop: Prop) -> str:
    # if there is a default, the parameter is optional
    if prop.default:
        return "?"
    return ""


def format_parameter(name: str, param: Prop, indent: str = "") -> str:
    """
    Formats a function parameter, including its type and default value
    as its TypeScript representation.
    """
    description = format_description(param.description)

    optional = format_optional(param)

    formatted_type = format_parameter_type(param)

    return tw_indent(
        PARAMETER_TEMPLATE.format(
            description=description,
            name=name,
            optional=optional,
            type_=formatted_type,
        ),
        indent,
    )


def format_function_model(function_model: FunctionSchema) -> str:
    """
    Generates the TypeScript representation of a function model.
    """
    description = format_description(function_model.description)
    if function_model.parameters:
        parameters = [
            format_parameter(k, v, indent="")
            for k, v in function_model.parameters.properties.items()  # type: ignore
        ]
    else:
        parameters = ""
    formatted_parameters = tw_indent("".join(parameters), "    ")
    return FUNCTION_TEMPLATE.format(
        description=description,
        name=function_model.name,
        parameters=formatted_parameters,
        return_type="any",
    )


def format_function_as_typescript(function_to_convert: FunctionSchema) -> str:
    """Converts a FunctionSchema into its TypeScript type signature
    representation."""
    return format_function_model(function_to_convert)
