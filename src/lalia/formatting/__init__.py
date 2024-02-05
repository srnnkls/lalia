from enum import StrEnum
import json

from lalia.formatting.templates import (
    FUNCTION_TEMPLATE,
    NAMESPACE_TEMPLATE,
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
        paragraphs = description.replace("\r", "").split("\n")
        formatted_lines = ["// " + paragraph for paragraph in paragraphs]
        return "\n".join(formatted_lines)
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
    # using getattr to avoid errors for props that don't have a default property
    default = getattr(prop, "default", None)
    return "?" if default is not None else ""


def format_default(prop: Prop) -> str:
    # using getattr to avoid errors for props that don't have a default property
    default = getattr(prop, "default", None)

    # booleans are lowercase in TS
    match prop:
        case BooleanProp():
            return f" // default: {json.dumps(default)}" if default is not None else ""
        case _:
            return f" // default: {default}" if default is not None else ""


def format_parameter(name: str, param: Prop) -> str:
    """
    Formats a function parameter, including its type and default value
    as its TypeScript representation.
    """
    description = format_description(param.description)
    optional = format_optional(param)
    default_value = format_default(param)
    formatted_type = format_parameter_type(param)

    # include a newline only if there is a description
    description_line = f"{description}\n" if description else ""

    param_format = PARAMETER_TEMPLATE.format(
        description=description_line,
        name=name,
        optional=optional,
        type_=formatted_type,
        default=default_value,
    )
    return param_format


def format_function_model(function_model: FunctionSchema) -> str:
    """
    Generates the TypeScript representation of a function model.
    """
    description = format_description(function_model.description)

    # function description needs extra newline
    description_section = "\n" + description if description else ""

    # TODO: use pattern matching
    if function_model.parameters and function_model.parameters.properties:
        parameters = [
            format_parameter(k, v)
            for k, v in function_model.parameters.properties.items()
        ]
        formatted_parameters = "\n".join(parameters)
        # TODO: extra section template
        parameters_section = "(_: {\n" + formatted_parameters + "\n})"
    else:
        parameters_section = "()"

    return FUNCTION_TEMPLATE.format(
        description=description_section,
        name=function_model.name,
        parameters=parameters_section,
    )


def format_function_as_typescript(function_to_convert: FunctionSchema) -> str:
    """Converts a FunctionSchema into its TypeScript type signature
    representation."""
    return format_function_model(function_to_convert)


def format_functions_as_typescript_namespace(
    functions_to_convert: list[FunctionSchema],
) -> str:
    """Converts a list of FunctionSchemas into their TypeScript namespace
    representation."""
    function_models = [
        format_function_model(function) for function in functions_to_convert
    ]
    function_declarations = "\n".join(function_models)
    return NAMESPACE_TEMPLATE.format(functions=function_declarations)
