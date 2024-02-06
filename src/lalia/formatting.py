import json
from enum import StrEnum
from inspect import cleandoc
from typing import Any, Generic, Protocol, TypeVar

from lalia.chat.messages.messages import Message
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

OPENAI_NAMESPACE_TEMPLATE = cleandoc(
    """
    // Tools

    // Functions

    namespace functions {{
    {functions}

    }} // namespace functions
    """
)
OPENAI_FUNCTION_TEMPLATE = cleandoc(
    """
    {description}
    type {name} = {parameters} => any;
    """
)
OPENAI_PARAMETER_TEMPLATE = "{description}{name}{optional}: {type_},{default}"


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


# Pyright demanded this to be contravariant
T_contra = TypeVar("T_contra", contravariant=True)


class Formatter(Protocol, Generic[T_contra]):
    def format(self, value: T_contra) -> str: ...


class OpenAIFunctionFormatter:
    namespace_template = OPENAI_NAMESPACE_TEMPLATE
    function_template = OPENAI_FUNCTION_TEMPLATE
    parameter_template = OPENAI_PARAMETER_TEMPLATE

    def format(
        self,
        value: FunctionSchema | dict[str, Any] | list[FunctionSchema | dict[str, Any]],
    ) -> str:
        match value:
            case FunctionSchema():
                # single FunctionSchema instance
                function_schemas = [value]
            case dict():
                # single dictionary
                function_schemas = [FunctionSchema(**value)]
            case list() as schemas:
                # list of FunctionSchema instances or dicts
                function_schemas = [
                    FunctionSchema(**schema) if isinstance(schema, dict) else schema
                    for schema in schemas
                ]
            case _:
                raise TypeError(
                    f"Unsupported input type for TypeScript formatting: {type(value)}"
                )

        return self._format_functions_as_typescript_namespace(function_schemas)

    def _format_description(self, description: str | None) -> str:
        """
        Returns a formatted string for the description, suitable for inclusion in
        TypeScript code.
        """
        if description:
            paragraphs = description.replace("\r", "").split("\n")
            formatted_lines = [f"// {paragraph}" for paragraph in paragraphs]
            return "\n".join(formatted_lines)
        return ""

    def _format_optional(self, prop: Prop) -> str:
        """
        Returns a string representing whether a parameter is optional.
        """
        # if there is a default, the parameter is optional
        # using getattr to avoid errors for props that don't have a default property
        default = getattr(prop, "default", None)
        return "?" if default is not None else ""

    def _format_default(self, prop: Prop) -> str:
        """
        Returns a string representing the default value of a parameter.
        """
        # using getattr to avoid errors for props that don't have a default property
        default = getattr(prop, "default", None)

        # booleans are lowercase in TS
        match prop:
            case BooleanProp():
                return (
                    f" // default: {json.dumps(default)}" if default is not None else ""
                )
            case _:
                return f" // default: {default}" if default is not None else ""

    def _format_parameter_type(self, param: Prop | None) -> str:
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
                return f"{self._format_parameter_type(items)}[]"
            case (
                OneOfProp(one_of=items)
                | AnyOfProp(any_of=items)
                | AllOfProp(all_of=items)
            ):
                return " | ".join(self._format_parameter_type(item) for item in items)
            case ObjectProp() as prop:
                return json_schema_type_to_ts(prop.type_)
            case NotProp():
                return "any"
            case _:
                return "any"

    def _format_parameter(self, name: str, param: Prop) -> str:
        """
        Formats a function parameter, including its type and default value
        as its TypeScript representation.
        """
        description = self._format_description(param.description)
        optional = self._format_optional(param)
        default_value = self._format_default(param)
        formatted_type = self._format_parameter_type(param)

        # include a newline only if there is a description
        description_line = f"{description}\n" if description else ""

        param_format = self.parameter_template.format(
            description=description_line,
            name=name,
            optional=optional,
            type_=formatted_type,
            default=default_value,
        )
        return param_format

    def _format_function_model(self, function_model: FunctionSchema) -> str:
        """
        Generates the TypeScript representation of a function model.
        """
        description = self._format_description(function_model.description)

        # function description needs extra newline
        description_section = f"\n{description}" if description else ""

        # TODO: use pattern matching
        if function_model.parameters and function_model.parameters.properties:
            parameters = [
                self._format_parameter(k, v)
                for k, v in function_model.parameters.properties.items()
            ]
            formatted_parameters = "\n".join(parameters)
            # TODO: extra section template
            parameters_section = f"(_: {{\n{formatted_parameters}\n}})"
        else:
            parameters_section = "()"

        return self.function_template.format(
            description=description_section,
            name=function_model.name,
            parameters=parameters_section,
        )

    def _format_function_as_typescript(
        self, function_to_convert: FunctionSchema
    ) -> str:
        """
        Converts a FunctionSchema into its TypeScript type signature
        representation.
        """
        return self._format_function_model(function_to_convert)

    def _format_functions_as_typescript_namespace(
        self, functions_to_convert: list[FunctionSchema]
    ) -> str:
        """
        Converts a list of FunctionSchemas into their TypeScript namespace
        representation.
        """
        function_models = [
            self._format_function_model(function) for function in functions_to_convert
        ]
        function_declarations = "\n".join(function_models)
        return self.namespace_template.format(functions=function_declarations)


class OpenAIMessageFormatter:
    def format(self, value: Message) -> str:
        raise NotImplementedError
