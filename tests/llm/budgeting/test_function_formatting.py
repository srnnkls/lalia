from inspect import cleandoc

import pytest

from lalia.formatting import (
    format_description,
    format_function_as_typescript,
    format_parameter,
    format_parameter_type,
)
from lalia.functions import get_schema
from lalia.io.serialization.json_schema import (
    ArrayProp,
    BooleanProp,
    JsonSchemaType,
    NullProp,
    NumberProp,
    StringProp,
)


@pytest.fixture()
def string_prop():
    return StringProp(description="A string.")


@pytest.fixture()
def string_prop_with_enum():
    return StringProp(description="A string.", enum=["A", "B"])


@pytest.fixture()
def number_prop():
    return NumberProp(description="A number.")


@pytest.fixture()
def number_prop_with_enum():
    return NumberProp(description="A number.", enum=[0, 1])


@pytest.fixture()
def bool_prop():
    return BooleanProp(description="A boolean.")


@pytest.fixture()
def null_prop():
    return NullProp(description="A null.")


@pytest.fixture()
def foo_expected():
    return cleandoc(
        """
namespace functions {
// This is a test function. It combines number a, appends string b and option c.
  type foo = (_: {
    // This is a integer number.
    a: number;
    // This will be appended.
    b?: string | number;
    // This is an Enum.
    c?: "option1" | "option2";
  }) => any;
} // namespace functions
"""
    )


@pytest.fixture()
def foo_json_expected():
    return {
        "name": "foo",
        "parameters": {
            "properties": {
                "a": {
                    "description": "This is a integer number.",
                    "title": "A",
                    "type": JsonSchemaType.INTEGER,
                },
                "b": {
                    "description": "This will be appended.",
                    "title": "B",
                    "default": "test",
                    "anyOf": [
                        {
                            "type": JsonSchemaType.STRING,
                        },
                        {
                            "type": JsonSchemaType.INTEGER,
                        },
                    ],
                },
                "c": {
                    "description": "This is an Enum.",
                    "default": "option1",
                    "allOf": [
                        {
                            "title": "MyEnum",
                            "enum": ["option1", "option2"],
                            "type": JsonSchemaType.STRING,
                        }
                    ],
                },
            },
            "required": ["a"],
            "type": JsonSchemaType.OBJECT,
            "additionalProperties": False,
        },
        "description": "This is a test function.\nIt combines number a, appends string b and option c.",
    }


@pytest.fixture()
def function_schema(foo_function):
    """Has name: str, parameters: ObjectProp, description: str,
    return_type: ReturnType"""
    return get_schema(foo_function)


class TestDesciptionFormatting:
    def test_description_formatting(self):
        assert format_description("Test description") == "\n// Test description"
        # multiline descriptions should be converted to single line
        assert (
            format_description("Multiline\ndescription") == "\n// Multiline description"
        )
        assert format_description(
            "Multiline\ndescription with some\nmore lines\n"
            "than usual.\n\nAlso a paragraph."
        ) == (
            "\n// Multiline description with some more lines "
            "than usual. Also a paragraph."
        )


class TestParameterTypeFormatting:
    def test_string_type(self, string_prop):
        assert format_parameter_type(string_prop) == "string"

    def test_number_type(self, number_prop):
        assert format_parameter_type(number_prop) == "number"

    def test_bool_type(self, bool_prop):
        assert format_parameter_type(bool_prop) == "boolean"

    def test_null_type(self, null_prop):
        assert format_parameter_type(null_prop) == "null"

    def test_array_type(self, string_prop, number_prop, bool_prop, null_prop):
        assert format_parameter_type(ArrayProp(items=string_prop)) == "string[]"
        assert format_parameter_type(ArrayProp(items=number_prop)) == "number[]"
        assert format_parameter_type(ArrayProp(items=bool_prop)) == "boolean[]"
        assert format_parameter_type(ArrayProp(items=null_prop)) == "null[]"

    def test_object_type(self, function_schema):
        assert format_parameter_type(function_schema.parameters) == "object"

    def test_none_type(self):
        assert format_parameter_type(None) == "any"

    def test_custom_type(self):
        # some other type supplied
        assert format_parameter_type("string") == "any"  # type: ignore
        assert format_parameter_type(5) == "any"  # type: ignore

    def test_props_with_enum_formatting(
        self, string_prop_with_enum, number_prop_with_enum
    ):
        assert format_parameter_type(string_prop_with_enum) == '"A" | "B"'
        assert format_parameter_type(string_prop_with_enum) != '"B" | "A"'
        assert format_parameter_type(string_prop_with_enum) != "'A' | 'B'"
        assert format_parameter_type(string_prop_with_enum) != "A | B"
        assert format_parameter_type(number_prop_with_enum) == "0 | 1"
        assert format_parameter_type(number_prop_with_enum) != "1 | 0"
        assert format_parameter_type(number_prop_with_enum) != '"0" | "1"'
        assert format_parameter_type(number_prop_with_enum) != "'0' | '1'"


class TestParameterFormatting:
    def test_parameters(self, string_prop, number_prop, bool_prop, null_prop):
        assert format_parameter("name", string_prop) == "\n// A string.\nname: string;"
        assert format_parameter("name", number_prop) == "\n// A number.\nname: number;"
        assert format_parameter("name", bool_prop) == "\n// A boolean.\nname: boolean;"
        assert format_parameter("name", null_prop) == "\n// A null.\nname: null;"

    def test_array_parameters(self, string_prop, number_prop, bool_prop, null_prop):
        assert (
            format_parameter(
                "numbers", ArrayProp(description="Array of numbers.", items=number_prop)
            )
            == "\n// Array of numbers.\nnumbers: number[];"
        )
        assert (
            format_parameter(
                "strings", ArrayProp(description="Array of strings.", items=string_prop)
            )
            == "\n// Array of strings.\nstrings: string[];"
        )
        assert (
            format_parameter(
                "bools", ArrayProp(description="Array of bools.", items=bool_prop)
            )
            == "\n// Array of bools.\nbools: boolean[];"
        )
        assert (
            format_parameter(
                "nulls", ArrayProp(description="Array of nulls.", items=null_prop)
            )
            == "\n// Array of nulls.\nnulls: null[];"
        )

    def test_default_formatting(self):
        assert (
            format_parameter(
                "sigma",
                NumberProp(description="A number with default.", default=42),
            )
            == "\n// A number with default.\nsigma?: number;"
        )
        assert (
            format_parameter(
                "name",
                StringProp(description="A string with default.", default="unknown"),
            )
            == "\n// A string with default.\nname?: string;"
        )
        assert (
            format_parameter(
                "do_stuff",
                BooleanProp(description="A boolean with default.", default=True),
            )
            == "\n// A boolean with default.\ndo_stuff?: boolean;"
        )
        assert (
            format_parameter(
                "sigma",
                NumberProp(description="A number without default."),
            )
            == "\n// A number without default.\nsigma: number;"
        )
        assert (
            format_parameter(
                "name",
                StringProp(description="A string without default."),
            )
            == "\n// A string without default.\nname: string;"
        )


class TestFunctionFormatting:
    def test_function_formatting(self, foo_function, foo_expected):
        foo_formatted = format_function_as_typescript(get_schema(foo_function))
        assert foo_formatted == foo_expected


class TestSerialization:
    def test_function_serialization(self, foo_function, foo_json_expected):
        func_schema = get_schema(foo_function)
        func_json = func_schema.to_dict()
        assert func_json == foo_json_expected
