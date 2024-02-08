from inspect import cleandoc

import pytest

from lalia.formatting import OpenAIFunctionFormatter
from lalia.functions import get_schema
from lalia.io.serialization.json_schema import (
    ArrayProp,
    BooleanProp,
    JsonSchemaType,
    NotProp,
    NullProp,
    NumberProp,
    StringProp,
)


@pytest.fixture()
def formatter():
    return OpenAIFunctionFormatter()


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
def not_prop():
    return NotProp(not_=StringProp(), description="A not.")


@pytest.fixture()
def foo_expected():
    return cleandoc(
        """
// Tools

// Functions

namespace functions {

// This is a test function.
// It combines number a, appends string b and option c.
type foo = (_: {
// This is a integer number.
a: number,
// This will be appended.
b?: string | number, // default: test
// This is an Enum.
c?: "option1" | "option2", // default: option1
}) => any;

} // namespace functions
"""
    )


@pytest.fixture()
def baz_expected():
    return cleandoc(
        """
// Tools

// Functions

namespace functions {

// This is a test function. It has no attributes and just returns a string.
type baz = () => any;

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
    def test_description_formatting(self, formatter):
        assert (
            formatter._format_description("Test description") == "// Test description"
        )
        # multiline descriptions should be converted to single line
        assert (
            formatter._format_description("Multiline\ndescription")
            == "// Multiline\n// description"
        )
        assert formatter._format_description(
            "Multiline\ndescription with some\nmore lines\n"
            "than usual.\n\nAlso a paragraph."
        ) == (
            "// Multiline\n// description with some\n// more lines\n"
            "// than usual.\n// \n// Also a paragraph."
        )


class TestParameterTypeFormatting:
    def test_string_type(self, string_prop, formatter):
        assert formatter._format_parameter_type(string_prop) == "string"

    def test_number_type(self, number_prop, formatter):
        assert formatter._format_parameter_type(number_prop) == "number"

    def test_bool_type(self, bool_prop, formatter):
        assert formatter._format_parameter_type(bool_prop) == "boolean"

    def test_null_type(self, null_prop, formatter):
        assert formatter._format_parameter_type(null_prop) == "null"

    def test_array_type(
        self, string_prop, number_prop, bool_prop, null_prop, formatter
    ):
        assert (
            formatter._format_parameter_type(ArrayProp(items=string_prop)) == "string[]"
        )
        assert (
            formatter._format_parameter_type(ArrayProp(items=number_prop)) == "number[]"
        )
        assert (
            formatter._format_parameter_type(ArrayProp(items=bool_prop)) == "boolean[]"
        )
        assert formatter._format_parameter_type(ArrayProp(items=null_prop)) == "null[]"

    def test_object_type(self, function_schema, formatter):
        assert formatter._format_parameter_type(function_schema.parameters) == "object"

    def test_none_type(self, formatter):
        assert formatter._format_parameter_type(None) == "any"

    def test_not_type(self, formatter):
        assert formatter._format_parameter_type(NotProp(not_=StringProp())) == "any"

    def test_custom_type(self, formatter):
        # some other type supplied
        assert formatter._format_parameter_type("string") == "any"  # type: ignore
        assert formatter._format_parameter_type(5) == "any"  # type: ignore

    def test_props_with_enum_formatting(
        self, string_prop_with_enum, number_prop_with_enum, formatter
    ):
        assert formatter._format_parameter_type(string_prop_with_enum) == '"A" | "B"'
        assert formatter._format_parameter_type(string_prop_with_enum) != '"B" | "A"'
        assert formatter._format_parameter_type(string_prop_with_enum) != "'A' | 'B'"
        assert formatter._format_parameter_type(string_prop_with_enum) != "A | B"
        assert formatter._format_parameter_type(number_prop_with_enum) == "0 | 1"
        assert formatter._format_parameter_type(number_prop_with_enum) != "1 | 0"
        assert formatter._format_parameter_type(number_prop_with_enum) != '"0" | "1"'
        assert formatter._format_parameter_type(number_prop_with_enum) != "'0' | '1'"


class TestParameterFormatting:
    def test_parameters(
        self, string_prop, number_prop, bool_prop, null_prop, formatter
    ):
        assert (
            formatter._format_parameter("name", string_prop)
            == "// A string.\nname: string,"
        )
        assert (
            formatter._format_parameter("name", number_prop)
            == "// A number.\nname: number,"
        )
        assert (
            formatter._format_parameter("name", bool_prop)
            == "// A boolean.\nname: boolean,"
        )
        assert (
            formatter._format_parameter("name", null_prop) == "// A null.\nname: null,"
        )

    def test_array_parameters(
        self, string_prop, number_prop, bool_prop, null_prop, formatter
    ):
        assert (
            formatter._format_parameter(
                "numbers", ArrayProp(description="Array of numbers.", items=number_prop)
            )
            == "// Array of numbers.\nnumbers: number[],"
        )
        assert (
            formatter._format_parameter(
                "strings", ArrayProp(description="Array of strings.", items=string_prop)
            )
            == "// Array of strings.\nstrings: string[],"
        )
        assert (
            formatter._format_parameter(
                "bools", ArrayProp(description="Array of bools.", items=bool_prop)
            )
            == "// Array of bools.\nbools: boolean[],"
        )
        assert (
            formatter._format_parameter(
                "nulls", ArrayProp(description="Array of nulls.", items=null_prop)
            )
            == "// Array of nulls.\nnulls: null[],"
        )

    def test_default_formatting(self, formatter):
        assert (
            formatter._format_parameter(
                "sigma",
                NumberProp(description="A number with default.", default=42),
            )
            == "// A number with default.\nsigma?: number, // default: 42.0"
        )
        assert (
            formatter._format_parameter(
                "name",
                StringProp(description="A string with default.", default="unknown"),
            )
            == "// A string with default.\nname?: string, // default: unknown"
        )
        assert (
            formatter._format_parameter(
                "do_stuff",
                BooleanProp(description="A boolean with default.", default=True),
            )
            == "// A boolean with default.\ndo_stuff?: boolean, // default: true"
        )
        assert (
            formatter._format_parameter(
                "sigma",
                NumberProp(description="A number without default."),
            )
            == "// A number without default.\nsigma: number,"
        )
        assert (
            formatter._format_parameter(
                "name",
                StringProp(description="A string without default."),
            )
            == "// A string without default.\nname: string,"
        )


class TestFunctionFormatting:
    def test_internal_function_formatting(
        self, foo_function, foo_expected, baz_function, baz_expected, formatter
    ):
        foo_formatted = formatter._format_functions_as_typescript_namespace(
            [get_schema(foo_function)]
        )
        assert foo_formatted == foo_expected

        baz_formatted = formatter._format_functions_as_typescript_namespace(
            [get_schema(baz_function)]
        )
        assert baz_formatted == baz_expected

    def test_public_format_method(
        self, foo_function, foo_expected, baz_function, baz_expected, formatter
    ):
        foo_formatted = formatter.format([get_schema(foo_function)])
        assert foo_formatted == foo_expected

        baz_formatted = formatter.format([get_schema(baz_function)])
        assert baz_formatted == baz_expected


class TestSerialization:
    def test_function_serialization(self, foo_function, foo_json_expected):
        func_schema = get_schema(foo_function)
        func_json = func_schema.to_dict()
        assert func_json == foo_json_expected
