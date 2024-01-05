from inspect import cleandoc

import pytest

from lalia.formatting import (
    format_description,
    format_function_as_typescript,
    format_parameter,
    format_parameter_type,
    format_return_type,
)
from lalia.functions import get_schema
from lalia.functions.types import (
    ArrayProp,
    BoolProp,
    NullProp,
    NumberProp,
    ReturnType,
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
    return BoolProp(description="A boolean.")


@pytest.fixture()
def null_prop():
    return NullProp(description="A null.")


@pytest.fixture()
def return_int():
    return ReturnType(type=int)


@pytest.fixture()
def return_int_or_str():
    return ReturnType(type=int | str)


@pytest.fixture()
def return_str_or_int():
    return ReturnType(type=str | int)


@pytest.fixture()
def return_custom_type():
    return ReturnType(type="MyCustomType")


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
            "required": ["a"],
            "properties": {
                "a": {"description": "This is a integer number.", "type": "number"},
                "b": {
                    "description": "This will be appended.",
                    "default": "test",
                    "anyof": [
                        {
                            "description": "This will be appended.",
                            "default": "test",
                            "type": "string",
                        },
                        {
                            "description": "This will be appended.",
                            "default": "test",
                            "type": "number",
                        },
                    ],
                },
                "c": {
                    "description": "This is an Enum.",
                    "default": "option1",
                    "enum": ["option1", "option2"],
                    "type": "string",
                },
            },
            "type": "object",
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
        assert format_description("Test description") == "// Test description"
        # multiline descriptions should be converted to single line
        assert (
            format_description("Multiline\ndescription") == "// Multiline description"
        )
        assert format_description(
            "Multiline\ndescription with some\nmore lines\n"
            "than usual.\n\nAlso a paragraph."
        ) == (
            "// Multiline description with some more lines "
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


class TestReturnTypeFormatting:
    def test_number_type(self, return_int):
        assert format_return_type(return_int) == "number"

    def test_custom_type(self, return_custom_type):
        assert format_return_type(return_custom_type) == "any"

    def test_union_type(self, return_int_or_str, return_str_or_int):
        assert format_return_type(return_str_or_int) == "string | number"
        assert format_return_type(return_str_or_int) != "number | string"
        assert format_return_type(return_int_or_str) != "string | number"
        assert format_return_type(return_int_or_str) == "number | string"


class TestParameterFormatting:
    def test_parameters(self, string_prop, number_prop, bool_prop, null_prop):
        assert format_parameter("name", string_prop) == "// A string.\nname: string;"
        assert format_parameter("name", number_prop) == "// A number.\nname: number;"
        assert format_parameter("name", bool_prop) == "// A boolean.\nname: boolean;"
        assert format_parameter("name", null_prop) == "// A null.\nname: null;"

    def test_array_parameters(self, string_prop, number_prop, bool_prop, null_prop):
        assert (
            format_parameter(
                "numbers", ArrayProp(description="Array of numbers.", items=number_prop)
            )
            == "// Array of numbers.\nnumbers: number[];"
        )
        assert (
            format_parameter(
                "strings", ArrayProp(description="Array of strings.", items=string_prop)
            )
            == "// Array of strings.\nstrings: string[];"
        )
        assert (
            format_parameter(
                "bools", ArrayProp(description="Array of bools.", items=bool_prop)
            )
            == "// Array of bools.\nbools: boolean[];"
        )
        assert (
            format_parameter(
                "nulls", ArrayProp(description="Array of nulls.", items=null_prop)
            )
            == "// Array of nulls.\nnulls: null[];"
        )

    def test_default_formatting(self):
        assert (
            format_parameter(
                "sigma",
                NumberProp(description="A number with default.", default=42),
            )
            == "// A number with default.\nsigma?: number;"
        )
        assert (
            format_parameter(
                "name",
                StringProp(description="A string with default.", default="unknown"),
            )
            == "// A string with default.\nname?: string;"
        )
        assert (
            format_parameter(
                "do_stuff",
                BoolProp(description="A boolean with default.", default=True),
            )
            == "// A boolean with default.\ndo_stuff?: boolean;"
        )
        assert (
            format_parameter(
                "sigma",
                NumberProp(description="A number without default."),
            )
            == "// A number without default.\nsigma: number;"
        )
        assert (
            format_parameter(
                "name",
                StringProp(description="A string without default."),
            )
            == "// A string without default.\nname: string;"
        )


class TestFunctionFormatting:
    def test_function_formatting(self, foo_function, foo_expected):
        foo_formatted = format_function_as_typescript(get_schema(foo_function))
        assert foo_formatted == foo_expected


class TestSerialization:
    def test_function_serialization(self, foo_function, foo_json_expected):
        func_schema = get_schema(foo_function)
        func_json = func_schema.to_json_schema()
        assert func_json == foo_json_expected

    def test_string_prop_serialization(self, string_prop):
        assert string_prop.to_json_schema() == {
            "description": "Represents a property of type string in a function signature.",
            "properties": {
                "description": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Description",
                },
                "default": {
                    "anyOf": [{}, {"type": "null"}],
                    "default": None,
                    "title": "Default",
                },
                "enum": {
                    "anyOf": [
                        {"items": {"type": "string"}, "type": "array"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Enum",
                },
            },
            "title": "StringProp",
            "type": "object",
        }

    def test_number_prop_serialization(self, number_prop):
        assert number_prop.to_json_schema() == {
            "description": "Represents a property of type number in a function signature.",
            "properties": {
                "description": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "title": "Description",
                },
                "default": {
                    "anyOf": [{}, {"type": "null"}],
                    "default": None,
                    "title": "Default",
                },
                "minimum": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                    "title": "Minimum",
                },
                "maximum": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "default": None,
                    "title": "Maximum",
                },
                "enum": {
                    "anyOf": [
                        {"items": {"type": "integer"}, "type": "array"},
                        {"items": {"type": "number"}, "type": "array"},
                        {"type": "null"},
                    ],
                    "default": None,
                    "title": "Enum",
                },
            },
            "title": "NumberProp",
            "type": "object",
        }
