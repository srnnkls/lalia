import pytest

from lalia.functions import FunctionSchema, get_callable, get_name, get_schema
from lalia.io.serialization.json_schema import (
    AllOfProp,
    AnyOfProp,
    IntegerProp,
    ObjectProp,
    StringProp,
)


class Adder:
    def __init__(self, value):
        self.value = value

    def __call__(self, x):
        return self.value + x


@pytest.fixture()
def foo_schema_expected():
    return FunctionSchema(
        name="foo",
        parameters=ObjectProp(
            properties={
                "a": IntegerProp(
                    description="This is a integer number.",
                    title="A",
                ),
                "b": AnyOfProp(
                    any_of=[StringProp(), IntegerProp()],
                    description="This will be appended.",
                    default="test",
                    title="B",
                ),
                "c": AllOfProp(
                    all_of=[
                        StringProp(
                            title="MyEnum",
                            enum=["option1", "option2"],
                        )
                    ],
                    description="This is an Enum.",
                    default="option1",
                ),
            },
            additional_properties=False,
            required=["a"],
        ),
        description="This is a test function.\nIt combines number a, appends string b and option c.",
    )


def test_get_name(foo_function):
    assert get_name(foo_function) == "foo"


def test_get_callable(foo_function):
    result = get_callable(foo_function)
    assert result is foo_function

    instance = Adder(5)
    result = get_callable(instance)
    assert result == instance.__call__

    # TODO: handle this case
    non_callable = "I am not callable"
    assert non_callable == get_callable(non_callable)


def test_get_schema(foo_function, foo_schema_expected):
    func_schema = get_schema(foo_function)
    assert func_schema == foo_schema_expected

    non_callable = "I am not callable"
    with pytest.raises(ValueError):
        get_schema(non_callable)
