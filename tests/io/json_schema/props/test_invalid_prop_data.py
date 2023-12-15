import pytest
from pydantic import ValidationError
from pytest import raises

from lalia.io.serialization.json_schema import (
    IntegerProp,
    NumberProp,
    StringProp,
)


@pytest.mark.parametrize(
    "prop_class,args,expected_exception",
    [
        (IntegerProp, {"maximum": -1}, ValidationError),
        (IntegerProp, {"minimum": -1}, ValidationError),
        (NumberProp, {"maximum": -1.0}, ValidationError),
        (NumberProp, {"minimum": -1.0}, ValidationError),
        (StringProp, {"maxLength": -1}, ValidationError),
        (StringProp, {"minLength": -1}, ValidationError),
        (StringProp, {"enum": [123, 456]}, ValidationError),
        (StringProp, {"type": "number"}, ValidationError),
    ],
)
def test_invalid_data(prop_class, args, expected_exception):
    with raises(expected_exception):
        prop_class(**args)
