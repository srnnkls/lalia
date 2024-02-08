import pytest
from pydantic import ValidationError
from pytest import raises

from lalia.io.serialization.json_schema import (
    AllOfProp,
    AnyOfProp,
    ArrayProp,
    BooleanProp,
    IntegerProp,
    NullProp,
    NumberProp,
    ObjectProp,
    OneOfProp,
    StringProp,
)


@pytest.mark.parametrize(
    "prop_class,args,expected_exception",
    [
        # IntegerProp
        (IntegerProp, {"maximum": "not an int"}, ValidationError),
        (IntegerProp, {"minimum": "not an int"}, ValidationError),
        (IntegerProp, {"exclusive_maximum": "not an int"}, ValidationError),
        (IntegerProp, {"exclusive_minimum": "not an int"}, ValidationError),
        (IntegerProp, {"multiple_of": "not an int"}, ValidationError),
        (IntegerProp, {"enum": ["a", "b"]}, ValidationError),
        (IntegerProp, {"default": "not an int"}, ValidationError),
        # NumberProp
        (NumberProp, {"maximum": "not a number"}, ValidationError),
        (NumberProp, {"minimum": "not a number"}, ValidationError),
        (NumberProp, {"exclusive_maximum": "not a number"}, ValidationError),
        (NumberProp, {"exclusive_minimum": "not a number"}, ValidationError),
        (NumberProp, {"multiple_of": "not a number"}, ValidationError),
        (NumberProp, {"enum": ["a", "b"]}, ValidationError),
        (NumberProp, {"default": "not a number"}, ValidationError),
        # StringProp
        (StringProp, {"max_length": -1}, ValidationError),
        (StringProp, {"min_length": -2}, ValidationError),
        (StringProp, {"pattern": 123}, ValidationError),
        (StringProp, {"format": 123}, ValidationError),
        (StringProp, {"enum": [123, 456]}, ValidationError),
        (StringProp, {"default": 123}, ValidationError),
        # BooleanProp
        (BooleanProp, {"default": "not a bool"}, ValidationError),
        # ArrayProp
        (ArrayProp, {"min_contains": "not an int"}, ValidationError),
        (ArrayProp, {"max_contains": "not an int"}, ValidationError),
        (ArrayProp, {"min_items": -3}, ValidationError),
        (ArrayProp, {"max_items": -4}, ValidationError),
        # ObjectProp
        (ObjectProp, {"properties": "not a dict"}, ValidationError),
        (ObjectProp, {"pattern_properties": "not a dict"}, ValidationError),
        (ObjectProp, {"required": "not a list"}, ValidationError),
        (ObjectProp, {"min_properties": -1}, ValidationError),
        (ObjectProp, {"max_properties": -2}, ValidationError),
        (ObjectProp, {"additional_properties": "not a bool or Prop"}, ValidationError),
        # OneOfProp, AnyOfProp, AllOfProp
        (OneOfProp, {"one_of": "not a list"}, ValidationError),
        (AnyOfProp, {"any_of": "not a list"}, ValidationError),
        (AllOfProp, {"all_of": "not a list"}, ValidationError),
    ],
)
def test_invalid_data(prop_class, args, expected_exception):
    with raises(expected_exception):
        prop_class(**args)
