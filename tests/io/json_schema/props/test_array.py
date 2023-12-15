from lalia.io.serialization.json_schema import (
    AnyOfProp,
    ArrayProp,
    IntegerProp,
    PropType,
)


def test_array_prop_valid_data():
    prop = ArrayProp(
        items=IntegerProp(),
    )
    assert isinstance(prop.items, IntegerProp)
    assert prop.type_ == PropType.ARRAY


def test_array_prop_invalid_nested_data():
    prop = ArrayProp(
        items=AnyOfProp([IntegerProp()]),
    )
    assert isinstance(prop.items, AnyOfProp)
    assert isinstance(prop.items.any_of, list)
    assert isinstance(prop.items.any_of[0], IntegerProp)
    assert prop.type_ == PropType.ARRAY
