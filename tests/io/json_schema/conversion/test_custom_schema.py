from lalia.io.serialization.json_schema import (
    AllOfProp,
    AnyOfProp,
    ArrayProp,
    BooleanProp,
    IntegerProp,
    NumberProp,
    ObjectProp,
    StringProp,
)


def test_object_prop_initialization(class_schema):
    custom_schema = ObjectProp(**class_schema)
    assert isinstance(custom_schema, ObjectProp)
    assert hasattr(custom_schema, "properties")
    assert custom_schema.type_ == "object"


def test_individual_properties(class_schema):
    custom_schema = ObjectProp(**class_schema)

    # in line assetion to silence pyright
    assert (root := custom_schema.properties) is not None

    assert isinstance(root["i"], IntegerProp)
    assert isinstance(root["f"], NumberProp)
    assert isinstance(root["b"], BooleanProp)
    assert isinstance(root["fi"], NumberProp)
    assert isinstance(root["s"], StringProp)


def test_nested_properties_c(class_schema):
    custom_schema = ObjectProp(**class_schema)
    c_properties = custom_schema.properties["c"].properties  # type: ignore
    assert isinstance(c_properties["l"], ArrayProp)  # type: ignore
    assert isinstance(c_properties["d"], ObjectProp)  # type: ignore
    assert isinstance(c_properties["dv"], ObjectProp)  # type: ignore


def test_composite_properties(class_schema):
    custom_schema = ObjectProp(**class_schema)
    properties = custom_schema.properties
    assert isinstance(properties["e"], AllOfProp)  # type: ignore
    assert isinstance(properties["v"], AnyOfProp)  # type: ignore
    assert isinstance(properties["n"], AnyOfProp)  # type: ignore


def test_nested_items_composite_properties(class_schema):
    custom_schema = ObjectProp(**class_schema)
    c_properties = custom_schema.properties["c"].properties  # type: ignore
    dv_properties = c_properties["dv"].additional_properties  # type: ignore
    assert isinstance(dv_properties, AnyOfProp)
    assert all(
        isinstance(item, StringProp | IntegerProp) for item in dv_properties.any_of
    )


def test_array_items(class_schema):
    custom_schema = ObjectProp(**class_schema)
    c_properties = custom_schema.properties["c"].properties  # type: ignore
    l_items = c_properties["l"].items  # type: ignore
    assert isinstance(l_items, IntegerProp)
