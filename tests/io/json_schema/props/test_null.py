from lalia.io.serialization.json_schema import JsonSchemaType, NullProp


def test_null_prop_valid_data():
    prop = NullProp()
    assert prop.type_ == JsonSchemaType.NULL
