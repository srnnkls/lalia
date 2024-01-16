from lalia.io.serialization.json_schema import JsonSchemaType, ObjectProp


def test_object_prop_valid_data():
    prop = ObjectProp()
    assert prop.type_ == JsonSchemaType.OBJECT
