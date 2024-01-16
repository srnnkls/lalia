from lalia.io.serialization.json_schema import IntegerProp, JsonSchemaType


def test_integer_prop_valid_data():
    prop = IntegerProp(maximum=100, minimum=10, title="Test Integer")
    assert prop.maximum == 100
    assert prop.minimum == 10
    assert prop.title == "Test Integer"
    assert prop.type_ == JsonSchemaType.INTEGER
