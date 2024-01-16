from lalia.io.serialization.json_schema import JsonSchemaType, NumberProp


def test_number_prop_valid_data():
    prop = NumberProp(maximum=100.5, minimum=10.1, title="Test Number")
    assert prop.maximum == 100.5
    assert prop.minimum == 10.1
    assert prop.title == "Test Number"
    assert prop.type_ == JsonSchemaType.NUMBER
