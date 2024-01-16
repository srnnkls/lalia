from lalia.io.serialization.json_schema import JsonSchemaType, StringProp


def test_string_prop_valid_data():
    prop = StringProp(
        maxLength=10,
        minLength=1,
        pattern=r"^[a-zA-Z]+$",
        enum=["abc", "def"],
    )
    assert prop.max_length == 10
    assert prop.min_length == 1
    assert prop.pattern == r"^[a-zA-Z]+$"
    assert prop.enum == ["abc", "def"]
    assert prop.type_ == JsonSchemaType.STRING
