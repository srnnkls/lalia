def test_general_structure(class_schema):
    assert isinstance(class_schema, dict)
    assert "properties" in class_schema
    assert "type" in class_schema and class_schema["type"] == "object"


def test_property_presence(class_schema):
    properties = class_schema["properties"]
    for prop in ["i", "f", "fi", "s", "b", "e", "v", "n"]:
        assert prop in properties


def test_nested_structure_b(class_schema):
    b_properties = class_schema["properties"]["c"]["properties"]
    assert "l" in b_properties
    assert "d" in b_properties
    assert "dv" in b_properties


def test_nested_types_b(class_schema):
    b_properties = class_schema["properties"]["c"]["properties"]
    assert b_properties["l"]["type"] == "array"
    assert b_properties["d"]["type"] == "object"
    assert b_properties["dv"]["type"] == "object"
