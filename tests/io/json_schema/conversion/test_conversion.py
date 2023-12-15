from lalia.io.serialization.json_schema import ObjectProp


def test_class_schema_comparison(class_schema, expected_class_schema_object):
    assert class_schema == expected_class_schema_object


def test_custom_schema_comparison(class_schema, expected_custom_schema_object):
    custom_schema = ObjectProp(**class_schema)
    assert custom_schema == expected_custom_schema_object


def test_schema_instance_comparison(schema_instance, expected_schema_instance):
    assert schema_instance == expected_schema_instance
