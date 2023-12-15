from lalia.io.serialization.json_schema import BooleanProp, PropType


def test_boolean_prop_valid_data():
    prop = BooleanProp(title="Test Boolean")
    assert prop.title == "Test Boolean"
    assert prop.type_ == PropType.BOOLEAN
