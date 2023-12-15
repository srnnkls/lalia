from lalia.io.serialization.json_schema import ObjectProp, PropType


def test_object_prop_valid_data():
    prop = ObjectProp()
    assert prop.type_ == PropType.OBJECT
