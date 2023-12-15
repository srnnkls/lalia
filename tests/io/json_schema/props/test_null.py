from lalia.io.serialization.json_schema import NullProp, PropType


def test_null_prop_valid_data():
    prop = NullProp()
    assert prop.type_ == PropType.NULL
