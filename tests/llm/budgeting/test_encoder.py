import pytest
from pydantic import ValidationError

from lalia.llm.budgeting.budgeter import Encoder


def test_encoder_initialization(encoder):
    assert encoder is not None
    assert encoder.encoding_name == "cl100k_base"
    assert encoder.encoder.name == "cl100k_base"


def test_invalid_encoder_initialization():
    with pytest.raises(ValidationError):
        # assuming 'non existent encoding' is not a valid model in tiktoken
        Encoder("non existent encoding")


def test_encode_decode(encoder, string_fixture):
    assert encoder.decode(encoder.encode(string_fixture)) == string_fixture

    encoded = encoder.encode(string_fixture)
    assert isinstance(encoded, list) and all(isinstance(item, int) for item in encoded)

    decoded = encoder.decode(encoded)
    assert isinstance(decoded, str)
    assert decoded == string_fixture


def test_invalid_encode_input(encoder):
    with pytest.raises(ValueError):
        encoder.encode(123)  # intentionally wrong type


def test_invalid_decode_input(encoder):
    with pytest.raises(ValueError):
        encoder.decode("invalid input")  # intentionally wrong type


def test_encode_decode_empty_string(encoder):
    assert encoder.encode("") == []
    assert encoder.decode([]) == ""


def test_encode_decode_long_string(encoder):
    long_string = "hello world " * 1000  # a very long string
    encoded = encoder.encode(long_string)
    assert isinstance(encoded, list)
    assert encoder.decode(encoded) == long_string


def test_encode_decode_special_characters(encoder):
    special_string = "特殊字符, emojis 😊, and more!"
    assert encoder.decode(encoder.encode(special_string)) == special_string


def test_encoding_consistency(encoder, string_fixture):
    encoded_first = encoder.encode(string_fixture)
    encoded_second = encoder.encode(string_fixture)
    assert encoded_first == encoded_second


def test_unsupported_encoder(encoder):
    with pytest.raises(ValueError):
        new_encoder = Encoder.from_model("text-davinci-003")
        assert new_encoder.encoder.name != encoder.encoder.name


def test_invalid_model_in_from_model_method():
    with pytest.raises(ValueError):
        encoder = Encoder()
        encoder.from_model("non existent model")
