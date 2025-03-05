import pytest
from pydantic import BaseModel, TypeAdapter

from lalia.io.parsers import LLMParser
from lalia.llm.openai import ChatModel, OpenAIChat


@pytest.fixture()
def type_():
    class A(BaseModel):
        b: str
        c: int

    return A


@pytest.fixture()
def llm(openai_api_key):
    return OpenAIChat(
        # model=ChatModel.GPT_4O,
        api_key=openai_api_key,
        temperature=0.0,
    )


@pytest.fixture()
def expected(type_):
    return type_(b="test", c=99)


@pytest.fixture()
def llm_parser(llm):
    return LLMParser(llms=[llm], max_retries=10)


@pytest.mark.openai
def test_llm_parser_valid_input(llm_parser, type_, expected):
    parsed_json, _ = llm_parser.parse('{"b": "test", "c": 99}', type=type_)
    assert parsed_json == expected
    parsed_yaml, _ = llm_parser.parse("b: test\nc: 99", type=type_)
    assert parsed_yaml == expected


@pytest.mark.openai
def test_llm_parser_missing_field(llm_parser, type_):
    parsed, _ = llm_parser.parse('{"b": "test"}', type=type_)
    assert "c" in parsed.model_fields


@pytest.mark.openai
def test_llm_parser_wrong_field(llm_parser, type_):
    parsed, _ = llm_parser.parse('{"a": "test", "c": 99}', type=type_)
    assert "b" in parsed.model_fields


@pytest.mark.openai
def test_llm_parser_list_field(llm_parser, type_):
    parsed, _ = llm_parser.parse('{"b": ["test"], "c": 99}', type=type_)
    assert "b" in parsed.model_fields
    assert parsed.b == "test"


@pytest.mark.openai
def test_llm_parser_dict_field(llm_parser, type_):
    payload_key_value = '{"b": {"test": "test"}, "c": 99}'
    parsed, _ = llm_parser.parse(payload_key_value, type=type_)
    assert "b" in parsed.model_fields
    assert isinstance(parsed.b, str)

    payload_key = '{"b": {"test": ""}, "c": 99}'
    parsed, _ = llm_parser.parse(payload_key, type=type_)
    assert "b" in parsed.model_fields
    assert isinstance(parsed.b, str)
