import pytest
from pydantic import BaseModel, TypeAdapter

from lalia.io.parsers import LLMParser
from lalia.llm.openai import ChatModel, OpenAIChat


@pytest.fixture()
def adapter():
    class A(BaseModel):
        b: str
        c: int

    return TypeAdapter(A)


@pytest.fixture()
def llm(openai_api_key):
    return OpenAIChat(
        model=ChatModel.GPT_3_5_TURBO_0613,
        api_key=openai_api_key,
        temperature=0.0,
    )


@pytest.fixture()
def expected():
    return {"b": "test", "c": 99}


@pytest.fixture()
def llm_parser(llm):
    return LLMParser(llms=[llm], max_retries=10)


@pytest.mark.openai
def test_llm_parser_valid_input(llm_parser, adapter, expected):
    parsed_json, _ = llm_parser.parse('{"b": "test", "c": 99}', adapter=adapter)
    assert parsed_json == expected
    parsed_yaml, _ = llm_parser.parse("b: test\nc: 99", adapter=adapter)
    assert parsed_yaml == expected


@pytest.mark.openai
def test_llm_parser_missing_field(llm_parser, adapter):
    parsed, _ = llm_parser.parse('{"b": "test"}', adapter=adapter)
    assert "c" in parsed


@pytest.mark.openai
def test_llm_parser_wrong_field(llm_parser, adapter):
    parsed, _ = llm_parser.parse('{"a": "test", "c": 99}', adapter=adapter)
    assert "b" in parsed


@pytest.mark.openai
def test_llm_parser_list_field(llm_parser, adapter):
    parsed, _ = llm_parser.parse('{"b": ["test"], "c": 99}', adapter=adapter)
    assert "b" in parsed
    assert parsed["b"] == "test"


@pytest.mark.openai
def test_llm_parser_dict_field(llm_parser, adapter):
    payload_key_value = '{"b": {"test": "test"}, "c": 99}'
    parsed, _ = llm_parser.parse(payload_key_value, adapter=adapter)
    assert "b" in parsed
    assert isinstance(parsed["b"], str)
    assert parsed["b"] == "test"

    payload_key = '{"b": {"test": ""}, "c": 99}'
    parsed, _ = llm_parser.parse(payload_key, adapter=adapter)
    assert "b" in parsed
    assert isinstance(parsed["b"], str)
