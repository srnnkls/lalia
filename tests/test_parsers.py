from pydantic import BaseModel

from cobi.utils.auth.secrets import get_openai_token
from lalia.io.parsers import LLMParser
from lalia.llm.openai import ChatModel, OpenAIChat

llm = OpenAIChat(
    model=ChatModel.GPT_3_5_TURBO_0613,
    api_key=get_openai_token(),
    temperature=0.0,
)


class A(BaseModel):
    b: str
    c: int


parser = LLMParser(llms=[llm], max_retries=10)

parser.parse('{"b": "test", "c": 99}', model=A)
parser.parse("b: test\nc: 99", model=A)
parser.parse('{"b": "test"}', model=A)
parser.parse('{"a": "test", "c": 99}', model=A)
parser.parse('{"b": ["test"], "c": 99}', model=A)
parser.parse('{"b": {"test": "test"}, "c": 99}', model=A)
parser.parse('{"b": {"test": ""}, "c": 99}', model=A)
