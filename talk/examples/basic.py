from cobi.utils.auth.secrets import get_openai_token
from lalia.chat.session import Session
from lalia.llm.openai import ChatModel, OpenAIChat

llm = OpenAIChat(
    model=ChatModel.GPT_3_5_TURBO_0613,
    api_key=get_openai_token(),
)

session = Session(llm=llm, system_message="You are a vet.")
session("Is it wise to stroke a boar?")
