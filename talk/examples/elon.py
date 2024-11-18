import os

from lalia.chat.messages import SystemMessage
from lalia.chat.session import Session
from lalia.llm.openai import ChatModel, OpenAIChat

llm = OpenAIChat(
    model=ChatModel.GPT_4O,
    api_key=os.environ.get("OPENAI_API_KEY"),
)

elon = Session(
    llm=llm, system_message="You are Elon Musk, introduce yourself through emojis."
)

elon()
elon("You have been rejected by Berghain.")
elon.autocommit = False
elon.messages.add(SystemMessage("Stop using emojis."))
elon("What is your favorite car?")
elon.rollback()
elon("What is your favorite rocket?")
elon.commit()
elon.revert()
elon.reset()
