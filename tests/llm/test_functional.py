from pydantic.dataclasses import dataclass

from lalia.chat.messages import UserMessage


def test_llm_call_unstructured(fake_llm):
    def prompt(query: str) -> list[UserMessage]:
        return [UserMessage(content=query)]

    @fake_llm.call(prompt=prompt)
    def query(query: str) -> str:
        """
        Asks the LLM for anything.
        """
        ...

    response = query("Can you tell me a joke?")
    assert isinstance(response, str)


def test_llm_call_structured(fake_llm):
    def recommend_book_prompt(genre: str) -> list[UserMessage]:
        return [
            UserMessage(content=f"Can you recommend me a book in the genre of {genre}?")
        ]

    @dataclass
    class Book:
        title: str
        author: str
        genre: str

    @fake_llm.call(prompt=recommend_book_prompt)
    def recommend_book(genre: str) -> Book:
        """
        Recommends a book in the given genre, keeping the state of the conversation.
        """
        ...

    response = recommend_book("post-modernism")
    assert isinstance(response, Book)
