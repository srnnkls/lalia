from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import field

import pytest
from pydantic.dataclasses import dataclass
from rich import print as pprint

from lalia.chat.messages.tags import PredicateRegistry, Tag, TagPattern, _And, _Or


@dataclass(frozen=True)
class LikeMessage:
    content: str = ""
    tags: set[Tag] = field(default_factory=set)


@dataclass
class LikeMessageBuffer:
    messages: list[LikeMessage] = field(default_factory=list)

    def filter(
        self,
        predicate: Callable[[LikeMessage], bool] = lambda message: True,
        tags: Tag
        | TagPattern
        | Callable[[set[Tag]], bool]
        | dict[str, str] = lambda tags: True,
    ) -> LikeMessageBuffer:
        match tags:
            case Tag() | TagPattern():
                tag_predicate = PredicateRegistry.derive_predicate(tags)
            case dict():
                tag_pattern = TagPattern.from_dict(tags)
                tag_predicate = PredicateRegistry.derive_predicate(tag_pattern)
            case Callable():
                tag_predicate = tags
            case _:
                raise TypeError(f"Unsupported type for tags: '{type(tags).__name__}'")

        return LikeMessageBuffer(
            [
                message
                for message in self.messages
                if tag_predicate(message.tags) and predicate(message)
            ]
        )


@pytest.fixture
def messages_with_tags():
    return LikeMessageBuffer(
        messages=[
            LikeMessage(
                content="first",
                tags={Tag(key="a", value="1"), Tag(key="b", value="2")},
            ),
            LikeMessage(content="second", tags={Tag(key="b", value="2")}),
            LikeMessage(content="third", tags={Tag(key="c", value="3")}),
        ]
    )


def test_tags_operators():
    and_tags = Tag(key="a", value="1") & Tag(key="b", value="2")
    assert isinstance(and_tags, _And)
    assert and_tags.predicates == (
        PredicateRegistry.derive_predicate(Tag(key="a", value="1")),
        PredicateRegistry.derive_predicate(Tag(key="b", value="2")),
    )
    assert and_tags({Tag(key="a", value="1"), Tag(key="b", value="2")})
    assert not and_tags({Tag(key="a", value="2"), Tag(key="b", value="2")})
    or_tags = Tag(key="a", value="1") | Tag(key="b", value="2")
    assert isinstance(or_tags, _Or)
    assert or_tags.predicates == (
        PredicateRegistry.derive_predicate(Tag(key="a", value="1")),
        PredicateRegistry.derive_predicate(Tag(key="b", value="2")),
    )
    assert or_tags({Tag(key="a", value="1"), Tag(key="b", value="2")})
    assert not or_tags({Tag(key="a", value="2")})


def test_tags_filter(messages_with_tags):
    m = messages_with_tags
    assert (
        m.filter(tags=Tag(key="a", value="1")).messages
        == messages_with_tags.messages[:1]
    )
    assert (
        m.filter(tags=Tag(key="b", value="2")).messages
        == messages_with_tags.messages[:2]
    )
    assert (
        m.filter(tags=Tag(key="c", value="3")).messages
        == messages_with_tags.messages[2:]
    )
    assert m.filter(tags=Tag("d", "3")).messages == []


def test_tags_operators_filter(messages_with_tags):
    m = messages_with_tags
    assert (
        m.filter(tags=Tag(key="a", value="1") & Tag(key="b", value="2")).messages
        == messages_with_tags.messages[:1]
    )
