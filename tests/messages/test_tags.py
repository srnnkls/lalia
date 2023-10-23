from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import field

from pydantic import validate_call
from pydantic.dataclasses import dataclass
from rich import print as pprint

from lalia.chat.messages.tags import Tag, TagPattern, predicate_registry


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
                tag_predicate = predicate_registry.derive_predicate(tags)
            case dict():
                tag_pattern = TagPattern.from_dict(tags)
                tag_predicate = predicate_registry.derive_predicate(tag_pattern)
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


m = LikeMessageBuffer(
    messages=[
        LikeMessage(
            content="first",
            tags={Tag(key="a", value="1"), Tag(key="b", value="2")},
        ),
        LikeMessage(content="second", tags={Tag(key="b", value="2")}),
        LikeMessage(content="third", tags={Tag(key="c", value="3")}),
    ]
)

pprint(m.filter(tags=Tag(key="a", value="1") & Tag(key="b", value="2")))
pprint(
    m.filter(
        tags=(Tag(key="a", value="1") & Tag(key="b", value="2"))
        | Tag(key="c", value="3")
    )
)
pprint(m.filter(tags=Tag(key="a", value="1")))
pprint(
    m.filter(
        tags=TagPattern(
            key=re.compile(".*"),
            value=re.compile(".*"),
        )
    )
)
pprint(
    m.filter(
        tags={".*": ".*"},
    )
)
pprint(
    m.filter(
        lambda message: Tag(key="a", value="1") in message.tags
        and Tag(key="b", value="2") in message.tags
    )
)
pprint(
    m.filter(
        lambda message: {Tag(key="a", value="1"), Tag(key="b", value="2")}
        <= message.tags
    )
)
pprint(m.filter(predicate=lambda message: message.content == "first"))
