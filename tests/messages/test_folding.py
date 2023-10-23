from rich import print

from lalia.chat.messages import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from lalia.chat.messages.buffer import FoldState, MessageBuffer
from lalia.chat.messages.tags import Tag, TagPattern

Tag.register_key_color("user", "yellow")
Tag.register_key_color("assistant", "blue")
Tag.register_key_color("system", "red")
Tag.register_key_color("general", "magenta")

m = MessageBuffer(
    [
        SystemMessage(
            content="You are a pirate.",
            tags={Tag("system", "directive"), Tag("general", "first")},
        ),
        UserMessage(
            content="Hello there!",
            tags={Tag("user", "introduction"), Tag("general", "second")},
        ),
        AssistantMessage(
            content="Arrrgh, I am a pirate.",
            tags={Tag("assistant", "introduction"), Tag("general", "third")},
        ),
        UserMessage(
            content="Arrrrrrr, I am a pirate, too!",
            tags={Tag("user", "response"), Tag("general", "fourth")},
        ),
    ],
    default_fold_tags={TagPattern("system", ".*")},
)

m.add(AssistantMessage(content="Arrrrrgh, nice!"))

m.fold(TagPattern("user", ".*") | TagPattern("assistant", ".*"))
assert m.folds.messages == [
    FoldState.FOLDED,
    FoldState.FOLDED,
    FoldState.FOLDED,
    FoldState.FOLDED,
]

m.unfold(TagPattern("user", ".*"))
assert m.folds.messages == [
    FoldState.FOLDED,
    FoldState.UNFOLDED,
    FoldState.FOLDED,
    FoldState.UNFOLDED,
]

m.unfold(TagPattern("assistant", ".*"))
assert m.folds.messages == [
    FoldState.FOLDED,
    FoldState.UNFOLDED,
    FoldState.UNFOLDED,
    FoldState.UNFOLDED,
]

m.fold(TagPattern("user", ".*"))
assert m.folds.messages == [
    FoldState.FOLDED,
    FoldState.FOLDED,
    FoldState.UNFOLDED,
    FoldState.FOLDED,
]

m.fold(TagPattern("assistant", ".*"))
assert m.folds.messages == [
    FoldState.FOLDED,
    FoldState.FOLDED,
    FoldState.FOLDED,
    FoldState.FOLDED,
]

m.unfold(TagPattern("assistant", "introduction"))
assert m.folds.messages == [
    FoldState.FOLDED,
    FoldState.FOLDED,
    FoldState.UNFOLDED,
    FoldState.FOLDED,
]

m.fold(TagPattern("assistant", "intro*"))
assert m.folds.messages == [
    FoldState.FOLDED,
    FoldState.FOLDED,
    FoldState.FOLDED,
    FoldState.FOLDED,
]

m.unfold(TagPattern("us*", ".*"))
assert m.folds.messages == [
    FoldState.FOLDED,
    FoldState.UNFOLDED,
    FoldState.FOLDED,
    FoldState.UNFOLDED,
]

m.unfold(TagPattern("sys*", ".*"))
assert m.folds.messages == [
    FoldState.UNFOLDED,
    FoldState.UNFOLDED,
    FoldState.FOLDED,
    FoldState.UNFOLDED,
]
