from lalia.chat.messages.fold_state import FoldState
from lalia.chat.messages.tags import TagPattern


def test_fold_unfold_regex(tagged_messages):
    m = tagged_messages

    m.fold(TagPattern("user", ".*") | TagPattern("assistant", ".*"))
    assert m.folds.message_states == [
        FoldState.FOLDED,
        FoldState.FOLDED,
        FoldState.FOLDED,
        FoldState.FOLDED,
    ]

    m.unfold(TagPattern("user", ".*"))
    assert m.folds.message_states == [
        FoldState.FOLDED,
        FoldState.UNFOLDED,
        FoldState.FOLDED,
        FoldState.UNFOLDED,
    ]

    m.unfold(TagPattern("assistant", ".*"))
    assert m.folds.message_states == [
        FoldState.FOLDED,
        FoldState.UNFOLDED,
        FoldState.UNFOLDED,
        FoldState.UNFOLDED,
    ]

    m.fold(TagPattern("user", ".*"))
    assert m.folds.message_states == [
        FoldState.FOLDED,
        FoldState.FOLDED,
        FoldState.UNFOLDED,
        FoldState.FOLDED,
    ]

    m.fold(TagPattern("assistant", ".*"))
    assert m.folds.message_states == [
        FoldState.FOLDED,
        FoldState.FOLDED,
        FoldState.FOLDED,
        FoldState.FOLDED,
    ]

    m.unfold(TagPattern("assistant", "introduction"))
    assert m.folds.message_states == [
        FoldState.FOLDED,
        FoldState.FOLDED,
        FoldState.UNFOLDED,
        FoldState.FOLDED,
    ]

    m.fold(TagPattern("assistant", "intro*"))
    assert m.folds.message_states == [
        FoldState.FOLDED,
        FoldState.FOLDED,
        FoldState.FOLDED,
        FoldState.FOLDED,
    ]

    m.unfold(TagPattern("us*", ".*"))
    assert m.folds.message_states == [
        FoldState.FOLDED,
        FoldState.UNFOLDED,
        FoldState.FOLDED,
        FoldState.UNFOLDED,
    ]

    m.unfold(TagPattern("sys*", ".*"))
    assert m.folds.message_states == [
        FoldState.UNFOLDED,
        FoldState.UNFOLDED,
        FoldState.FOLDED,
        FoldState.UNFOLDED,
    ]


def test_fold_formatting():
    pass


def test_filter_folded(tagged_messages):
    m = tagged_messages
    m.filter(tags=TagPattern("system", ".*"))
    assert m.messages == tagged_messages.messages[:1]
    assert m.folds.message_states == tagged_messages.folds.message_states[:1]

    m = tagged_messages
    m.filter(tags=TagPattern("user", ".*"))
    assert m.messages == tagged_messages.messages[1:3]
    assert m.folds.message_states == tagged_messages.folds.message_states[1:3]
