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
