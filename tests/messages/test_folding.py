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


def test_fold_no_args(tagged_messages):
    m = tagged_messages
    m.fold()
    implicitly_folded_states = list(m.folds.message_states)

    m.fold(tags=m.folds.default_fold_tags)
    explicitly_folded_states = list(m.folds.message_states)
    print(implicitly_folded_states, explicitly_folded_states)

    assert implicitly_folded_states == explicitly_folded_states


def test_unfold_no_args(tagged_messages):
    m = tagged_messages
    m.unfold()
    implicitly_unfolded_states = list(m.folds.message_states)

    m.unfold(tags={})
    explicitly_unfolded_states = list(m.folds.message_states)
    print(implicitly_unfolded_states, explicitly_unfolded_states)

    assert implicitly_unfolded_states == explicitly_unfolded_states


def test_expand(tagged_messages):
    m = tagged_messages
    m.fold(TagPattern(".*", ".*"))
    with m.expand(TagPattern("user", ".*")) as expanded_user:
        assert expanded_user.folds.message_states == [
            FoldState.FOLDED,
            FoldState.UNFOLDED,
            FoldState.FOLDED,
            FoldState.UNFOLDED,
        ]


def test_collapse(tagged_messages):
    m = tagged_messages
    with m.collapse(
        TagPattern("system", ".*") | TagPattern("assistant", ".*")
    ) as collapsed_system_and_assistant:
        assert collapsed_system_and_assistant.folds.message_states == [
            FoldState.FOLDED,
            FoldState.UNFOLDED,
            FoldState.FOLDED,
            FoldState.UNFOLDED,
        ]
