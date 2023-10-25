from __future__ import annotations

from enum import Enum


class FoldState(Enum):
    """
    Represents the state of a fold.
    """

    UNFOLDED = 0
    FOLDED = 1

    def __invert__(self) -> FoldState:
        return FoldState.UNFOLDED if self is FoldState.FOLDED else FoldState.FOLDED
