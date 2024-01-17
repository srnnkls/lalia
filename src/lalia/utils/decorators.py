from __future__ import annotations

from collections.abc import Callable
from functools import update_wrapper, wraps
from typing import TYPE_CHECKING, Generic, TypeVar, cast

T = TypeVar("T")
RT = TypeVar("RT")


class _Classproperty(Generic[T, RT]):
    """
    Class property attribute (read-only).

    Same usage as @property, but taking the class as the first argument.

        class C:
            @classproperty
            def x(cls):
                return 0

        print(C.x)    # 0
        print(C().x)  # 0

    Source: https://github.com/python/cpython/issues/89519#issuecomment-1397534245
    """

    def __init__(self, func: Callable[[type[T]], RT]) -> None:
        self.__wrapped__ = cast(Callable[[type[T]], RT], func)
        update_wrapper(self, func)

    def __set_name__(self, owner: type[T], name: str) -> None:
        # Update based on class context.
        self.__module__ = owner.__module__
        self.__name__ = name
        self.__qualname__ = owner.__qualname__ + "." + name

    def __get__(self, instance: T | type[None], owner: type[T] | None = None) -> RT:
        if owner is None:
            owner = type(instance)
        if owner is None:
            raise AssertionError("Unreachable")
        return self.__wrapped__(owner)

    def __call__(self, *args, **kwargs):
        return self.__wrapped__(*args, **kwargs)


if TYPE_CHECKING:

    def mock(func: Callable[[T], RT]):
        wraps(func)
        return cast(RT, None)

    classproperty = mock
else:
    classproperty = _Classproperty
