from dataclasses import Field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, final, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class Dataclass(Protocol):  # type: ignore
    @property
    def __dataclass_fields__(self) -> dict[str, Field]: ...


if TYPE_CHECKING:

    @final
    class Dataclass(Protocol):
        __dataclass_fields__: ClassVar[dict[str, Any]]


# TODO: Complete? Maybe add missing types
# https://github.com/pydantic/pydantic-core/tree/main/src/serializers/type_serializers
Serializable = (
    int
    | float
    | str
    | bool
    | None
    | list
    | dict
    | tuple
    | set
    | frozenset
    | bytes
    | Dataclass
    | BaseModel
)
