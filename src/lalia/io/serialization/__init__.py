from collections.abc import Mapping
from dataclasses import Field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


@runtime_checkable
class Dataclass(Protocol):
    @property
    def __dataclass_fields__(self) -> Mapping[str, Field]: ...


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
    | type[BaseModel]
)
