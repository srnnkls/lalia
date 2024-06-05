from typing import (
    Any,
    ClassVar,
    Hashable,
    Protocol,
    runtime_checkable,
)

from pydantic import TypeAdapter

from lalia.io.serialization import Serializable


@runtime_checkable
class StorageBackend(Protocol):
    def exists(self, id_: Hashable) -> bool: ...

    def load(self, id_: Hashable) -> dict[str, Any]: ...

    def save(self, obj: Serializable, id_: Hashable): ...


class DictStorageBackend:
    data: ClassVar[dict[str, Any]] = {}

    def exists(self, id_: Hashable) -> bool:
        return id_ in self.data

    def load(self, id_: Hashable) -> dict[str, Any]:
        if id_ in self.data:
            return self.data[id_]
        else:
            raise KeyError(f"Could not find '{id_!r}' in {self}")

    def save(self, obj: Serializable, id_: Hashable):
        adapter = TypeAdapter(type(obj))
        self.data[id_] = adapter.dump_python(obj)  # type: ignore
