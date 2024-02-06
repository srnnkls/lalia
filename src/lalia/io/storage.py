from typing import (
    Any,
    ClassVar,
    Generic,
    Hashable,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from pydantic import TypeAdapter

from lalia.io.serialization import Serializable

IDType_contra = TypeVar("IDType_contra", bound=Hashable, contravariant=True)


@runtime_checkable
class StorageBackend(Protocol[IDType_contra]):
    def exists(self, id_: IDType_contra) -> bool: ...

    def load(self, id_: IDType_contra) -> dict[str, Any]: ...

    def save(self, obj: Serializable, id_: IDType_contra): ...


class DictStorageBackend(Generic[IDType_contra]):
    data: ClassVar[dict[str, Any]] = {}

    def exists(self, id_: IDType_contra) -> bool:
        return id_ in self.data

    def load(self, id_: IDType_contra) -> dict[str, Any]:
        if id_ in self.data:
            return self.data[id_]
        else:
            raise KeyError(f"Could not find '{id_!r}' in {self}")

    def save(self, obj: Serializable, id_: IDType_contra):
        adapter = TypeAdapter(type(obj))
        self.data[id_] = adapter.dump_python(obj)  # type: ignore
