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

from lalia.io.logging import get_logger
from lalia.io.serialization import Serializable

logger = get_logger(__name__)

IDType_contra = TypeVar("IDType_contra", bound=Hashable, contravariant=True)


@runtime_checkable
class StorageBackend(Protocol[IDType_contra]):
    def load(self, id_: IDType_contra) -> dict[str, Any]:
        ...

    def save(self, obj: Serializable, id_: IDType_contra):
        ...


class DictStorageBackend(Generic[IDType_contra]):
    data: ClassVar[dict[str, Any]] = {}

    def load(self, id_: IDType_contra) -> dict[str, Any]:
        if id_ in self.data:
            return self.data[id_]
        else:
            raise KeyError(f"Could not find '{id_!r}' in {self}")

    def save(self, obj: Serializable, id_: IDType_contra):
        if id_ in self.data:
            logger.warning(f"Overwriting '{id_!r}' in {self}")
        adapter = TypeAdapter(type(obj))
        self.data[id_] = adapter.dump_python(obj)  # type: ignore
