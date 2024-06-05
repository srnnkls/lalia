import importlib
from collections.abc import Callable, Sequence
from types import BuiltinFunctionType, FunctionType
from typing import Any, ClassVar

from pydantic import TypeAdapter


def is_callable_instance(callable_: object) -> bool:
    if not callable(callable_):
        return False
    return not isinstance(callable_, FunctionType | BuiltinFunctionType)


def _import_by_qualname(qualname: str) -> Callable[..., Any]:
    """
    Import an object by its fully qualified name.
    """
    try:
        module_name, function_name = qualname.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Could not import the function: {qualname}") from e


class CallableRegistry:
    """
    A registry for callables.
    """

    _callables: ClassVar[dict[str, Callable[..., Any]]] = {}

    @classmethod
    def register_callable(cls, callable_: Callable[..., Any]):
        """
        Register a callable.
        """
        if is_callable_instance(instance := callable_):
            obj = type(instance)
        else:
            obj = callable_
        key = f"{obj.__module__}.{obj.__qualname__}"
        cls._callables[key] = obj

    @classmethod
    def get_callable(cls, name: str) -> Callable[..., Any] | None:
        """
        Get a callable from the registry.
        """
        if name in cls._callables:
            return cls._callables[name]
        else:
            return None


def get_callable(name: str) -> Callable[..., Any]:
    """
    Try to get a callable from the registry, otherwise import it by its fully qualified
    name.
    """
    registered_callable = CallableRegistry.get_callable(name)
    if registered_callable is not None:
        return registered_callable
    else:
        return _import_by_qualname(name)


def _parse_serialized_callable(
    serialized_callable: dict[str, Any]
) -> Callable[..., Any]:
    """
    Parse a serialized callable.
    """
    type_ = serialized_callable["type"]
    name = serialized_callable["name"]
    module = serialized_callable["module"]
    attributes = serialized_callable["attributes"]

    if type_ == "function":
        return get_callable(f"{module}.{name}")
    elif type_ == name:
        cls = get_callable(f"{module}.{name}")
        return cls(**attributes)
    else:
        raise ValueError(f"Unknown callable type: {type_}")


def parse_callable(
    callable_: Callable[..., Any] | dict[str, Any]
) -> Callable[..., Any]:
    """
    Parse a serialized callable.
    """
    match callable_:
        case Callable():
            return callable_
        case dict():
            return _parse_serialized_callable(callable_)
        case _:
            raise ValueError(f"Unknown callable type: {type(callable_)}")


def parse_callables(
    serialized_callables: Sequence[dict[str, Any]] | Sequence[Callable[..., Any]]
) -> list[Callable[..., Any]]:
    """
    Parse a list of serialized callables.
    """
    callables = []
    for serialized_callable in serialized_callables:
        callables.append(parse_callable(serialized_callable))
    return callables


def serialize_callable(callable_: Callable[..., Any]) -> dict[str, Any]:
    if is_callable_instance(instance := callable_):
        cls = type(instance)
        name = cls.__qualname__
        module = cls.__module__
        adapter = TypeAdapter(cls)
        attributes = adapter.dump_python(instance)
    else:
        name = callable_.__qualname__
        module = callable_.__module__
        attributes = {}

    return {
        "type": type(callable_).__name__,
        "name": name,
        "module": module,
        "attributes": attributes,
    }


def serialize_callables(
    callables: Sequence[Callable[..., Any]]
) -> list[dict[str, Any]]:
    serialized = []
    for callable_ in callables:
        serialized.append(serialize_callable(callable_))
    return serialized
