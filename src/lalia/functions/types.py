from __future__ import annotations

import builtins
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import TypeAdapter, computed_field
from pydantic.dataclasses import dataclass

from lalia.chat.finish_reason import FinishReason

T = TypeVar("T")
A = TypeVar("A")


class TypeScriptTypes(StrEnum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    ANY_ARRAY = "any[]"
    SET_ANY = "Set<any>"
    RECORD_STRING_ANY = "Record<string, any>"
    NULL = "null"
    # TODO: add all the types!

    @classmethod
    def from_python_type(cls, py_type: type[Any]) -> str:
        """
        Determines the TypeScript equivalent of a given Python type.
        """
        match py_type:
            case builtins.str:
                return cls.STRING
            case builtins.bool:
                return cls.BOOLEAN
            case builtins.int:
                return cls.NUMBER
            case builtins.float:
                return cls.NUMBER
            case builtins.tuple:
                return cls.ANY_ARRAY
            case builtins.set:
                return cls.SET_ANY
            case builtins.dict:
                return cls.RECORD_STRING_ANY
            case None:
                return cls.NULL
            case _:
                return "any"  # default case


@dataclass
class Error:
    message: str


@dataclass
class BaseProp(ABC):
    """Represents basic type functionality."""

    description: str | None = None
    default: Any | None = None

    @property
    @abstractmethod
    def type(self) -> str:
        pass

    @classmethod
    def to_json_schema(cls) -> dict:
        return TypeAdapter(cls).json_schema()

    @classmethod
    def from_json_schema(cls, schema: dict) -> BaseProp:
        return cls(**schema)

    @classmethod
    def parse_type_and_description(cls, param_type, description, enum, default):
        # catch the case where the parameter is a variant
        if "|" in param_type:
            type_parts = [t.strip().replace('"', "") for t in param_type.split("|")]
            variant_props = [
                cls.parse_type_and_description(t, description, enum, default)
                for t in type_parts
            ]
            return VariantProp(
                description=description, anyof=variant_props, default=default
            )

        match param_type:
            case "string":
                return StringProp(description=description, enum=enum, default=default)
            case "number" | "integer":
                return NumberProp(description=description, enum=enum, default=default)
            case "boolean":
                return BoolProp(description=description, default=default)
            case "array":
                return ArrayProp(description=description, default=default)
            case "object":
                return ObjectProp(description=description, default=default)
            case "null":
                return NullProp(description=description)
            case _:
                raise ValueError(f"Unknown type: {param_type}")


@dataclass
class StringProp(BaseProp):
    """Represents a property of type string in a function signature."""

    enum: list[str] | None = None

    @computed_field
    @property
    def type(self) -> str:
        return "string"


@dataclass
class NumberProp(BaseProp):
    """Represents a property of type number in a function signature."""

    minimum: int | None = None
    maximum: int | None = None
    enum: list[int] | list[float] | None = None

    @computed_field
    @property
    def type(self) -> str:
        return "number"


@dataclass
class BoolProp(BaseProp):
    """Represents a property of type boolean in a function signature."""

    @computed_field
    @property
    def type(self) -> str:
        return "boolean"


@dataclass
class NullProp(BaseProp):
    """Represents a property that can be null in a function signature."""

    @computed_field
    @property
    def type(self) -> str:
        return "null"


@dataclass
class ArrayProp(BaseProp):
    """Represents a property of type array in a function signature."""

    items: PropItem | None = None

    @computed_field
    @property
    def type(self) -> str:
        return "array"


@dataclass
class ObjectProp(BaseProp):
    """Represents a property of type object in a function signature."""

    required: list[str] | None = None
    properties: dict[str, PropItem] | None = None

    @computed_field
    @property
    def type(self) -> str:
        return "object"


@dataclass
class VariantProp(BaseProp):
    """Represents a property of type enum in a function signature."""

    required: list[str] | None = None
    anyof: list[PropItem] | None = None

    @property
    def type(self) -> str:
        if isinstance(self.anyof, list):
            types = [d.type for d in self.anyof]
            return " | ".join(f"{t}" for t in types)
        else:
            return "any"


ArrayProp.__annotations__["items"] = (
    StringProp | NumberProp | BoolProp | NullProp | ObjectProp | ArrayProp | None
)


PropItem = (
    StringProp | NumberProp | BoolProp | NullProp | ArrayProp | ObjectProp | VariantProp
)


@dataclass
class ReturnType:
    """Holds the type information for a function's return value."""

    type: Any

    @classmethod
    def to_json_schema(cls) -> dict:
        return TypeAdapter(cls).json_schema()


@dataclass
class FunctionSchema:
    """Describes a function schema, including its parameters and return type."""

    name: str
    parameters: PropItem
    description: str = ""
    return_type: ReturnType | None = None

    def to_json_schema(self) -> dict:
        schema_dict = TypeAdapter(FunctionSchema).dump_python(self, exclude_none=True)

        return schema_dict

    @classmethod
    def from_json_schema(cls, schema: dict) -> FunctionSchema:
        # this will create an instance of FunctionSchema, even if 'parameters' and
        # 'return_type' are complex, nested objects. Pydantic handles the
        # deserialization recursively.
        return cls(**schema)


@dataclass
class Result:
    """
    An anonymous result type that wraps a result or an error.
    """

    value: Any | None = None
    error: Error | None = None
    finish_reason: FinishReason = FinishReason.DELEGATE


@dataclass
class FunctionCallResult(Generic[A, T]):
    """
    A result type that is a superset of `Result` containg additional metadata.
    """

    name: str
    arguments: dict[str, A]
    value: T | None = None
    error: Error | None = None
    finish_reason: FinishReason = FinishReason.DELEGATE

    def to_string(self) -> str:
        match self.error, self.value:
            case None, result:
                return str(result)
            case Error(message), None:
                return f"Error: {message}"
            case _:
                raise ValueError("Either `error` or `result` must be `None`")
