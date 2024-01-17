from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeGuard, get_args

from pydantic import (
    Discriminator,
    Field,
    Tag,
)
from pydantic.alias_generators import to_camel, to_snake
from pydantic.dataclasses import dataclass

from lalia.utils.decorators import classproperty


class PropDiscriminator(StrEnum):
    TYPE = "type_"
    COMPOSITE = "composite"

    @property
    def alias(self) -> str:
        return to_camel(self.value).rstrip("_")


class JsonSchemaType(StrEnum):
    NUMBER = "number"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"

    @classproperty
    def discriminator(cls) -> PropDiscriminator:
        return PropDiscriminator.TYPE


class JsonSchemaComposite(StrEnum):
    ONE_OF = "oneOf"
    ANY_OF = "anyOf"
    ALL_OF = "allOf"
    NOT_ = "not"

    @classproperty
    def discriminator(cls) -> str:
        return PropDiscriminator.COMPOSITE

    def to_snake(self) -> str:
        return to_snake(self.value)


@dataclass
class StringProp:
    description: str | None = None
    default: str | None = None
    max_length: int | None = Field(default=None, alias="maxLength", ge=0)
    min_length: int | None = Field(default=None, alias="minLength", ge=0)
    pattern: str | re.Pattern | None = None
    title: str | None = None
    enum: list[str] | None = None
    type_: Literal[JsonSchemaType.STRING] = Field(
        default=JsonSchemaType.STRING, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class IntegerProp:
    description: str | None = None
    default: int | None = None
    maximum: int | None = Field(default=None, ge=0)
    minimum: int | None = Field(default=None, ge=0)
    title: str | None = None
    enum: list[int] | None = None
    type_: Literal[JsonSchemaType.INTEGER] = Field(
        default=JsonSchemaType.INTEGER, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class NumberProp:
    description: str | None = None
    default: float | None = None
    maximum: float | None = Field(default=None, ge=0)
    minimum: float | None = Field(default=None, ge=0)
    title: str | None = None
    enum: list[int] | None = None
    type_: Literal[JsonSchemaType.NUMBER] = Field(
        default=JsonSchemaType.NUMBER, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class BooleanProp:
    description: str | None = None
    default: bool | None = None
    title: str | None = None
    type_: Literal[JsonSchemaType.BOOLEAN] = Field(
        default=JsonSchemaType.BOOLEAN, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class ArrayProp:
    description: str | None = None
    default: Any | None = None
    items: Prop | None = None
    title: str | None = None
    type_: Literal[JsonSchemaType.ARRAY] = Field(
        default=JsonSchemaType.ARRAY, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class NullProp:
    description: str | None = None
    default: Any | None = None
    title: str | None = None
    type_: Literal[JsonSchemaType.NULL] = Field(
        default=JsonSchemaType.NULL, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class ObjectProp:
    description: str | None = None
    default: Any | None = None
    title: str | None = None
    properties: dict[str, Prop] | None = None
    additional_properties: bool | Prop | None = Field(
        default=None, alias="additionalProperties"
    )
    required: list[str] | None = None
    type_: Literal[JsonSchemaType.OBJECT] = Field(
        default=JsonSchemaType.OBJECT, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class OneOfProp:
    one_of: list[Prop] = Field(alias=JsonSchemaComposite.ONE_OF)
    description: str | None = None
    default: Any | None = None
    title: str | None = None


@dataclass
class AnyOfProp:
    any_of: list[Prop] = Field(alias=JsonSchemaComposite.ANY_OF)
    description: str | None = None
    default: Any | None = None
    title: str | None = None


@dataclass
class AllOfProp:
    all_of: list[Prop] = Field(alias=JsonSchemaComposite.ALL_OF)
    description: str | None = None
    default: Any | None = None


@dataclass
class NotProp:
    not_: Prop = Field(alias="not")
    description: str | None = None
    default: Any | None = None
    title: str | None = None


def discriminate_composite_prop(payload: Any) -> str | None:
    if isinstance(payload, dict):
        for composite_type in JsonSchemaComposite:
            if composite_type in payload:
                return composite_type

    if is_composite_prop(payload):
        for composite_type in JsonSchemaComposite:
            if hasattr(payload, composite_type.to_snake()):
                return composite_type


def discriminate_prop(payload: Any) -> str | None:
    if isinstance(payload, dict):
        if JsonSchemaType.discriminator.alias in payload:
            return JsonSchemaType.discriminator
        elif any(composite_type in payload for composite_type in JsonSchemaComposite):
            return JsonSchemaComposite.discriminator

    elif is_type_prop(payload):
        return JsonSchemaType.discriminator
    elif is_composite_prop(payload):
        return JsonSchemaComposite.discriminator


TypeProp = Annotated[
    StringProp
    | IntegerProp
    | NumberProp
    | BooleanProp
    | ArrayProp
    | ObjectProp
    | NullProp,
    Field(discriminator=JsonSchemaType.discriminator),
]

CompositeProp = Annotated[
    Annotated[OneOfProp, Tag(JsonSchemaComposite.ONE_OF)]
    | Annotated[AnyOfProp, Tag(JsonSchemaComposite.ANY_OF)]
    | Annotated[AllOfProp, Tag(JsonSchemaComposite.ALL_OF)]
    | Annotated[NotProp, Tag(JsonSchemaComposite.NOT_)],
    Discriminator(discriminate_composite_prop),
]

Prop = Annotated[
    Annotated[TypeProp, Tag(JsonSchemaType.discriminator)]
    | Annotated[CompositeProp, Tag(JsonSchemaComposite.discriminator)],
    Discriminator(discriminate_prop),
]


TYPE_PROP_UNION = get_args(TypeProp)[0]
COMPOSITE_PROP_UNION = tuple(
    [get_args(prop)[0] for prop in get_args(get_args(CompositeProp)[0])]
)


def is_type_prop(payload: Any) -> TypeGuard[TypeProp]:
    return isinstance(payload, TYPE_PROP_UNION)


def is_composite_prop(payload: Any) -> TypeGuard[CompositeProp]:  # type: ignore
    return isinstance(payload, COMPOSITE_PROP_UNION)
