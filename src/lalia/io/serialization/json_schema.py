from __future__ import annotations

import re
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    Discriminator,
    Field,
    Tag,
)
from pydantic.alias_generators import to_camel, to_snake
from pydantic.dataclasses import dataclass


class PropDiscriminator(StrEnum):
    TYPE = "type_"
    COMPOSITE = "composite"

    @property
    def alias(self) -> str:
        return to_camel(self.value).rstrip("_")


class PropType(StrEnum):
    NUMBER = "number"
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"

    @classmethod
    @property
    def discriminator(cls) -> PropDiscriminator:
        return PropDiscriminator.TYPE


class JsonSchemaComposite(StrEnum):
    ONE_OF = "oneOf"
    ANY_OF = "anyOf"
    ALL_OF = "allOf"
    NOT_ = "not"

    @classmethod
    @property
    def discriminator(cls) -> str:
        return PropDiscriminator.COMPOSITE

    def to_snake(self) -> str:
        return to_snake(self.value)


@dataclass
class StringProp:
    max_length: int | None = Field(default=None, alias="maxLength", ge=0)
    min_length: int | None = Field(default=None, alias="minLength", ge=0)
    pattern: str | re.Pattern | None = None
    title: str | None = None
    enum: list[str] | None = None
    type_: Literal[PropType.STRING] = Field(
        default=PropType.STRING, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class IntegerProp:
    maximum: int | None = Field(default=None, ge=0)
    minimum: int | None = Field(default=None, ge=0)
    title: str | None = None
    enum: list[int] | None = None
    type_: Literal[PropType.INTEGER] = Field(
        default=PropType.INTEGER, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class NumberProp:
    maximum: float | None = Field(default=None, ge=0)
    minimum: float | None = Field(default=None, ge=0)
    title: str | None = None
    enum: list[int] | None = None
    type_: Literal[PropType.NUMBER] = Field(
        default=PropType.NUMBER, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class BooleanProp:
    title: str | None = None
    type_: Literal[PropType.BOOLEAN] = Field(
        default=PropType.BOOLEAN, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class ArrayProp:
    items: Prop | None = None
    title: str | None = None
    type_: Literal[PropType.ARRAY] = Field(
        default=PropType.ARRAY, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class NullProp:
    type_: Literal[PropType.NULL] = Field(
        default=PropType.NULL, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class ObjectProp:
    properties: dict[str, Prop] | None = None
    additional_properties: Prop | None = Field(
        default=None, alias="additionalProperties"
    )
    required: list[str] | None = None
    title: str | None = None
    type_: Literal[PropType.OBJECT] = Field(
        default=PropType.OBJECT, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class OneOfProp:
    one_of: list[Prop] = Field(alias=JsonSchemaComposite.ONE_OF)


@dataclass
class AnyOfProp:
    any_of: list[Prop] = Field(alias=JsonSchemaComposite.ANY_OF)


@dataclass
class AllOfProp:
    all_of: list[Prop] = Field(alias=JsonSchemaComposite.ALL_OF)


@dataclass
class NotProp:
    not_: Prop = Field(alias="not")


def discriminate_composite_prop(payload: Any) -> str | None:
    match payload:
        case dict():
            for composite_type in JsonSchemaComposite:
                if composite_type in payload:
                    return composite_type
            raise ValueError("Could not discriminate composite property.")
        case _:
            for composite_type in JsonSchemaComposite:
                if hasattr(payload, composite_type.to_snake()):
                    return composite_type
            return None


def discriminate_prop(payload: Any) -> str:
    match payload:
        case dict() as payload:
            return (
                PropType.discriminator
                if PropType.discriminator.alias in payload
                else JsonSchemaComposite.discriminator
            )
        case _ as obj:
            return (
                PropType.discriminator
                if hasattr(obj, PropType.discriminator)
                else JsonSchemaComposite.discriminator
            )


TypeProp = Annotated[
    StringProp
    | IntegerProp
    | NumberProp
    | BooleanProp
    | ArrayProp
    | ObjectProp
    | NullProp,
    Field(discriminator=PropType.discriminator),
]

CompositeProp = Annotated[
    Annotated[OneOfProp, Tag(JsonSchemaComposite.ONE_OF)]
    | Annotated[AnyOfProp, Tag(JsonSchemaComposite.ANY_OF)]
    | Annotated[AllOfProp, Tag(JsonSchemaComposite.ALL_OF)]
    | Annotated[NotProp, Tag(JsonSchemaComposite.NOT_)],
    Discriminator(discriminate_composite_prop),
]


Prop = Annotated[
    Annotated[TypeProp, Tag(PropType.discriminator)]
    | Annotated[CompositeProp, Tag(JsonSchemaComposite.discriminator)],
    Discriminator(discriminate_prop),
]
