from __future__ import annotations

import re
from dataclasses import fields
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeGuard, get_args

from pydantic import (
    AliasChoices,
    Discriminator,
    Field,
    Tag,
)
from pydantic.alias_generators import to_camel, to_snake
from pydantic.dataclasses import dataclass

from lalia.utils.decorators import classproperty

JSON_SCHEMA_ANY_TAG = "any"


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


class JsonSchemaKeyword(StrEnum):
    ADDITIONAL_PROPERTIES = "additionalProperties"
    MAX_PROPERTIES = "maxProperties"
    MIN_PROPERTIES = "minProperties"
    PATTERN_PROPERTIES = "patternProperties"
    PROPERTY_NAMES = "propertyNames"
    ADDITIONAL_ITEMS = "additionalItems"
    MAX_ITEMS = "maxItems"
    MIN_ITEMS = "minItems"
    UNIQUE_ITEMS = "uniqueItems"
    MIN_CONTAINS = "minContains"
    MAX_CONTAINS = "maxContains"
    PREFIX_ITEMS = "prefixItems"
    UNEVALUATED_ITEMS = "unevaluatedItems"
    UNEVALUATED_PROPERTIES = "unevaluatedProperties"
    MAX_LENGTH = "maxLength"
    MIN_LENGTH = "minLength"
    MULTIPLE_OF = "multipleOf"
    EXCLUSIVE_MAXIMUM = "exclusiveMaximum"
    EXCLUSIVE_MINIMUM = "exclusiveMinimum"

    def to_snake(self) -> str:
        return to_snake(self.value)


@dataclass
class AnyProp:
    description: str | None = None
    title: str | None = None


@dataclass
class StringProp:
    description: str | None = None
    default: str | None = None
    max_length: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.MAX_LENGTH,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.MAX_LENGTH.to_snake(), JsonSchemaKeyword.MAX_LENGTH
        ),
        ge=0,
    )
    min_length: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.MIN_LENGTH,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.MIN_LENGTH.to_snake(), JsonSchemaKeyword.MIN_LENGTH
        ),
        ge=0,
    )
    pattern: str | re.Pattern | None = None
    format: str | None = None
    title: str | None = None
    enum: list[str] | None = None
    type_: Literal[JsonSchemaType.STRING] = Field(
        default=JsonSchemaType.STRING, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class IntegerProp:
    description: str | None = None
    default: int | None = None
    maximum: int | None = None
    minimum: int | None = None
    exclusive_maximum: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.EXCLUSIVE_MAXIMUM,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.EXCLUSIVE_MAXIMUM.to_snake(),
            JsonSchemaKeyword.EXCLUSIVE_MAXIMUM,
        ),
    )
    exclusive_minimum: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.EXCLUSIVE_MINIMUM,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.EXCLUSIVE_MINIMUM.to_snake(),
            JsonSchemaKeyword.EXCLUSIVE_MINIMUM,
        ),
    )
    multiple_of: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.MULTIPLE_OF,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.MULTIPLE_OF.to_snake(), JsonSchemaKeyword.MULTIPLE_OF
        ),
    )
    title: str | None = None
    enum: list[int] | None = None
    type_: Literal[JsonSchemaType.INTEGER] = Field(
        default=JsonSchemaType.INTEGER, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class NumberProp:
    description: str | None = None
    default: float | None = None
    maximum: float | None = None
    minimum: float | None = None
    exclusive_maximum: float | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.EXCLUSIVE_MAXIMUM,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.EXCLUSIVE_MAXIMUM.to_snake(),
            JsonSchemaKeyword.EXCLUSIVE_MAXIMUM,
        ),
    )
    exclusive_minimum: float | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.EXCLUSIVE_MINIMUM,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.EXCLUSIVE_MINIMUM.to_snake(),
            JsonSchemaKeyword.EXCLUSIVE_MINIMUM,
        ),
    )
    multiple_of: float | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.MULTIPLE_OF,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.MULTIPLE_OF.to_snake(), JsonSchemaKeyword.MULTIPLE_OF
        ),
    )
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
    prefix_items: list[dict[str, Any]] | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.PREFIX_ITEMS,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.PREFIX_ITEMS.to_snake(), JsonSchemaKeyword.PREFIX_ITEMS
        ),
    )
    unevaluated_items: bool | dict[str, Any] | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.UNEVALUATED_ITEMS,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.UNEVALUATED_ITEMS.to_snake(),
            JsonSchemaKeyword.UNEVALUATED_ITEMS,
        ),
    )
    contains: Prop | None = None
    min_contains: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.MIN_CONTAINS,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.MIN_CONTAINS.to_snake(), JsonSchemaKeyword.MIN_CONTAINS
        ),
    )
    max_contains: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.MAX_CONTAINS,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.MAX_CONTAINS.to_snake(), JsonSchemaKeyword.MAX_CONTAINS
        ),
    )
    min_items: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.MIN_ITEMS,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.MIN_ITEMS.to_snake(), JsonSchemaKeyword.MIN_ITEMS
        ),
        ge=0,
    )
    max_items: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.MAX_ITEMS,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.MAX_ITEMS.to_snake(), JsonSchemaKeyword.MAX_ITEMS
        ),
        ge=0,
    )
    unique_items: bool | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.UNIQUE_ITEMS,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.UNIQUE_ITEMS.to_snake(), JsonSchemaKeyword.UNIQUE_ITEMS
        ),
    )
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
        default=None,
        serialization_alias=JsonSchemaKeyword.ADDITIONAL_PROPERTIES,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.ADDITIONAL_PROPERTIES.to_snake(),
            JsonSchemaKeyword.ADDITIONAL_PROPERTIES,
        ),
    )
    pattern_properties: dict[str | re.Pattern, Any] | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.PATTERN_PROPERTIES,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.PATTERN_PROPERTIES.to_snake(),
            JsonSchemaKeyword.PATTERN_PROPERTIES,
        ),
    )
    unevaluated_properties: bool | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.UNEVALUATED_PROPERTIES,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.UNEVALUATED_PROPERTIES.to_snake(),
            JsonSchemaKeyword.UNEVALUATED_PROPERTIES,
        ),
    )
    property_names: dict[str, str] | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.PROPERTY_NAMES,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.PROPERTY_NAMES.to_snake(),
            JsonSchemaKeyword.PROPERTY_NAMES,
        ),
    )
    min_properties: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.MIN_PROPERTIES,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.MIN_PROPERTIES.to_snake(),
            JsonSchemaKeyword.MIN_PROPERTIES,
        ),
        ge=0,
    )
    max_properties: int | None = Field(
        default=None,
        serialization_alias=JsonSchemaKeyword.MAX_PROPERTIES,
        validation_alias=AliasChoices(
            JsonSchemaKeyword.MAX_PROPERTIES.to_snake(),
            JsonSchemaKeyword.MAX_PROPERTIES,
        ),
        ge=0,
    )
    required: list[str] | None = None
    type_: Literal[JsonSchemaType.OBJECT] = Field(
        default=JsonSchemaType.OBJECT, alias=PropDiscriminator.TYPE.alias
    )


@dataclass
class OneOfProp:
    one_of: list[Prop] = Field(
        serialization_alias=JsonSchemaComposite.ONE_OF,
        validation_alias=AliasChoices(
            JsonSchemaComposite.ONE_OF, JsonSchemaComposite.ONE_OF.name.lower()
        ),
    )
    description: str | None = None
    default: Any | None = None
    title: str | None = None


@dataclass
class AnyOfProp:
    any_of: list[Prop] = Field(
        serialization_alias=JsonSchemaComposite.ANY_OF,
        validation_alias=AliasChoices(
            JsonSchemaComposite.ANY_OF, JsonSchemaComposite.ANY_OF.name.lower()
        ),
    )
    description: str | None = None
    default: Any | None = None
    title: str | None = None


@dataclass
class AllOfProp:
    all_of: list[Prop] = Field(
        serialization_alias=JsonSchemaComposite.ALL_OF,
        validation_alias=AliasChoices(
            JsonSchemaComposite.ALL_OF, JsonSchemaComposite.ALL_OF.name.lower()
        ),
    )
    description: str | None = None
    default: Any | None = None


@dataclass
class NotProp:
    not_: Prop = Field(
        serialization_alias=JsonSchemaComposite.NOT_,
        validation_alias=AliasChoices(
            JsonSchemaComposite.NOT_, JsonSchemaComposite.NOT_.name.lower()
        ),
    )
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
        elif any(field.name in payload for field in fields(AnyProp)):
            return JSON_SCHEMA_ANY_TAG

    elif is_type_prop(payload):
        if isinstance(payload, AnyProp):
            return JSON_SCHEMA_ANY_TAG
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
    | Annotated[AnyProp, Tag(JSON_SCHEMA_ANY_TAG)]
    | Annotated[CompositeProp, Tag(JsonSchemaComposite.discriminator)],
    Discriminator(discriminate_prop),
]


TYPE_PROP_UNION = get_args(TypeProp)[0]
COMPOSITE_PROP_UNION = tuple(
    get_args(prop)[0] for prop in get_args(get_args(CompositeProp)[0])
)


def is_type_prop(payload: Any) -> TypeGuard[TypeProp]:
    return isinstance(payload, TYPE_PROP_UNION)


def is_composite_prop(payload: Any) -> TypeGuard[CompositeProp]:
    return isinstance(payload, COMPOSITE_PROP_UNION)
