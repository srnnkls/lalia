from enum import Enum

import pytest
from pydantic import TypeAdapter
from pydantic.dataclasses import dataclass

from lalia.functions import dereference_schema
from lalia.io.serialization.json_schema import (
    AllOfProp,
    AnyOfProp,
    ArrayProp,
    BooleanProp,
    IntegerProp,
    NullProp,
    NumberProp,
    ObjectProp,
    StringProp,
)


class E(Enum):
    Y = 0
    Z = 1


@dataclass
class C:
    l: list[int]  # noqa: E741
    d: dict[str, str]
    dv: dict[str, str | int]


@dataclass
class A:
    i: int
    f: float
    b: bool
    fi: float
    s: str
    c: C
    e: E = E.Z
    v: int | str | None = None
    n: int | None = None


@pytest.fixture()
def a():
    return A(
        i=1,
        f=3.1415,
        b=True,
        fi=1.0,
        s="hello",
        c=C(l=[2, 3], d={"mellow": "fellow"}, dv={"yellow": "jellow", "sellow": 5}),
        e=E.Y,
        v=5,
        n=None,
    )


@pytest.fixture()
def adapter():
    return TypeAdapter(A)


@pytest.fixture()
def class_schema(adapter):
    return dereference_schema(adapter.json_schema())


@pytest.fixture()
def schema_instance(adapter, a):
    return adapter.dump_python(a)


@pytest.fixture()
def expected_class_schema_object():
    return {
        "properties": {
            "i": {"title": "I", "type": "integer"},
            "f": {"title": "F", "type": "number"},
            "b": {"title": "B", "type": "boolean"},
            "fi": {"title": "Fi", "type": "number"},
            "s": {"title": "S", "type": "string"},
            "c": {
                "properties": {
                    "l": {"items": {"type": "integer"}, "title": "L", "type": "array"},
                    "d": {
                        "additionalProperties": {"type": "string"},
                        "title": "D",
                        "type": "object",
                    },
                    "dv": {
                        "additionalProperties": {
                            "anyOf": [{"type": "string"}, {"type": "integer"}]
                        },
                        "title": "Dv",
                        "type": "object",
                    },
                },
                "required": ["l", "d", "dv"],
                "title": "C",
                "type": "object",
            },
            "e": {
                "allOf": [{"enum": [0, 1], "title": "E", "type": "integer"}],
                "default": 1,
            },
            "v": {
                "anyOf": [{"type": "integer"}, {"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "V",
            },
            "n": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "default": None,
                "title": "N",
            },
        },
        "required": ["i", "f", "b", "fi", "s", "c"],
        "title": "A",
        "type": "object",
    }


@pytest.fixture()
def expected_custom_schema_object():
    return ObjectProp(
        properties={
            "i": IntegerProp(
                title="I",
            ),
            "f": NumberProp(
                title="F",
            ),
            "b": BooleanProp(title="B"),
            "fi": NumberProp(
                title="Fi",
            ),
            "s": StringProp(
                title="S",
            ),
            "c": ObjectProp(
                properties={
                    "l": ArrayProp(
                        items=IntegerProp(),
                        title="L",
                    ),
                    "d": ObjectProp(
                        additional_properties=StringProp(),
                        title="D",
                    ),
                    "dv": ObjectProp(
                        additional_properties=AnyOfProp(
                            any_of=[
                                StringProp(),
                                IntegerProp(),
                            ]
                        ),
                        title="Dv",
                    ),
                },
                required=["l", "d", "dv"],
                title="C",
            ),
            "e": AllOfProp(
                all_of=[
                    IntegerProp(
                        title="E",
                        enum=[0, 1],
                    )
                ],
                default=1,
            ),
            "v": AnyOfProp(
                any_of=[
                    IntegerProp(),
                    StringProp(),
                    NullProp(),
                ],
                title="V",
            ),
            "n": AnyOfProp(
                any_of=[
                    IntegerProp(),
                    NullProp(),
                ],
                title="N",
            ),
        },
        required=["i", "f", "b", "fi", "s", "c"],
        title="A",
    )


@pytest.fixture()
def expected_schema_instance():
    return {
        "i": 1,
        "f": 3.1415,
        "b": True,
        "fi": 1.0,
        "s": "hello",
        "c": {
            "l": [2, 3],
            "d": {"mellow": "fellow"},
            "dv": {"yellow": "jellow", "sellow": 5},
        },
        "e": E.Y,
        "v": 5,
        "n": None,
    }
