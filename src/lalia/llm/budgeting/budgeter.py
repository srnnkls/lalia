from enum import StrEnum

from pydantic.dataclasses import dataclass
from tiktoken import encoding_name_for_model, get_encoding

from lalia.llm.models import ChatModel


class Encoding(StrEnum):
    CL100K_BASE = "cl100k_base"

    @classmethod
    def from_model(cls, model: ChatModel | str) -> str:
        try:
            encoding_name = encoding_name_for_model(model)
            return cls(encoding_name)
        except KeyError:
            raise ValueError(f"Unsupported model: {model}") from KeyError


@dataclass
class Encoder:
    encoding_name: str = Encoding.CL100K_BASE

    def __post_init__(self):
        self.encoder = get_encoding(self.encoding_name)

    @classmethod
    def from_model(cls, model: ChatModel | str) -> "Encoder":
        return cls(Encoding.from_model(model))

    def encode(self, text: str) -> list[int]:
        try:
            return self.encoder.encode(text)
        except Exception as e:
            raise ValueError(f"Encoding failed with error: {e}") from e

    def decode(self, tokens: list[int]) -> str:
        try:
            return self.encoder.decode(tokens)
        except Exception as e:
            raise ValueError(f"Decoding failed with error: {e}") from e
