from enum import StrEnum


class ChatModel(StrEnum):
    GPT_3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_4_0613 = "gpt-4-0613"

    @property
    def token_limit(self) -> int:
        return TOKEN_LIMITS[self]


TOKEN_LIMITS = {
    ChatModel.GPT_3_5_TURBO_0613: 4096,
    ChatModel.GPT_4_0613: 8192,
}


class FunctionCallDirective(StrEnum):
    NONE = "none"
    AUTO = "auto"
