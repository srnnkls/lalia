from enum import StrEnum


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"

    @property
    def color(self) -> str:
        return ROLE_TO_COLOR[self]


ROLE_TO_COLOR = {
    Role.SYSTEM: "red",
    Role.USER: "orange1",
    Role.ASSISTANT: "deep_sky_blue1",
    Role.FUNCTION: "magenta",
}
