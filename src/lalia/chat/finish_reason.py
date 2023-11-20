from enum import StrEnum


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    CONTENT_FILTER = "content_filter"
    DELEGATE = "delegate"
    NULL = "null"
    ERROR = "error"
