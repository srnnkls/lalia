from enum import StrEnum


class FinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    FUNCTION_CALL = "function_call"
    CONTENT_FILTER = "content_filter"
    DELEGATE = "delegate"
    NULL = "null"
    FUNCTION_CALL_FAILURE = "function_call_failure"
    FUNCTION_CALL_ERROR = "function_call_error"
    FAILURE = "failure"
