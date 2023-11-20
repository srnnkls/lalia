import logging
import re
from collections.abc import Iterator
from enum import StrEnum
from typing import Any, ClassVar

from rich.logging import RichHandler
from rich.pretty import pretty_repr

import lalia


class MsgFormat(StrEnum):
    RICH = "%(message)s"
    PLAIN = "%(asctime)s %(name)s %(levelname)s\n %(message)s"


class DateFormat(StrEnum):
    RICH = "[%X]"
    PLAIN = "[%X]"


class FormatTypeSpec(StrEnum):
    STRING = "s"
    STRING_UPPERCASE = "S"
    CHAR = "c"
    CHAR_UPPERCASE = "C"
    DECIMAL_INTEGER = "d"
    INTEGER = "i"
    OCTAL_INTEGER = "o"
    UNSIGNED_DECIMAL_INTEGER = "u"
    HEXADECIMAL_INTEGER = "x"
    HEXADECIMAL_INTEGER_UPPERCASE = "X"
    EXPONENTIAL_NOTATION = "e"
    EXPONENTIAL_NOTATION_UPPERCASE = "E"
    FLOATING_POINT = "f"
    GENERAL_FORMAT = "g"
    GENERAL_FORMAT_UPPERCASE = "G"
    SIGNED_HEX = "a"
    SIGNED_HEX_UPPERCASE = "A"
    POINTER_TO_INTEGER = "n"
    POINTER = "p"
    REPRESANTATION = "r"

    @classmethod
    def to_character_class(cls) -> str:
        specs = "".join(spec for spec in cls)
        return f"[{specs}]"


"""
A regex to match C-style format strings, e.g. "%s %d".
Adapted from https://stackoverflow.com/a/30018957.
"""
C_FORMAT_STRING_REGEX = re.compile(
    rf"""
    %                                                  # literal "%"
    (?:                                                # first option: a format spec
    (?:[-+0 #]{{0,5}})                                 # optional flags
    (?:\d+|\*)?                                        # width
    (?:\.(?:\d+|\*))?                                  # precision
    (?:h|l|ll|w|I|I32|I64)?                            # size
    ({FormatTypeSpec.to_character_class()})            # type
    ) |                                                # OR
    %%                                                 # second option: literal "%%"
    """,
    re.VERBOSE,
)


class LoggerRegistry:
    _loggers: ClassVar[dict[str, logging.Logger]] = {}

    @classmethod
    def register(cls, name: str, logger: logging.Logger):
        cls._loggers[name] = logger

    @classmethod
    def get_logger(
        cls, name: str, handler_type: type[logging.Handler]
    ) -> logging.Logger:
        if name not in cls._loggers:
            return _create_logger(name, handler_type)

        logger = cls._loggers[name]

        if handler_type in map(type, logger.handlers):
            return logger

        handler = init_handler(handler_type)
        logger.addHandler(handler)

        return logger


class LogRecord(logging.LogRecord):
    def getMessage(self) -> str:
        def prettify_str_spec_args(message: str, args: Any) -> Iterator[str | object]:
            format_type_specs = (
                match.group(1) for match in C_FORMAT_STRING_REGEX.finditer(message)
            )
            for spec, arg in zip(format_type_specs, args, strict=True):
                if spec == FormatTypeSpec.STRING:
                    yield pretty_repr(arg)
                else:
                    yield arg

        if isinstance(self.msg, str) and self.args:
            args_formatted = prettify_str_spec_args(self.msg, self.args)
            return self.msg % tuple(args_formatted)

        if isinstance(self.msg, str):
            return self.msg

        return pretty_repr(self.msg)


logging.setLogRecordFactory(LogRecord)


def _init_rich_handler(
    rich_hanlder_type: type[RichHandler], *args, **kwargs
) -> RichHandler:
    formatter = logging.Formatter(MsgFormat.RICH, datefmt=DateFormat.RICH)

    handler = rich_hanlder_type(*args, rich_tracebacks=True, **kwargs)
    handler.setFormatter(formatter)

    return handler


def _init_stream_handler(
    stream_handler_type: type[logging.StreamHandler], *args, **kwargs
) -> logging.StreamHandler:
    formatter = logging.Formatter(MsgFormat.PLAIN, datefmt=DateFormat.PLAIN)

    handler = stream_handler_type(*args, **kwargs)
    handler.setFormatter(formatter)

    return handler


def init_handler(
    handler_type: type[logging.Handler], *args, **kwargs
) -> logging.Handler:
    """
    Initialize a handler of the given type.
    """
    if issubclass(handler_type, RichHandler):
        return _init_rich_handler(handler_type, *args, **kwargs)
    if issubclass(handler_type, logging.StreamHandler):
        return _init_stream_handler(handler_type, *args, **kwargs)
    raise ValueError(f"Unknown handler type {type(handler_type)}")


def _create_logger(
    name: str, handler_type: type[logging.Handler] = RichHandler
) -> logging.Logger:
    """
    Create a logger with the given name set up with a RichHandler.
    """
    logger = logging.getLogger(name)

    handler = init_handler(handler_type)
    logger.addHandler(handler)

    logger.propagate = False

    return logger


def get_logger(
    name: str, handler_type: type[logging.Handler] = RichHandler
) -> logging.Logger:
    """
    Get a logger with the given name set up with a RichHandler.
    """

    return LoggerRegistry.get_logger(name, handler_type)


def list_loggers() -> list[str]:
    """
    List all registered loggers.
    """
    return list(LoggerRegistry._loggers)


root_logger = get_logger(lalia.__name__)
