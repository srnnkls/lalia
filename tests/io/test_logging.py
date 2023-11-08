import io
import logging
import time

import pytest

from lalia.io.logging import DateFormat, MsgFormat, get_logger, init_handler


@pytest.fixture
def buffered_logger():
    log_buffer = io.StringIO()

    logger = get_logger(__name__)
    logger.setLevel(logging.DEBUG)

    stream_handler = init_handler(logging.StreamHandler, stream=log_buffer)
    logger.addHandler(stream_handler)

    logger.propagate = False

    yield logger, log_buffer

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_logging_to_buffer(buffered_logger):
    logger, log_buffer = buffered_logger

    test_message = "This is a test log message"
    timestamp = time.strftime(DateFormat.PLAIN, time.localtime())
    logger.debug(test_message)

    log_buffer.seek(0)

    log_output = log_buffer.getvalue().strip()

    expexted = MsgFormat.PLAIN % {
        "asctime": timestamp,
        "name": __name__,
        "levelname": "DEBUG",
        "message": test_message,
    }

    assert log_output == expexted
