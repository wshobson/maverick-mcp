"""Tests for maverick.platform.telemetry."""

import io
import json
import logging

from maverick.platform.config import TelemetrySettings
from maverick.platform.telemetry import (
    StructuredFormatter,
    get_logger,
    new_request_id,
    reset_logging,
    set_request_id,
    setup_logging,
)


def _capture_one(logger_name: str, message: str, **extra) -> dict:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredFormatter())
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        logger.info(message, extra=extra)
    finally:
        logger.removeHandler(handler)
    return json.loads(stream.getvalue())


def test_json_shape():
    record = _capture_one("maverick.test", "hello")
    assert record["message"] == "hello"
    assert record["logger"] == "maverick.test"
    assert record["level"] == "INFO"
    assert "timestamp" in record
    assert record["module"] == "test_telemetry"


def test_extra_fields_pass_through():
    record = _capture_one("maverick.test", "hi", ticker="AAPL")
    assert record["ticker"] == "AAPL"


def test_sensitive_fields_are_masked():
    record = _capture_one("maverick.test", "auth", api_key="sk-123", password="x")
    assert record["api_key"] == "***"
    assert record["password"] == "***"


def test_sensitive_fields_are_masked_case_insensitively():
    record = _capture_one("maverick.test", "auth", API_KEY="x")
    assert record["API_KEY"] == "***"


def test_request_id_included_when_set():
    rid = new_request_id()
    set_request_id(rid)
    try:
        record = _capture_one("maverick.test", "traced")
        assert record["request_id"] == rid
    finally:
        set_request_id(None)


def test_exception_block():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(StructuredFormatter())
    logger = logging.getLogger("maverick.exc")
    logger.addHandler(handler)
    logger.setLevel(logging.ERROR)
    try:
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("failed")
    finally:
        logger.removeHandler(handler)
    record = json.loads(stream.getvalue())
    assert record["exception"]["type"] == "ValueError"
    assert "boom" in record["exception"]["message"]


def test_setup_logging_defaults_to_stderr(capsys):
    try:
        setup_logging(TelemetrySettings(log_level="INFO", json_logs=True))
        get_logger("maverick.setup").info("to stderr")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "to stderr" in captured.err
    finally:
        reset_logging()


def test_setup_logging_is_idempotent():
    try:
        setup_logging(TelemetrySettings(log_level="INFO", json_logs=True))
        setup_logging(TelemetrySettings(log_level="INFO", json_logs=True))
        logger = logging.getLogger("maverick")
        assert len(logger.handlers) == 1
    finally:
        reset_logging()
