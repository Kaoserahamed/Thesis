"""Focused behavior tests for the structured logging framework."""

from __future__ import annotations

import io
import json

from utils.logging_framework import (
    configure_logging,
    get_logger,
    log_context,
    log_execution,
    logging_snapshot,
    reset_logging,
)


def teardown_function():
    reset_logging()


def test_json_logging_preserves_envelope_and_context():
    stream = io.StringIO()
    configure_logging(json_output=True, stream=stream, force=True, run_id="run-test")

    with log_context(stage="preprocess"):
        get_logger("test").info(
            "hello",
            extra={"fields": {"event": "test.message", "message": "not-an-override"}},
        )

    payload = json.loads(stream.getvalue().splitlines()[-1])
    assert payload["message"] == "hello"
    assert payload["run_id"] == "run-test"
    assert payload["stage"] == "preprocess"
    assert payload["event"] == "test.message"


def test_reset_restores_default_namespace():
    configure_logging(namespace="custom", stream=io.StringIO(), force=True)
    reset_logging("custom")

    snapshot = logging_snapshot()
    assert snapshot["namespace"] == "utils"


def test_log_execution_reraises_original_exception():
    stream = io.StringIO()
    configure_logging(stream=stream, force=True)

    @log_execution("test.operation")
    def fail():
        raise ValueError("expected")

    try:
        fail()
    except ValueError:
        pass
    else:
        raise AssertionError("log_execution must re-raise the original exception")

    assert "test.operation" in stream.getvalue()


def test_error_tracker_posts_to_configured_webhook(monkeypatch):
    from utils.error_tracking import ErrorTracker, TrackerConfig

    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    tracker = ErrorTracker(TrackerConfig(webhook_url="https://monitor.example/errors"))
    tracker.record(ValueError("webhook smoke"), tags={"event": "test"})

    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url == "https://monitor.example/errors"
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 2.0
