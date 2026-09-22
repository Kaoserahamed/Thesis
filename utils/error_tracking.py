"""Crash-tolerant error tracking for the thesis pipeline and services.

Collects unhandled / wrapped exceptions into a bounded in-memory ring buffer
and an optional JSON-lines sink, with de-duplication by *signature* so the
``/metrics`` endpoint stays concise.

A singleton tracker is lazily created and shared between
:mod:`utils.logging_framework`, :mod:`utils.health` and the
``scripts/serve_health`` HTTP service.

Environment
-----------
``THESIS_ERROR_SINK`` – path to a JSONL file for recorded errors.
``THESIS_ERROR_MAX_BUFFER`` – ring-buffer size (default 1000).
``THESIS_ERROR_WEBHOOK_URL`` – optional HTTPS endpoint receiving JSON errors.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union

F = TypeVar("F", bound=Callable[..., Any])

LOGGER = logging.getLogger("utils.error_tracking")

_ENV_SINK = "THESIS_ERROR_SINK"
_ENV_MAX_BUFFER = "THESIS_ERROR_MAX_BUFFER"
_ENV_WEBHOOK = "THESIS_ERROR_WEBHOOK_URL"
_DEFAULT_BUFFER = 1000
_WEBHOOK_TIMEOUT_SECONDS = 2.0


@dataclass
class ErrorRecord:
    """A single captured error instance."""

    id: str
    timestamp: float
    exc_type: str
    message: str
    signature: str
    tags: Dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None
    stack: Optional[str] = None


@dataclass
class TrackerConfig:
    """Mutable configuration for the singleton tracker."""

    sink: Optional[Path] = None
    webhook_url: Optional[str] = None
    max_buffer: int = _DEFAULT_BUFFER
    enabled: bool = True


def _record_to_dict(rec: ErrorRecord) -> Dict[str, Any]:
    return {
        "id": rec.id,
        "timestamp": rec.timestamp,
        "exc_type": rec.exc_type,
        "message": rec.message,
        "signature": rec.signature,
        "tags": rec.tags,
        "location": rec.location,
        "stack": rec.stack,
    }


class ErrorTracker:
    """Thread-safe collector for exceptions with signature-based de-duplication."""

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.config = config or TrackerConfig()
        self._buffer: deque = deque(maxlen=self.config.max_buffer or 0)
        self._signatures: Counter = Counter()
        self._lock = threading.RLock()
        self._sink_handle: Any = None
        if self.config.sink is not None:
            self._open_sink(self.config.sink)

    @property
    def total(self) -> int:
        """Total number of error records seen."""
        with self._lock:
            return sum(self._signatures.values())

    @property
    def unique(self) -> int:
        """Number of distinct error signatures seen."""
        with self._lock:
            return len(self._signatures)

    @property
    def buffer_size(self) -> int:
        """Number of records retained in the ring buffer."""
        with self._lock:
            return len(self._buffer)

    def configure(
        self,
        *,
        sink: Optional[Union[str, Path]] = None,
        max_buffer: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """Update the tracker configuration at runtime."""
        with self._lock:
            if sink is not None:
                self.config.sink = Path(sink) if sink else None
                if self.config.sink is not None:
                    self._open_sink(self.config.sink)
            if max_buffer is not None:
                self.config.max_buffer = max(0, int(max_buffer))
                new_buffer: deque = deque(maxlen=self.config.max_buffer)
                new_buffer.extend(self._buffer)
                self._buffer = new_buffer
            if enabled is not None:
                self.config.enabled = bool(enabled)

    def record(self, exc: BaseException, tags: Optional[Dict[str, Any]] = None) -> ErrorRecord:
        """Capture a single exception.

        ``exc`` is its ``__traceback__`` is used for source location.
        ``tags`` are optional key/value labels (e.g. ``{"event": "train.failed"}``).
        """
        if not self.config.enabled:
            return self._empty_record(exc, tags)

        record = self._build_record(exc, tags)
        with self._lock:
            self._buffer.append(record)
            self._signatures[record.signature] += 1
            self._write_jsonl(record)
            self._send_webhook(record)
        LOGGER.warning(
            "error tracked: %s",
            record.signature,
            extra={"fields": {"signature": record.signature}},
        )
        return record

    def _open_sink(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._sink_handle is not None:
            self._sink_handle.close()
        self._sink_handle = path.open("a", encoding="utf-8")

    def _build_record(self, exc: BaseException, tags: Optional[Dict[str, Any]]) -> ErrorRecord:
        location = None
        if exc.__traceback__ is not None:
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            location = f"{frame.filename}:{frame.lineno}"
        message = str(exc)
        signature_source = f"{type(exc).__module__}.{type(exc).__qualname__}:{message}:{location}"
        signature = hashlib.sha256(signature_source.encode("utf-8")).hexdigest()[:16]
        return ErrorRecord(
            id=uuid.uuid4().hex,
            timestamp=time.time(),
            exc_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            message=message,
            signature=signature,
            tags=dict(tags or {}),
            location=location,
            stack=(
                "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                if exc.__traceback__ is not None
                else None
            ),
        )

    def _empty_record(self, exc: BaseException, tags: Optional[Dict[str, Any]]) -> ErrorRecord:
        return ErrorRecord(
            id="",
            timestamp=time.time(),
            exc_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            message=str(exc),
            signature="disabled",
            tags=dict(tags or {}),
        )

    def _write_jsonl(self, record: ErrorRecord) -> None:
        if self._sink_handle is None:
            return
        self._sink_handle.write(json.dumps(_record_to_dict(record), default=str) + "\n")
        self._sink_handle.flush()

    def _send_webhook(self, record: ErrorRecord) -> None:
        """Best-effort delivery to an external JSON webhook."""
        if not self.config.webhook_url:
            return
        payload = json.dumps(_record_to_dict(record), default=str).encode("utf-8")
        request = urllib.request.Request(
            self.config.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_WEBHOOK_TIMEOUT_SECONDS):
                pass
        except (OSError, urllib.error.URLError) as exc:
            LOGGER.warning("external error webhook failed: %s", exc)

    def snapshot(self, limit: int = 100) -> Dict[str, Any]:
        """Return bounded tracker state suitable for a status endpoint."""
        with self._lock:
            records = list(self._buffer)[-max(0, int(limit)) :]
            return {
                "total": self.total,
                "unique": self.unique,
                "buffer_size": self.buffer_size,
                "records": [_record_to_dict(record) for record in records],
            }


_TRACKER: Optional[ErrorTracker] = None
_TRACKER_LOCK = threading.Lock()


def get_tracker() -> ErrorTracker:
    """Return the process-wide error tracker singleton."""
    global _TRACKER
    if _TRACKER is None:
        with _TRACKER_LOCK:
            if _TRACKER is None:
                sink = os.environ.get(_ENV_SINK)
                max_buffer = int(os.environ.get(_ENV_MAX_BUFFER, str(_DEFAULT_BUFFER)))
                _TRACKER = ErrorTracker(
                    TrackerConfig(
                        sink=Path(sink) if sink else None,
                        webhook_url=os.environ.get(_ENV_WEBHOOK),
                        max_buffer=max_buffer,
                    )
                )
    return _TRACKER


def snapshot(limit: int = 100) -> Dict[str, Any]:
    """Return a snapshot of the process-wide tracker."""
    return get_tracker().snapshot(limit=limit)


def as_metrics_text() -> str:
    """Render tracker counters in Prometheus exposition format."""
    tracker = get_tracker()
    return (
        f"thesis_errors_total {tracker.total}\n"
        f"thesis_errors_unique {tracker.unique}\n"
        f"thesis_errors_buffer_size {tracker.buffer_size}\n"
    )


__all__ = [
    "ErrorRecord",
    "ErrorTracker",
    "TrackerConfig",
    "as_metrics_text",
    "get_tracker",
    "snapshot",
]
