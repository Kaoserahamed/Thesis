"""Structured logging framework for the thesis pipeline and services.

One :func:`configure_logging` call switches the ``utils``/script loggers to
JSON-lines or human-readable output; :func:`bind_context` and :func:`log_step`
attach run/stage/step fields to every record via :mod:`contextvars`, and
``@log_execution`` traces a callable's start, duration and outcome.

Environment overrides (one configuration for notebooks, scripts and CI):
``THESIS_LOG_LEVEL``, ``THESIS_LOG_JSON`` and ``THESIS_LOG_FILE``.
"""

from __future__ import annotations

import inspect
import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, TypeVar, Union, cast

from . import error_tracking

LOGGER_NAMESPACE = "utils"
LEVEL_ENV = "THESIS_LOG_LEVEL"
JSON_ENV = "THESIS_LOG_JSON"
FILE_ENV = "THESIS_LOG_FILE"

F = TypeVar("F", bound=Callable[..., Any])

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("thesis_log_context", default={})
_STATE: Dict[str, Any] = {"namespace": LOGGER_NAMESPACE, "signature": None}
_LEVEL_BY_SEVERITY = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

# Every attribute a plain LogRecord carries; anything else is a structured field.
_RESERVED = frozenset(
    set(vars(logging.LogRecord("", logging.INFO, "", 0, "", (), None)))
    | {"message", "asctime", "fields", "exc_text"}
)
_PAYLOAD_KEYS = frozenset({"timestamp", "level", "logger", "message", "exception"})


class _OwnedStreamHandler(logging.StreamHandler):
    """Stream handler installed by this module (removable by :func:`reset_logging`)."""


class _OwnedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating JSON-lines file handler installed by this module."""


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean-ish environment variable (``1/true/yes/on``)."""
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() in _TRUTHY


def new_run_id() -> str:
    """Short unique id that correlates every record of one execution."""
    return uuid.uuid4().hex[:12]


def _iso_timestamp(created: float) -> str:
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(created))
    return f"{base}.{int((created % 1) * 1000):03d}Z"


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def current_context() -> Dict[str, Any]:
    """Fields bound to the current thread/task (a copy, safe to mutate)."""
    return dict(_CONTEXT.get())


def set_context(**fields: Any) -> None:
    """Merge ``fields`` into the context attached to all later records."""
    merged = current_context()
    merged.update({key: value for key, value in fields.items() if value is not None})
    _CONTEXT.set(merged)


def clear_context() -> None:
    """Drop every bound context field."""
    _CONTEXT.set({})


@contextmanager
def log_context(**fields: Any) -> Iterator[Dict[str, Any]]:
    """Temporarily bind context fields (restored on exit, even on error)."""
    token = _CONTEXT.set({**current_context(), **fields})
    try:
        yield current_context()
    finally:
        _CONTEXT.reset(token)


def _record_fields(record: logging.LogRecord) -> Dict[str, Any]:
    """Bound context plus any ``extra=`` fields carried by the record."""
    fields = current_context()
    extra = getattr(record, "fields", None)
    if isinstance(extra, Mapping):
        fields.update(extra)
    for key, value in record.__dict__.items():
        if key not in _RESERVED and not key.startswith("_"):
            fields.setdefault(key, value)
    return fields


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object (JSON-lines friendly)."""

    def __init__(self, static_fields: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        self._static = dict(static_fields or {})

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": _iso_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(self._static)
        payload.update(
            {
                key: value
                for key, value in _record_fields(record).items()
                if key not in _PAYLOAD_KEYS
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Single-line human-readable formatter: ``time | level | logger | msg | k=v``."""

    def __init__(self, static_fields: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__()
        self._static = dict(static_fields or {})

    def format(self, record: logging.LogRecord) -> str:
        parts = [
            _iso_timestamp(record.created),
            f"{record.levelname:<8}",
            record.name,
            record.getMessage(),
        ]
        fields = {**self._static, **_record_fields(record)}
        if fields:
            parts.append(" ".join(f"{key}={value}" for key, value in fields.items()))
        if record.exc_info:
            parts.append("\n" + self.formatException(record.exc_info))
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _resolve_level(level: Union[int, str, None]) -> int:
    if level is None:
        level = os.environ.get(LEVEL_ENV, "INFO")
    if isinstance(level, str):
        resolved = logging.getLevelName(level.strip().upper())
        if not isinstance(resolved, int):
            raise ValueError(f"Unknown log level: {level!r}")
        return resolved
    return int(level)


def _target_logger(namespace: str) -> logging.Logger:
    return logging.getLogger(namespace) if namespace else logging.getLogger()


def _env_path() -> Optional[Path]:
    raw = os.environ.get(FILE_ENV)
    return Path(raw) if raw else None


def configure_logging(
    level: Union[int, str, None] = None,
    *,
    json_output: Optional[bool] = None,
    log_file: Union[str, Path, None] = None,
    namespace: Optional[str] = None,
    run_id: Optional[str] = None,
    stream: Optional[Any] = None,
    propagate: bool = True,
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
    force: bool = False,
) -> logging.Logger:
    """Configure structured logging once per process (idempotent).

    ``level`` defaults to ``$THESIS_LOG_LEVEL`` (INFO), ``json_output`` to
    ``$THESIS_LOG_JSON`` and ``log_file`` to ``$THESIS_LOG_FILE``.  The file sink
    always receives JSON-lines records; the stream sink is text unless
    ``json_output`` is set.  Repeating the same call is a no-op, ``force=True``
    rebuilds the handlers, and a short ``run_id`` is bound into the context so
    every record of one execution can be correlated.
    """
    resolved_level = _resolve_level(level)
    resolved_json = env_flag(JSON_ENV) if json_output is None else bool(json_output)
    resolved_file = Path(log_file) if log_file is not None else _env_path()
    resolved_namespace = namespace if namespace is not None else str(_STATE["namespace"])
    resolved_run_id = run_id or current_context().get("run_id") or new_run_id()
    signature = (
        resolved_namespace,
        resolved_level,
        resolved_json,
        str(resolved_file) if resolved_file else "",
        resolved_run_id,
        propagate,
    )

    logger = _target_logger(resolved_namespace)
    if not force and _STATE["signature"] == signature:
        return logger

    for handler in list(logger.handlers):
        if isinstance(handler, (_OwnedStreamHandler, _OwnedRotatingFileHandler)):
            logger.removeHandler(handler)
            handler.close()

    static_fields = {
        "run_id": resolved_run_id,
        "pid": os.getpid(),
        "service": resolved_namespace or "root",
    }
    stream_handler = _OwnedStreamHandler(stream if stream is not None else sys.stderr)
    stream_handler.setFormatter(
        JsonFormatter(static_fields) if resolved_json else TextFormatter(static_fields)
    )
    logger.addHandler(stream_handler)

    if resolved_file is not None:
        if str(resolved_file.parent) not in ("", "."):
            resolved_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = _OwnedRotatingFileHandler(
            str(resolved_file),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JsonFormatter({**static_fields, "sink": "file"}))
        logger.addHandler(file_handler)

    logger.setLevel(resolved_level)
    logger.propagate = propagate
    _STATE.update(namespace=resolved_namespace, signature=signature)
    set_context(run_id=resolved_run_id)

    logger.info(
        "logging configured",
        extra={
            "fields": {
                "event": "logging.configured",
                "json": resolved_json,
                "level": logging.getLevelName(resolved_level),
                "log_file": str(resolved_file) if resolved_file else None,
            }
        },
    )
    return logger


def reset_logging(namespace: Optional[str] = None) -> None:
    """Remove the framework handlers and clear context (used by tests)."""
    for name in {namespace or str(_STATE["namespace"]), LOGGER_NAMESPACE}:
        logger = _target_logger(name)
        for handler in list(logger.handlers):
            if isinstance(handler, (_OwnedStreamHandler, _OwnedRotatingFileHandler)):
                logger.removeHandler(handler)
                handler.close()
    _STATE["signature"] = None
    _STATE["namespace"] = LOGGER_NAMESPACE
    clear_context()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger inside the configured namespace.

    ``get_logger()`` returns the namespace logger, ``get_logger("health")``
    returns ``<namespace>.health``, so records always reach the handlers that
    :func:`configure_logging` installed.
    """
    namespace = str(_STATE["namespace"])
    if not name:
        return _target_logger(namespace)
    if not namespace or name == namespace or name.startswith(f"{namespace}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{namespace}.{name}")


# ---------------------------------------------------------------------------
# Structured tracing helpers
# ---------------------------------------------------------------------------


@contextmanager
def log_step(
    name: str,
    *,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    **fields: Any,
) -> Iterator[logging.Logger]:
    """Log the start, duration and outcome of a named pipeline step."""
    log = logger if logger is not None else get_logger("step")
    started = time.perf_counter()
    with log_context(step=name, **fields):
        log.log(level, "step started", extra={"fields": {"event": "step.started"}})
        try:
            yield log
        except Exception as exc:
            log.error(
                "step failed",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={"fields": {"event": "step.failed", "duration_ms": _ms(started)}},
            )
            raise
        else:
            log.log(
                level,
                "step finished",
                extra={"fields": {"event": "step.finished", "duration_ms": _ms(started)}},
            )


@contextmanager
def _execution_span(
    label: str,
    logger: Optional[logging.Logger],
    level: int,
    fields: Mapping[str, Any],
) -> Iterator[None]:
    log = logger if logger is not None else get_logger("execution")
    started = time.perf_counter()
    log.log(level, "started", extra={"fields": {"event": f"{label}.started", **fields}})
    try:
        yield
    except Exception as exc:
        error_tracking.get_tracker().record(exc, tags={"event": f"{label}.failed", **fields})
        log.error(
            "failed",
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={"fields": {"event": f"{label}.failed", "duration_ms": _ms(started), **fields}},
        )
        raise
    else:
        log.log(
            level,
            "finished",
            extra={"fields": {"event": f"{label}.finished", "duration_ms": _ms(started), **fields}},
        )


def log_execution(
    event: Optional[str] = None,
    *,
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    **fields: Any,
) -> Callable[[F], F]:
    """Trace a callable: ``started``/``finished``/``failed`` plus ``duration_ms``.

    Works for sync and async callables, re-raises the original exception and
    reports failures to :mod:`utils.error_tracking` so they also appear in the
    ``/metrics`` endpoint.
    """
    label = event

    def decorator(func: F) -> F:
        resolved = label or f"{func.__module__}.{func.__qualname__}"

        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with _execution_span(resolved, logger, level, fields):
                    return await func(*args, **kwargs)

            return cast(F, async_wrapper)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _execution_span(resolved, logger, level, fields):
                return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def logging_snapshot() -> Dict[str, Any]:
    """Effective logging configuration (exposed by the ``/version`` endpoint)."""
    logger = _target_logger(str(_STATE["namespace"]))
    return {
        "namespace": _STATE["namespace"],
        "level": logging.getLevelName(logger.level),
        "handlers": [type(handler).__name__ for handler in logger.handlers],
        "propagate": logger.propagate,
        "context": current_context(),
    }
