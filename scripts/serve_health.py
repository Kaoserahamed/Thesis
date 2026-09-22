"""HTTP health endpoint service (stdlib only, no extra dependencies).

Serves the :mod:`utils.health` registry as Prometheus-style endpoints so the
pipeline can be monitored or probed from CI:

    GET /health      – aggregate status (200/206/503)
    GET /ready       – readiness probe (critical checks only)
    GET /status      – per-check detail
    GET /metrics     – Prometheus exposition (errors + health)
    GET /version     – project version + logging snapshot

    --once           – run a single self-probe and exit (for CI smoke tests)

Usage::

    python scripts/serve_health.py            # serve on :8080
    python scripts/serve_health.py --port 9090
    python scripts/serve_health.py --once     # CI self-probe
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict

# Make ``utils`` importable when this script is run directly (not installed).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import __version__  # noqa: E402
from utils import error_tracking  # noqa: E402
from utils import health as health_mod  # noqa: E402
from utils.logging_framework import logging_snapshot  # noqa: E402

_STATUS_CODE = {
    health_mod.Status.HEALTHY: 200,
    health_mod.Status.DEGRADED: 206,
    health_mod.Status.UNHEALTHY: 503,
}


class HealthHandler(BaseHTTPRequestHandler):
    """Route handler for health + observability endpoints."""

    # quiet down default logging; we do structured logging ourselves
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        pass

    def _send(self, status_code: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body, default=str).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        routes: Dict[str, Callable[[], None]] = {
            "/health": self._health,
            "/ready": self._ready,
            "/status": self._status,
            "/metrics": self._metrics,
            "/version": self._version,
        }
        path = self.path.split("?", 1)[0]
        handler = routes.get(path)
        if handler is None:
            self._send(404, {"error": "not found", "path": path})
            return
        handler()

    def _health(self) -> None:
        report = health_mod.run_health(__version__)
        code = _STATUS_CODE[report.overall_status]
        self._send(
            code,
            {
                "status": report.overall_status.name.lower(),
                "healthy": report.is_healthy,
                "checks_run": len(report.checks),
            },
        )

    def _ready(self) -> None:
        reg = health_mod.get_registry()
        # Readiness = only critical checks must be healthy.
        critical_only = _HealthRegistry()
        for name in reg.names:
            check = reg.get(name)
            if check is not None and check.critical:
                critical_only.register(check)
        if critical_only.names:
            report = critical_only.run_all(version=__version__)
        else:
            report = health_mod.run_health(__version__)
        code = _STATUS_CODE[report.overall_status]
        self._send(code, {"ready": report.is_healthy, "checks": len(report.checks)})

    def _status(self) -> None:
        report = health_mod.run_health(__version__)
        self._send(_STATUS_CODE[report.overall_status], report.to_dict())

    def _metrics(self) -> None:
        body = error_tracking.as_metrics_text()
        # Add a health metric.
        report = health_mod.run_health(__version__)
        healthy_val = 1 if report.is_healthy else 0
        payload = body + f"thesis_health_status {healthy_val}\n"
        encoded = payload.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _version(self) -> None:
        self._send(
            200,
            {
                "version": __version__,
                "python": sys.version.split()[0],
                "logging": logging_snapshot(),
                "errors": error_tracking.snapshot(limit=10),
            },
        )


# Local alias so _ready() can build a throwaway registry.
_HealthRegistry = health_mod.HealthRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve health endpoints.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default 8080)")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single self-probe and exit (for CI).",
    )
    parser.add_argument(
        "--require-healthy",
        action="store_true",
        help="--once: exit non-zero even on DEGRADED (strict mode).",
    )
    args = parser.parse_args()

    if args.once:
        report = health_mod.run_health(__version__)
        status = report.overall_status.name.lower()
        print(json.dumps(report.to_dict(), default=str, indent=2))
        print(f"\nOverall: {status}")
        # DEGRADED is OK by default (e.g. TensorFlow not importable in CI);
        # only UNHEALTHY fails unless --require-healthy is set.
        if report.overall_status is health_mod.Status.UNHEALTHY:
            return 1
        if args.require_healthy and not report.is_healthy:
            return 1
        return 0

    server = ThreadingHTTPServer((args.host, args.port), HealthHandler)
    print(f"Health service: http://{args.host}:{args.port}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
