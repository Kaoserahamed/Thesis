"""Health-check registry with built-in and pluggable probes.

A :class:`HealthCheck` is a small callable returning a :class:`HealthResult`.
The default :data:`registry` ships with checks for Python, config, data, disk,
dependencies and TensorFlow — each isolated so a single failing probe can
never crash the whole report (important when TF is broken, as it currently is
in this environment).

HTTP exposure lives in ``scripts/serve_health.py``.
"""

from __future__ import annotations

import enum
import importlib
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

from . import error_tracking  # noqa: F401  (kept for future error-correlation)


class Status(enum.Enum):
    """Health-check outcome (ordered by severity)."""

    UNHEALTHY = 0
    DEGRADED = 1
    HEALTHY = 2

    def __bool__(self) -> bool:
        return self is Status.HEALTHY


@dataclass
class HealthResult:
    """Outcome of a single check."""

    status: Status
    message: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.name.lower(),
            "message": self.message,
            **(self.data or {}),
        }


class HealthCheck:
    """A named health probe.

    Parameters
    ----------
    name:
        Identifier used in the report (no spaces).
    description:
        Human-readable summary.
    critical:
        If ``True`` a non-healthy result makes the *overall* status unhealthy
        rather than merely degraded.
    fn:
        Zero-arg callable returning a :class:`HealthResult`.
    """

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[[], HealthResult],
        *,
        critical: bool = False,
        category: str = "system",
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self.critical = critical
        self.category = category

    def execute(self) -> HealthResult:
        """Run the probe, trapping any unexpected error."""
        try:
            return self.fn()
        except Exception as exc:  # noqa: BLE001 — checks must never crash the report
            return HealthResult(
                Status.UNHEALTHY,
                message=f"check raised {type(exc).__name__}: {exc}",
            )


@dataclass
class CheckReport:
    """Per-check report entry."""

    name: str
    category: str
    critical: bool
    result: HealthResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "critical": self.critical,
            **self.result.to_dict(),
        }


@dataclass
class HealthReport:
    """Aggregate health state."""

    timestamp: float
    overall_status: Status
    checks: List[CheckReport] = field(default_factory=list)
    version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_status": self.overall_status.name.lower(),
            "version": self.version,
            "checks": [c.to_dict() for c in self.checks],
        }

    @property
    def is_healthy(self) -> bool:
        return self.overall_status is Status.HEALTHY


class HealthRegistry:
    """Ordered collection of :class:`HealthCheck` probes."""

    def __init__(self) -> None:
        self._checks: Dict[str, HealthCheck] = {}

    def register(self, check: HealthCheck) -> None:
        self._checks[check.name] = check

    def unregister(self, name: str) -> None:
        self._checks.pop(name, None)

    @property
    def names(self) -> List[str]:
        return list(self._checks.keys())

    def get(self, name: str) -> Optional[HealthCheck]:
        return self._checks.get(name)

    def run_all(self, version: str = "") -> HealthReport:
        """Execute every registered check and aggregate the results."""
        overall = Status.HEALTHY
        reports: List[CheckReport] = []
        for check in self._checks.values():
            result = check.execute()
            reports.append(
                CheckReport(
                    name=check.name,
                    category=check.category,
                    critical=check.critical,
                    result=result,
                )
            )
            if result.status is Status.UNHEALTHY:
                overall = Status.UNHEALTHY
            elif result.status is Status.DEGRADED and overall is not Status.UNHEALTHY:
                overall = Status.DEGRADED
        # Critical checks that aren't HEALTHY force UNHEALTHY.
        for rep in reports:
            if rep.critical and rep.result.status is not Status.HEALTHY:
                if overall is not Status.UNHEALTHY:
                    overall = Status.UNHEALTHY
        return HealthReport(
            timestamp=time.time(),
            overall_status=overall,
            checks=reports,
            version=version,
        )


# ---------------------------------------------------------------------------
# Built-in checks
# ---------------------------------------------------------------------------

_PYTHON_MIN = (3, 10)
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _python_check() -> HealthResult:
    ok = sys.version_info >= _PYTHON_MIN
    return HealthResult(
        Status.HEALTHY if ok else Status.UNHEALTHY,
        f"Python {sys.version.split()[0]} (need >=3.10)",
        data={"version": sys.version.split()[0], "executable": sys.executable},
    )


def _config_check() -> HealthResult:
    try:
        with open(_CONFIG_PATH, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if not isinstance(cfg, dict):
            return HealthResult(Status.UNHEALTHY, "config root is not a mapping")
        project = cfg.get("project", {})
        return HealthResult(
            Status.HEALTHY,
            "config.yaml loaded",
            data={"name": project.get("name", "?"), "version": project.get("version", "?")},
        )
    except FileNotFoundError:
        return HealthResult(Status.UNHEALTHY, f"config.yaml not found at {_CONFIG_PATH}")
    except yaml.YAMLError as exc:
        return HealthResult(Status.UNHEALTHY, f"config.yaml parse error: {exc}")
    except Exception as exc:  # noqa: BLE001
        return HealthResult(Status.DEGRADED, f"config check error: {type(exc).__name__}: {exc}")


def _data_check() -> HealthResult:
    if not _DATA_ROOT.exists():
        return HealthResult(
            Status.DEGRADED,
            f"data root {_DATA_ROOT} does not exist (may not be downloaded yet)",
            data={"exists": False},
        )
    missing: List[str] = []
    for sub in ("raw", "processed"):
        if not (_DATA_ROOT / sub).exists():
            missing.append(sub)
    status = Status.HEALTHY if not missing else Status.DEGRADED
    return HealthResult(
        status,
        "all data dirs present" if not missing else f"missing: {missing}",
        data={"exists": True, "missing": missing},
    )


def _disk_check() -> HealthResult:
    try:
        usage = shutil.disk_usage(".")
        free_gb = usage.free / (1024**3)
        threshold_gb = float(os.environ.get("THESIS_DISK_FREE_GB", "5"))
        status = Status.HEALTHY if free_gb >= threshold_gb else Status.UNHEALTHY
        return HealthResult(
            status,
            f"{free_gb:.1f} GB free (need >= {threshold_gb:.0f} GB)",
            data={"free_gb": round(free_gb, 2), "threshold_gb": threshold_gb},
        )
    except Exception as exc:  # noqa: BLE001
        return HealthResult(Status.DEGRADED, f"disk check failed: {exc}")


def _dependencies_check() -> HealthResult:
    required = ["numpy", "pandas", "scipy", "yaml"]
    missing: List[str] = []
    found: Dict[str, str] = {}
    for mod in required:
        try:
            m = importlib.import_module(mod)
            found[mod] = getattr(m, "__version__", "?")
        except ImportError:
            missing.append(mod)
    status = Status.HEALTHY if not missing else Status.UNHEALTHY
    return HealthResult(
        status,
        "all core deps present" if not missing else f"missing: {missing}",
        data={"found": found, "missing": missing},
    )


def _tensorflow_check() -> HealthResult:
    """Probe TensorFlow via a subprocess so a broken TF can never crash us."""
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import tensorflow as tf; print(tf.__version__); " "tf.constant([[1,2],[3,4]])",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3"},
        )
        if proc.returncode == 0:
            version = (proc.stdout.strip().splitlines() or ["?"])[-1]
            return HealthResult(
                Status.HEALTHY,
                f"TensorFlow {version} imports and runs a tensor op",
                data={"version": version},
            )
        first_line = (proc.stderr.strip().splitlines() or ["?"])[0][:200]
        return HealthResult(
            Status.DEGRADED,
            f"TensorFlow failed to import (rc={proc.returncode}): {first_line}",
            data={"returncode": proc.returncode, "error": first_line},
        )
    except subprocess.TimeoutExpired:
        return HealthResult(Status.DEGRADED, "TensorFlow import timed out (30 s)")
    except FileNotFoundError:
        return HealthResult(Status.DEGRADED, "Python executable not found for probe")


def _hostname_check() -> HealthResult:
    host = socket.gethostname()
    return HealthResult(Status.HEALTHY, host, data={"hostname": host})


# ---------------------------------------------------------------------------
# Registry bootstrap
# ---------------------------------------------------------------------------


def _build_default_registry() -> HealthRegistry:
    reg = HealthRegistry()
    reg.register(HealthCheck("python", "Python runtime", _python_check, category="runtime"))
    reg.register(HealthCheck("config", "Config file loadable", _config_check, category="config"))
    reg.register(HealthCheck("data", "Data directory structure", _data_check, category="data"))
    reg.register(HealthCheck("disk", "Disk space", _disk_check, category="system"))
    reg.register(
        HealthCheck("dependencies", "Core deps importable", _dependencies_check, category="deps")
    )
    reg.register(
        HealthCheck("tensorflow", "TensorFlow probe (subprocess)", _tensorflow_check, category="ml")
    )
    reg.register(HealthCheck("hostname", "Hostname", _hostname_check, category="system"))
    return reg


registry: HealthRegistry = _build_default_registry()


def get_registry() -> HealthRegistry:
    """Return the module-level :class:`HealthRegistry` singleton."""
    return registry


def run_health(version: str = "") -> HealthReport:
    """Run all default checks and return the aggregated report."""
    return registry.run_all(version=version)


__all__ = [
    "Status",
    "HealthCheck",
    "HealthResult",
    "CheckReport",
    "HealthReport",
    "HealthRegistry",
    "registry",
    "get_registry",
    "run_health",
]
