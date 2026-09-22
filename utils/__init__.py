"""
Utility package for the Padma River morphology thesis.

Modules
-------
- metrics_numpy       : NumPy IoU / Dice / area-difference metrics (no TF)
- pipeline_utils      : GeoTIFF preprocessing, sequences, temporal split,
                        ExperimentConfig presets (no TF)
- data_utils          : GeoTIFF I/O, normalisation, water masks (no TF)
- model_utils         : architectures, losses, Keras metrics, training helpers
- visualization_utils : plotting and interactive risk-map helpers
- logging_framework   : structured JSON/text logging with context binding
- error_tracking      : crash-tolerant error collection + Prometheus /metrics
- health              : pluggable health-check registry + HTTP service

Only ``model_utils`` requires TensorFlow.  It is imported lazily / guarded so
that the TensorFlow-free modules above can still be imported and unit-tested
when TensorFlow is unavailable (this is what keeps the fast CI lane green).

Observability
-------------
Call :func:`configure_logging` once per process to switch the ``utils`` loggers
to JSON-lines or human-readable output across scripts, notebooks and services::

    from utils import configure_logging
    configure_logging()

Errors are automatically captured by the shared :func:`utils.error_tracking`
tracker (reachable at ``/metrics`` on the health service).
"""

from __future__ import annotations

import logging
import importlib
from typing import TYPE_CHECKING, Optional

from . import metrics_numpy  # noqa: F401
from . import pipeline_utils  # noqa: F401
from . import data_utils  # noqa: F401
from . import logging_framework  # noqa: F401
from . import error_tracking  # noqa: F401
from . import health  # noqa: F401

__version__ = "0.1.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Re-export the structured logging entry point so
# ``from utils import configure_logging`` keeps working.
configure_logging = logging_framework.configure_logging

if TYPE_CHECKING:
    from types import ModuleType

model_utils: Optional["ModuleType"] = None

try:
    from . import visualization_utils  # noqa: F401
except ImportError:
    pass


def __getattr__(name: str):
    """Load TensorFlow-dependent modules only when explicitly requested."""
    if name == "model_utils":
        module = importlib.import_module(".model_utils", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "configure_logging",
    "data_utils",
    "error_tracking",
    "health",
    "logging_framework",
    "metrics_numpy",
    "model_utils",
    "pipeline_utils",
    "visualization_utils",
]
