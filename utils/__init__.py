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

Only ``model_utils`` requires TensorFlow.  It is imported lazily / guarded so
that the TensorFlow-free modules above can still be imported and unit-tested
when TensorFlow is unavailable (this is what keeps the fast CI lane green).

Logging
-------
Call :func:`configure_logging` once per process (typically from a script or
notebook entry point) to get a consistent format and level across every
``utils`` logger::

    from utils import configure_logging
    configure_logging()
"""

import logging
from typing import Optional, TYPE_CHECKING

from . import metrics_numpy  # noqa: F401
from . import pipeline_utils  # noqa: F401
from . import data_utils  # noqa: F401

logging.getLogger(__name__).addHandler(logging.NullHandler())

if TYPE_CHECKING:
    from types import ModuleType

model_utils: Optional["ModuleType"]
try:
    from . import model_utils  # noqa: F401
except ImportError:
    model_utils = None

try:
    from . import visualization_utils  # noqa: F401
except ImportError:
    pass

__all__ = [
    "configure_logging",
    "data_utils",
    "metrics_numpy",
    "model_utils",
    "pipeline_utils",
    "visualization_utils",
]


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a single, consistent handler for every ``utils`` logger.

    Idempotent: calling it more than once (e.g. from a notebook that is
    re-executed) does not duplicate handlers.  Scripts and notebooks should
    call this once before running a long preprocessing or training job so that
    progress is timestamped and can be captured by the host environment.
    """
    package_logger = logging.getLogger(__name__)
    package_logger.setLevel(level)

    for handler in list(package_logger.handlers):
        if isinstance(handler, logging.NullHandler):
            package_logger.removeHandler(handler)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in package_logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        package_logger.addHandler(handler)
