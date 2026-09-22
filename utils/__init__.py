"""
Utility package for the Padma River morphology thesis.

Modules
-------
- model_utils         : shared architectures, losses, metrics, data pipeline
- data_utils          : GeoTIFF I/O and preprocessing helpers
- visualization_utils : plotting and interactive risk-map helpers

model_utils is imported lazily / guarded because it pulls in TensorFlow.
If TensorFlow cannot be loaded the submodule simply becomes unavailable so
that data_utils and the pure-numpy helpers in visualization_utils can still
be imported and tested.
"""

from typing import Optional, TYPE_CHECKING

from . import data_utils  # noqa: F401

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

__all__ = ["data_utils", "model_utils", "visualization_utils"]
