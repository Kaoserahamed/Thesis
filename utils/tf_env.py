"""TensorFlow environment bootstrap: quiet logs, deterministic kernels.

Imported *before* ``import tensorflow`` by every TensorFlow-dependent module of
this package, so the C++ log level and the oneDNN/XLA flags are configured
exactly once per process.
"""

from __future__ import annotations

import logging
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")

logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("absl").setLevel(logging.FATAL)


def silence_tf() -> None:
    """Route the TensorFlow Python logger to FATAL (call after importing it)."""
    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover - TensorFlow is a hard dependency here
        return
    tf.get_logger().setLevel("FATAL")
