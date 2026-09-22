"""Forecast-specific pure-NumPy helpers (thesis sections 7.4-7.5).

The best-model prediction notebooks under ``analysis/`` roll the winning
architecture forward autoregressively.  The two helpers below used to live
duplicated inline in ``scripts/notebookgen/prediction.py`` (``to_cube`` /
``make_sequences``) and -- through the generator -- in every generated
prediction notebook.  They are pure NumPy so they can be unit-tested in the
fast CI lane without TensorFlow, and the generator inlines them verbatim via
:func:`scripts.notebookgen.extract.grab` to keep the notebooks self-contained
from this single source of truth.
"""

from __future__ import annotations

import numpy as np

#: Number of bi-monthly periods per annual cycle used for the seasonal
#: sin/cos channels of the short-term (bi-monthly) forecast.
SEASONAL_PERIOD = 6

__all__ = [
    "SEASONAL_PERIOD",
    "frames_to_cube",
    "make_forecast_sequences",
]


def frames_to_cube(frames, seasonal: bool = False) -> np.ndarray:
    """Stack frames into a ``(T, H, W, C)`` tensor.

    Parameters
    ----------
    frames : sequence of (H, W) arrays
        Chronological water-mask frames.
    seasonal : bool
        When True, append sin/cos channels encoding the position inside the
        annual cycle (used by the bi-monthly short-term forecast); otherwise
        return a single-channel cube.

    Returns
    -------
    np.ndarray
        ``(T, H, W, 1)`` cube when ``seasonal`` is False, else ``(T, H, W, 3)``.
    """
    stack = np.stack(list(frames), axis=0).astype(np.float32)
    if not seasonal:
        return np.expand_dims(stack, -1)
    count = stack.shape[0]
    height, width = stack.shape[1], stack.shape[2]
    cube = np.empty((count, height, width, 3), dtype=np.float32)
    for t in range(count):
        phase = 2.0 * np.pi * ((t % SEASONAL_PERIOD) / float(SEASONAL_PERIOD))
        cube[t, ..., 0] = stack[t]
        cube[t, ..., 1] = np.sin(phase)
        cube[t, ..., 2] = np.cos(phase)
    return cube


def make_forecast_sequences(
    cube: np.ndarray, seq_len: int, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window sequences (stride 1) over a forecast cube.

    The target is the first channel of the frame ``horizon`` steps after the
    input window (thesis uses ``horizon=1``, i.e. the next frame).

    Returns
    -------
    (X, y) : tuple[np.ndarray, np.ndarray]
        ``X`` has shape ``(N, seq_len, H, W, C)`` and ``y`` has shape
        ``(N, H, W, 1)``.  When no window fits, both arrays are empty with
        the correct trailing dimensions.
    """
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
        raise ValueError("seq_len must be a positive integer")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    cube = np.asarray(cube)
    if cube.ndim != 4:
        raise ValueError("cube must be a (T, H, W, C) array")
    total, height, width, channels = cube.shape
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for i in range(total - seq_len - horizon + 1):
        inputs.append(cube[i : i + seq_len])
        targets.append(cube[i + seq_len + horizon - 1, ..., 0])
    if not inputs:
        return (
            np.empty((0, seq_len, height, width, channels), dtype=np.float32),
            np.empty((0, height, width, 1), dtype=np.float32),
        )
    return np.array(inputs), np.expand_dims(np.array(targets), -1)
