"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Small binary masks for metric testing
# ---------------------------------------------------------------------------


@pytest.fixture
def perfect_mask() -> np.ndarray:
    """A binary mask where every pixel is water (value 1)."""
    return np.ones((32, 32), dtype=np.float32)


@pytest.fixture
def empty_mask() -> np.ndarray:
    """A binary mask where every pixel is land (value 0)."""
    return np.zeros((32, 32), dtype=np.float32)


@pytest.fixture
def half_mask() -> np.ndarray:
    """A mask where the left half is water and the right half is land."""
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[:, :16] = 1.0
    return mask


@pytest.fixture
def quarter_mask() -> np.ndarray:
    """A mask where the top-left quarter is water."""
    mask = np.zeros((32, 32), dtype=np.float32)
    mask[:16, :16] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Synthetic temporal sequences for pipeline testing
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_sequence() -> tuple[np.ndarray, list[int]]:
    """Return 10 synthetic 16×16 binary frames spanning years 2010–2019."""
    rng = np.random.RandomState(42)
    frames = [rng.randint(0, 2, size=(16, 16)).astype(np.float32) for _ in range(10)]
    years = list(range(2010, 2020))
    return np.array(frames), years


@pytest.fixture
def increasing_sequence() -> tuple[np.ndarray, list[int]]:
    """Frames where water fraction strictly increases — deterministic."""
    frames = []
    for i in range(10):
        frame = np.zeros((16, 16), dtype=np.float32)
        frame[: 4 * (i + 1)] = 1.0  # more water each step
        frames.append(frame)
    years = list(range(2000, 2010))
    return np.array(frames), years
