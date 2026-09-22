"""Tests for the pure-NumPy forecast helpers in ``utils.forecast_utils``.

``frames_to_cube`` / ``make_forecast_sequences`` were extracted from the
inline ``to_cube`` / ``make_sequences`` helpers duplicated in every generated
prediction notebook (``scripts/notebookgen/prediction.py``); the generator now
inlines these functions verbatim, so this module keeps the notebook-driven
forecast path (the M dimension) verifiable from tests alone.

Everything here is pure NumPy and runs in the fast CI lane with no
TensorFlow, no data files and no notebook execution.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.forecast_utils import (
    SEASONAL_PERIOD,
    frames_to_cube,
    make_forecast_sequences,
)


def _frames(count: int = 5, height: int = 4, width: int = 6) -> list[np.ndarray]:
    rng = np.random.RandomState(0)
    return [rng.rand(height, width).astype(np.float32) for _ in range(count)]


class TestFramesToCube:
    def test_single_channel_shape(self) -> None:
        cube = frames_to_cube(_frames(5, 4, 6), seasonal=False)
        assert cube.shape == (5, 4, 6, 1)
        assert cube.dtype == np.float32

    def test_seasonal_shape(self) -> None:
        cube = frames_to_cube(_frames(5, 4, 6), seasonal=True)
        assert cube.shape == (5, 4, 6, 3)

    def test_first_channel_matches_input(self) -> None:
        frames = _frames(3, 4, 5)
        cube = frames_to_cube(frames, seasonal=True)
        for t, frame in enumerate(frames):
            np.testing.assert_allclose(cube[t, ..., 0], frame)

    def test_seasonal_channels_encode_annual_cycle(self) -> None:
        cube = frames_to_cube(_frames(SEASONAL_PERIOD + 1, 2, 2), seasonal=True)
        for t in range(SEASONAL_PERIOD + 1):
            phase = 2.0 * np.pi * ((t % SEASONAL_PERIOD) / float(SEASONAL_PERIOD))
            assert cube[t, 0, 0, 1] == pytest.approx(np.sin(phase))
            assert cube[t, 0, 0, 2] == pytest.approx(np.cos(phase))
        # The cycle wraps: period 0 and period SEASONAL_PERIOD agree.
        assert cube[0, 0, 0, 1] == pytest.approx(cube[SEASONAL_PERIOD, 0, 0, 1])
        assert cube[0, 0, 0, 2] == pytest.approx(cube[SEASONAL_PERIOD, 0, 0, 2])

    def test_seasonal_channels_are_spatially_constant(self) -> None:
        cube = frames_to_cube(_frames(2, 3, 4), seasonal=True)
        for t in range(2):
            assert np.ptp(cube[t, ..., 1]) == pytest.approx(0.0)
            assert np.ptp(cube[t, ..., 2]) == pytest.approx(0.0)


class TestMakeForecastSequences:
    def test_shapes_match_sliding_windows(self) -> None:
        cube = frames_to_cube(_frames(6, 4, 5), seasonal=False)
        x, y = make_forecast_sequences(cube, seq_len=4)
        assert x.shape == (2, 4, 4, 5, 1)
        assert y.shape == (2, 4, 5, 1)

    def test_target_is_next_frame_first_channel(self) -> None:
        cube = frames_to_cube(_frames(4, 3, 3), seasonal=True)
        x, y = make_forecast_sequences(cube, seq_len=2)
        np.testing.assert_allclose(y[0, ..., 0], cube[2, ..., 0])
        np.testing.assert_allclose(x[1, -1], cube[2])

    def test_horizon_shifts_target(self) -> None:
        cube = frames_to_cube(_frames(5, 2, 2), seasonal=False)
        _, y = make_forecast_sequences(cube, seq_len=2, horizon=2)
        np.testing.assert_allclose(y[0, ..., 0], cube[3, ..., 0])

    def test_too_short_cube_returns_empty(self) -> None:
        cube = frames_to_cube(_frames(2, 3, 3), seasonal=False)
        x, y = make_forecast_sequences(cube, seq_len=4)
        assert x.shape == (0, 4, 3, 3, 1)
        assert y.shape == (0, 3, 3, 1)

    def test_rejects_non_positive_controls(self) -> None:
        cube = frames_to_cube(_frames(4, 2, 2), seasonal=False)
        with pytest.raises(ValueError, match="positive integer"):
            make_forecast_sequences(cube, seq_len=0)
        with pytest.raises(ValueError, match="positive integer"):
            make_forecast_sequences(cube, seq_len=2, horizon=0)

    def test_rejects_non_cube_input(self) -> None:
        with pytest.raises(ValueError, match=r"\(T, H, W, C\)"):
            make_forecast_sequences(np.zeros((4, 4)), seq_len=2)
