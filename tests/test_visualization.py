"""Tests for visualisation helpers (change-frequency maps, plotting functions).

Uses the headless Agg backend so no display is required.  These tests do not
import the model notebooks — only the pure-numpy plotting helpers in
``utils/visualization_utils.py``.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# compute_change_frequencies
# ---------------------------------------------------------------------------

from utils.visualization_utils import compute_change_frequencies


class TestComputeChangeFrequencies:
    def test_returns_three_arrays(self):
        stack = np.random.randint(0, 2, size=(5, 16, 16)).astype(np.float32)
        result = compute_change_frequencies(stack)
        assert len(result) == 3
        erosion, accretion, instability = result
        assert erosion.shape == (16, 16)
        assert accretion.shape == (16, 16)
        assert instability.shape == (16, 16)

    def test_consecutive_all_erosion(self):
        """Land(1) -> water(0) in first transition only → erosion = 0.5."""
        stack = np.array([np.ones((8, 8)), np.zeros((8, 8)), np.zeros((8, 8))], dtype=np.float32)
        erosion, _, _ = compute_change_frequencies(stack)
        np.testing.assert_allclose(erosion, 0.5)

    def test_consecutive_all_accretion(self):
        """Water(0) -> land(1) in first transition only → accretion = 0.5."""
        stack = np.array([np.zeros((8, 8)), np.ones((8, 8)), np.ones((8, 8))], dtype=np.float32)
        _, accretion, _ = compute_change_frequencies(stack)
        np.testing.assert_allclose(accretion, 0.5)

    def test_no_change(self):
        """Identical consecutive frames → zero erosion and accretion."""
        stack = np.ones((4, 8, 8), dtype=np.float32)
        erosion, accretion, instability = compute_change_frequencies(stack)
        np.testing.assert_allclose(erosion, 0.0)
        np.testing.assert_allclose(accretion, 0.0)

    def test_with_baseline(self):
        """When baseline is given, transitions are computed against the baseline."""
        base = np.ones((8, 8), dtype=np.float32)  # all land
        stack = np.zeros((3, 8, 8), dtype=np.float32)  # all water
        erosion, accretion, _ = compute_change_frequencies(stack, baseline=base)
        np.testing.assert_allclose(erosion, 1.0)
        np.testing.assert_allclose(accretion, 0.0)

    def test_instability_bounded(self):
        """Instability index must be in [0, 1]."""
        rng = np.random.RandomState(42)
        stack = rng.randint(0, 2, size=(6, 12, 12)).astype(np.float32)
        _, _, instability = compute_change_frequencies(stack)
        assert instability.min() >= 0.0
        assert instability.max() <= 1.0

    def test_threshold_applied(self):
        """Values < 0.5 after thresholding should be treated as water (0)."""
        stack = np.full((2, 4, 4), 0.3, dtype=np.float32)
        erosion, accretion, _ = compute_change_frequencies(stack)
        np.testing.assert_allclose(erosion, 0.0)
        np.testing.assert_allclose(accretion, 0.0)


# ---------------------------------------------------------------------------
# Plot helpers (Agg backend, must return a Figure object)
# ---------------------------------------------------------------------------

from utils.visualization_utils import (  # noqa: E402
    plot_training_curves,
    plot_area_trend,
)


def _make_history(n_epochs=10):
    """Return a dict that mimics a Keras History object."""
    epochs = np.arange(n_epochs)
    return {
        "loss": np.exp(-epochs / 5),
        "val_loss": np.exp(-epochs / 5) + 0.01,
        "dice_coefficient": 0.5 + 0.05 * epochs,
        "val_dice_coefficient": 0.45 + 0.05 * epochs,
        "iou_metric": 0.3 + 0.03 * epochs,
        "val_iou_metric": 0.28 + 0.03 * epochs,
    }


class TestPlotTrainingCurves:
    def test_returns_figure_from_dict(self):
        fig = plot_training_curves(_make_history(), "Test Setup", seq_len=4)
        assert isinstance(fig, Figure)

    def test_returns_figure_from_history_object(self):
        class FakeHistory:
            def __init__(self, h):
                self.history = h

        fig = plot_training_curves(FakeHistory(_make_history()), "Test", seq_len=4)
        assert isinstance(fig, Figure)

    def test_saves_to_path(self, tmp_path):
        out = tmp_path / "curves.png"
        fig = plot_training_curves(_make_history(), "Test Setup", seq_len=4, out_path=str(out))
        assert out.exists()
        assert isinstance(fig, Figure)


class TestPlotAreaTrend:
    def test_returns_figure(self):
        years = [2010, 2011, 2012, 2013, 2014]
        areas = [100.0, 105.0, 102.0, 110.0, 108.0]
        fig = plot_area_trend(years, areas)
        assert isinstance(fig, Figure)

    def test_saves_to_path(self, tmp_path):
        years = [2010, 2011, 2012]
        areas = [100.0, 105.0, 102.0]
        out = tmp_path / "trend.png"
        fig = plot_area_trend(years, areas, out_path=str(out))
        assert out.exists()
        assert isinstance(fig, Figure)
