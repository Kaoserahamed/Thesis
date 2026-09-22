"""Tests for the differentiable Keras losses and metrics in ``utils.model_losses``.

These are the training-time objectives (equations 5.6-5.9 of the thesis).  They
are pure TensorFlow / Keras operations, so every test here is marked
``requires_tensorflow``.  None of them is ``slow`` -- constant tensors make each
call cheap -- so they run in the **fast** CI lane as long as TF is installed.

The corresponding NumPy evaluation-time equivalents (``iou_np``, ``dice_np``) are
covered by :mod:`tests.test_metrics`.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_tensorflow

pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from utils.model_losses import (  # noqa: E402
    bce_loss,
    combined_loss,
    dice_coefficient,
    dice_loss,
    iou_metric,
)


def _pair(shape=(8, 8), dtype=np.float32):
    """Two (shape) tensors stacked along a trailing channel dim, matching the
    ``(batch, h, w, 1)`` convention used by the builders."""
    return (
        np.ones((*shape, 1), dtype=dtype),
        np.ones((*shape, 1), dtype=dtype),
    )


# ---------------------------------------------------------------------------
# dice_coefficient
# ---------------------------------------------------------------------------


class TestDiceCoefficient:
    def test_identical_full(self):
        y_true, y_pred = _pair()
        assert dice_coefficient(y_true, y_pred).numpy() == pytest.approx(1.0)

    def test_identical_empty(self):
        y_true = np.zeros((8, 8, 1), dtype=np.float32)
        y_pred = np.zeros((8, 8, 1), dtype=np.float32)
        # Both sums are 0 → only ``smooth`` remains in numerator and
        # denominator → 1.0 (a "perfect" score for two empty masks).
        assert dice_coefficient(y_true, y_pred).numpy() == pytest.approx(1.0)

    def test_no_overlap(self):
        y_true = np.ones((8, 8, 1), dtype=np.float32)
        y_pred = np.zeros((8, 8, 1), dtype=np.float32)
        result = dice_coefficient(y_true, y_pred).numpy()
        assert result == pytest.approx(0.0, abs=1e-5)

    def test_partial_overlap(self):
        y_true = np.zeros((8, 8, 1), dtype=np.float32)
        y_true[:4, :4, 0] = 1.0
        y_pred = np.zeros((8, 8, 1), dtype=np.float32)
        y_pred[4:, 4:, 0] = 1.0
        assert dice_coefficient(y_true, y_pred).numpy() == pytest.approx(0.0, abs=1e-5)

    def test_half_overlap(self):
        y_true = np.ones((8, 8, 1), dtype=np.float32)
        y_pred = np.zeros((8, 8, 1), dtype=np.float32)
        y_pred[:, :4, 0] = 1.0
        assert dice_coefficient(y_true, y_pred).numpy() == pytest.approx(2 / 3, abs=1e-4)

    def test_symmetric(self):
        a = np.random.RandomState(1).rand(8, 8, 1).astype(np.float32)
        b = np.random.RandomState(2).rand(8, 8, 1).astype(np.float32)
        assert dice_coefficient(a, b).numpy() == pytest.approx(dice_coefficient(b, a).numpy())


# ---------------------------------------------------------------------------
# dice_loss
# ---------------------------------------------------------------------------


class TestDiceLoss:
    def test_zero_for_identical(self):
        y_true, y_pred = _pair()
        assert dice_loss(y_true, y_pred).numpy() == pytest.approx(0.0, abs=1e-5)

    def test_one_for_disjoint(self):
        y_true = np.ones((8, 8, 1), dtype=np.float32)
        y_pred = np.zeros((8, 8, 1), dtype=np.float32)
        assert dice_loss(y_true, y_pred).numpy() == pytest.approx(1.0, abs=1e-5)

    def test_bounded_in_0_1(self):
        a = np.random.RandomState(3).rand(8, 8, 1).astype(np.float32)
        b = np.random.RandomState(4).rand(8, 8, 1).astype(np.float32)
        assert 0.0 <= dice_loss(a, b).numpy() <= 1.0


# ---------------------------------------------------------------------------
# bce_loss
# ---------------------------------------------------------------------------


class TestBceLoss:
    def test_zero_for_identical(self):
        y_true, y_pred = _pair()
        assert bce_loss(y_true, y_pred).numpy() == pytest.approx(0.0, abs=1e-5)

    def test_positive(self):
        y_true = np.ones((8, 8, 1), dtype=np.float32)
        y_pred = np.zeros((8, 8, 1), dtype=np.float32)
        assert bce_loss(y_true, y_pred).numpy() > 0.0

    def test_symmetric_examples(self):
        y_true = np.ones((4, 4, 1), dtype=np.float32)
        y_pred = np.zeros((4, 4, 1), dtype=np.float32)
        # BCE is not perfectly symmetric at the 0/1 extremes (epsilon
        # clipping differs slightly), so compare within a generous tolerance.
        l1 = bce_loss(y_true, y_pred).numpy()
        l2 = bce_loss(y_pred, y_true).numpy()
        assert abs(l1 - l2) < 0.5
        assert l1 > 0.0
        assert l2 > 0.0


# ---------------------------------------------------------------------------
# combined_loss
# ---------------------------------------------------------------------------


class TestCombinedLoss:
    def test_delegates_to_bce_plus_dice(self):
        y_true, y_pred = _pair()
        combined = combined_loss(y_true, y_pred).numpy()
        bce = bce_loss(y_true, y_pred).numpy()
        dl = dice_loss(y_true, y_pred).numpy()
        assert combined == pytest.approx(bce + dl, abs=1e-5)

    def test_non_negative(self):
        a = np.random.RandomState(5).rand(8, 8, 1).astype(np.float32)
        b = np.random.RandomState(6).rand(8, 8, 1).astype(np.float32)
        assert combined_loss(a, b).numpy() >= 0.0


# ---------------------------------------------------------------------------
# iou_metric
# ---------------------------------------------------------------------------


class TestIouMetric:
    def test_perfect_match(self):
        y_true, y_pred = _pair()
        assert iou_metric(y_true, y_pred).numpy() == pytest.approx(1.0)

    def test_no_overlap(self):
        y_true = np.ones((8, 8, 1), dtype=np.float32)
        y_pred = np.zeros((8, 8, 1), dtype=np.float32)
        assert iou_metric(y_true, y_pred).numpy() == pytest.approx(0.0, abs=1e-5)

    def test_bounded_0_1(self):
        a = np.random.RandomState(7).rand(8, 8, 1).astype(np.float32)
        b = np.random.RandomState(8).rand(8, 8, 1).astype(np.float32)
        iou = iou_metric(a, b).numpy()
        assert 0.0 <= iou <= 1.0

    def test_threshold_default_matches_binary_threshold(self):
        y_true = np.zeros((8, 8, 1), dtype=np.float32)
        y_true[:4, :, 0] = 0.6
        y_pred = np.zeros((8, 8, 1), dtype=np.float32)
        y_pred[:4, :, 0] = 0.4  # just below default threshold
        assert iou_metric(y_true, y_pred).numpy() == pytest.approx(0.0, abs=1e-5)

    def test_partial_overlap_perfect(self):
        y_true = np.zeros((8, 8, 1), dtype=np.float32)
        y_true[:4, :4, 0] = 1.0
        y_pred = np.zeros((8, 8, 1), dtype=np.float32)
        y_pred[:4, :4, 0] = 1.0
        assert iou_metric(y_true, y_pred).numpy() == pytest.approx(1.0)
