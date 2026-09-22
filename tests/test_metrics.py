"""Tests for pixel-wise metrics and area-difference helpers (§ 5.8 of the thesis).

These tests are pure numpy and do **not** require TensorFlow at runtime,
but they import ``model_utils`` at module level (which pulls in TF).
When TF is unavailable they are skipped.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_tensorflow

pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from utils.model_utils import (  # noqa: E402
    BINARY_THRESHOLD,
    calculate_area_difference,
    dice_coefficient,
    dice_np,
    iou_np,
)

# ---------------------------------------------------------------------------
# IoU (Jaccard index)
# ---------------------------------------------------------------------------


def test_iou_np_perfect_match(half_mask):
    """Identical masks should give IoU == 1.0."""
    assert iou_np(half_mask, half_mask) == pytest.approx(1.0)


def test_iou_np_no_overlap(half_mask, empty_mask):
    """Masks with zero overlap should give IoU ≈ 0."""
    assert iou_np(half_mask, empty_mask) == pytest.approx(0.0, abs=1e-5)


def test_iou_np_partial_overlap(half_mask, quarter_mask):
    """
    half_mask  : left 16 cols water  (union = 512)
    quarter_mask: top-left 16×16 water (intersection = 256)
    IoU = 256 / (512 + 256 - 256) = 256 / 512 = 0.5
    """
    assert iou_np(half_mask, quarter_mask) == pytest.approx(0.5, abs=1e-4)


def test_iou_np_custom_threshold():
    """Values below the custom threshold count as land."""
    y_true = np.zeros((4, 4), dtype=np.float32)
    y_true[:2, :] = 0.6  # above 0.5 → water
    y_pred = np.zeros((4, 4), dtype=np.float32)
    y_pred[:2, :] = 0.4  # below 0.5 → land
    assert iou_np(y_true, y_pred, threshold=0.5) == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Dice coefficient
# ---------------------------------------------------------------------------


def test_dice_np_perfect_match(half_mask):
    """Identical masks should give Dice == 1.0."""
    assert dice_np(half_mask, half_mask) == pytest.approx(1.0)


def test_dice_np_no_overlap(half_mask, empty_mask):
    """Completely disjoint masks → Dice ≈ 0."""
    assert dice_np(half_mask, empty_mask) == pytest.approx(0.0, abs=1e-5)


def test_dice_np_partial_overlap(half_mask, quarter_mask):
    """
    intersection = 256, |A| = 512, |B| = 256
    Dice = 2*256 / (512 + 256 + eps) ≈ 0.6667
    """
    assert dice_np(half_mask, quarter_mask) == pytest.approx(2 / 3, abs=1e-3)


def test_dice_ge_iou(half_mask, quarter_mask):
    """Dice should always be ≥ IoU for the same pair."""
    d = dice_np(half_mask, quarter_mask)
    i = iou_np(half_mask, quarter_mask)
    assert d >= i - 1e-6


# ---------------------------------------------------------------------------
# Area difference ΔA = A_pred − A_true
# ---------------------------------------------------------------------------


def test_calculate_area_difference_zero(half_mask):
    """Identical masks → ΔA = 0."""
    pixel_area = 0.25  # km² / pixel (arbitrary)
    assert calculate_area_difference(half_mask, half_mask, pixel_area) == pytest.approx(0.0)


def test_calculate_area_difference_known():
    """100 water pixels in truth, 50 in prediction, 0.01 km²/pixel."""
    y_true = np.zeros((20, 20), dtype=np.float32)
    y_true[:10] = 1.0  # 200 pixels
    y_pred = np.zeros((20, 20), dtype=np.float32)
    y_pred[:5] = 1.0  # 100 pixels
    pixel_area = 0.01
    # ΔA = 100 * 0.01 − 200 * 0.01 = 1.0 − 2.0 = −1.0
    assert calculate_area_difference(y_true, y_pred, pixel_area) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# TensorFlow-based differentiable loss / metric
# ---------------------------------------------------------------------------


@pytest.mark.requires_tensorflow
def test_dice_coefficient_perfect():
    """Soft Dice between identical tensors should be ≈ 1.0."""
    import tensorflow as tf

    y = tf.constant([[[[1.0]], [[0.0]], [[1.0]]]])
    assert float(dice_coefficient(y, y)) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.requires_tensorflow
def test_dice_coefficient_disjoint():
    """Soft Dice between completely disjoint tensors should be ≈ 0."""
    import tensorflow as tf

    y_true = tf.constant([[[[1.0, 0.0], [0.0, 0.0]]]])
    y_pred = tf.constant([[[[0.0, 0.0], [0.0, 1.0]]]])
    assert float(dice_coefficient(y_true, y_pred)) == pytest.approx(0.0, abs=1e-5)


def test_binary_threshold_is_05():
    """The project default threshold must match the thesis spec."""
    assert BINARY_THRESHOLD == 0.5
