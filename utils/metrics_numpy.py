"""
Pure-NumPy evaluation metrics for river morphology prediction.

These functions mirror the TensorFlow/Keras-based losses and metrics in
``model_utils`` but operate on plain :mod:`numpy` arrays, so they can be
imported and unit-tested **without** a deep-learning runtime.

Reference
---------
"Analyzing & Forecasting River Morphological Evolution Using Machine
Learning & Spatiotemporal Neural Models: A Case Study on the Padma River"
Chapter 5 – Methodology.
"""

from __future__ import annotations

import numpy as np

BINARY_THRESHOLD: float = 0.5


def iou_np(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = BINARY_THRESHOLD) -> float:
    """Intersection-over-Union (Jaccard index) on binarised arrays.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Continuous or binary prediction masks of identical shape.
    threshold : float
        Values strictly greater than this threshold count as water (1).

    Returns
    -------
    float
        IoU score in [0, 1] (with a small epsilon for numerical stability).
    """
    yt = (y_true > threshold).astype(np.float32)
    yp = (y_pred > threshold).astype(np.float32)
    inter = np.sum(yt * yp)
    union = np.sum(yt) + np.sum(yp) - inter
    return float(inter / (union + 1e-6))


def dice_np(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = BINARY_THRESHOLD) -> float:
    """F1 / Dice coefficient on binarised arrays.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Continuous or binary prediction masks of identical shape.
    threshold : float
        Values strictly greater than this threshold count as water (1).

    Returns
    -------
    float
        Dice score in [0, 1] (with a small epsilon for numerical stability).
    """
    yt = (y_true > threshold).astype(np.float32)
    yp = (y_pred > threshold).astype(np.float32)
    inter = np.sum(yt * yp)
    return float((2.0 * inter) / (np.sum(yt) + np.sum(yp) + 1e-6))


def calculate_area_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pixel_area_km2: float,
    threshold: float = BINARY_THRESHOLD,
) -> float:
    r"""Signed water-area difference ΔA = A_pred − A_true  (km²).

    A positive value means the prediction overestimates water area;
    a negative value means it underestimates.  Equation 5.6 of the thesis.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Masks of identical shape.
    pixel_area_km2 : float
        Ground area covered by one pixel, in km².
    threshold : float
        Binarisation threshold (default 0.5).

    Returns
    -------
    float
        Signed area difference in km².
    """
    true_area = np.sum((y_true > threshold).astype(np.float32)) * pixel_area_km2
    pred_area = np.sum((y_pred > threshold).astype(np.float32)) * pixel_area_km2
    return float(pred_area - true_area)
