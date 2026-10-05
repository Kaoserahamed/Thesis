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


def precision_recall_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = BINARY_THRESHOLD,
) -> tuple[float, float]:
    """
    Calculate precision and recall for binary masks.
    
    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Continuous or binary prediction masks of identical shape
    threshold : float
        Binarization threshold (default 0.5)
    
    Returns
    -------
    tuple[float, float]
        (precision, recall) in [0, 1]
        Returns (0.0, 0.0) if no positive predictions or ground truth
    """
    yt = (y_true > threshold).astype(np.float32)
    yp = (y_pred > threshold).astype(np.float32)
    
    true_positives = np.sum(yt * yp)
    predicted_positives = np.sum(yp)
    actual_positives = np.sum(yt)
    
    # Handle edge cases
    precision = float(true_positives / (predicted_positives + 1e-6))
    recall = float(true_positives / (actual_positives + 1e-6))
    
    return precision, recall


def aggregate_metrics(
    y_true_batch: np.ndarray,
    y_pred_batch: np.ndarray,
    pixel_area_km2: float = 1.0,
    threshold: float = BINARY_THRESHOLD,
) -> dict[str, float]:
    """
    Calculate all standard metrics for a batch of predictions.
    
    Parameters
    ----------
    y_true_batch : np.ndarray
        Ground truth masks of shape (batch, height, width) or (batch, height, width, 1)
    y_pred_batch : np.ndarray
        Predicted masks of same shape as y_true_batch
    pixel_area_km2 : float
        Ground area per pixel in km² (default 1.0)
    threshold : float
        Binarization threshold (default 0.5)
    
    Returns
    -------
    dict[str, float]
        Dictionary with keys: iou, dice, precision, recall, area_diff_km2
        Each value is the mean across the batch
    
    Examples
    --------
    >>> y_true = np.random.rand(10, 256, 256, 1) > 0.5
    >>> y_pred = np.random.rand(10, 256, 256, 1)
    >>> metrics = aggregate_metrics(y_true, y_pred, pixel_area_km2=0.0036)
    >>> print(f"Mean IoU: {metrics['iou']:.3f}")
    """
    # Squeeze channel dimension if present
    if y_true_batch.ndim == 4 and y_true_batch.shape[-1] == 1:
        y_true_batch = y_true_batch.squeeze(-1)
    if y_pred_batch.ndim == 4 and y_pred_batch.shape[-1] == 1:
        y_pred_batch = y_pred_batch.squeeze(-1)
    
    # Validate shapes
    if y_true_batch.shape != y_pred_batch.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_batch.shape} vs y_pred {y_pred_batch.shape}"
        )
    
    if y_true_batch.ndim != 3:
        raise ValueError(
            f"Expected 3D batch (batch, height, width), got shape {y_true_batch.shape}"
        )
    
    batch_size = y_true_batch.shape[0]
    
    # Calculate metrics for each sample
    iou_scores = []
    dice_scores = []
    precision_scores = []
    recall_scores = []
    area_diffs = []
    
    for i in range(batch_size):
        iou_scores.append(iou_np(y_true_batch[i], y_pred_batch[i], threshold))
        dice_scores.append(dice_np(y_true_batch[i], y_pred_batch[i], threshold))
        
        prec, rec = precision_recall_np(y_true_batch[i], y_pred_batch[i], threshold)
        precision_scores.append(prec)
        recall_scores.append(rec)
        
        area_diffs.append(
            calculate_area_difference(
                y_true_batch[i], y_pred_batch[i], pixel_area_km2, threshold
            )
        )
    
    return {
        "iou": float(np.mean(iou_scores)),
        "dice": float(np.mean(dice_scores)),
        "precision": float(np.mean(precision_scores)),
        "recall": float(np.mean(recall_scores)),
        "area_diff_km2": float(np.mean(area_diffs)),
    }


def validate_mask_array(
    mask: np.ndarray,
    expected_shape: tuple[int, ...] | None = None,
    allow_nan: bool = False,
) -> None:
    """
    Validate a mask array for common issues.
    
    Parameters
    ----------
    mask : np.ndarray
        Mask array to validate
    expected_shape : tuple, optional
        Expected shape, if None then no shape check
    allow_nan : bool
        Whether NaN values are allowed (default False)
    
    Raises
    ------
    ValueError : If validation fails
    TypeError : If input is not a numpy array
    
    Examples
    --------
    >>> mask = np.random.rand(256, 256)
    >>> validate_mask_array(mask, expected_shape=(256, 256))
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(mask)}")
    
    if expected_shape is not None:
        if mask.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch: expected {expected_shape}, got {mask.shape}"
            )
    
    if not allow_nan and np.any(np.isnan(mask)):
        raise ValueError("Mask contains NaN values")
    
    if np.any(np.isinf(mask)):
        raise ValueError("Mask contains infinite values")

