"""Differentiable losses and Keras metrics (thesis equations 5.6-5.9).

TensorFlow-dependent; the NumPy equivalents used at evaluation time live in
:mod:`utils.metrics_numpy`.
"""

from __future__ import annotations

from .tf_env import silence_tf

import tensorflow as tf
from tensorflow.keras import backend as K

from .metrics_numpy import BINARY_THRESHOLD

silence_tf()


def dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    """Soft Dice coefficient (differentiable) used as a training metric."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """Region-based loss: ``1 - Dice`` (equation 5.7 of the thesis)."""
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_loss(y_true, y_pred):
    """Pixel-wise binary cross-entropy (equation 5.8 of the thesis)."""
    return K.mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))


def combined_loss(y_true, y_pred):
    """
    Hybrid objective ``L_BCE + L_Dice`` (equation 5.9).

    BCE gives stable early gradients; Dice sharpens boundaries.  The sum
    balances pixel-level accuracy with region-level overlap, which is the
    configuration used for *all* five architectures.
    """
    return bce_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, threshold: float = BINARY_THRESHOLD):
    """Binarised Intersection-over-Union as a Keras metric."""
    y_pred_b = K.cast(y_pred > threshold, "float32")
    y_true_b = K.cast(y_true > threshold, "float32")
    intersection = K.sum(y_true_b * y_pred_b)
    union = K.sum(y_true_b) + K.sum(y_pred_b) - intersection
    return (intersection + K.epsilon()) / (union + K.epsilon())
