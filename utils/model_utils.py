"""
Model Utilities for River Morphology Prediction
===============================================

This module centralises **every** deep-learning building block used across
the five architectures compared in the thesis:

    1. ConvLSTM                     (baseline spatiotemporal recurrent model)
    2. U-Net + LSTM                 (encoder-decoder with LSTM bottleneck)
    3. Attention U-Net + ConvLSTM   (attention-gated skip connections)
    4. Swin Spatio-Temporal Transformer
    5. Vision Transformer (ViT / ViViT) spatio-temporal model

It also provides the shared data pipeline (sequence generation, strict
temporal splitting), the hybrid BCE + Dice loss, the pixel-wise metrics
(IoU, Dice, Precision, Recall, area difference) and a training/evaluation
helper so that every notebook in ``models/`` stays thin, readable and
reproducible.

Why a shared module?
--------------------
The thesis trains *the same* architectures at three temporal resolutions
(yearly, quarterly, bi-monthly) under *identical* preprocessing, sequence
generation and temporal-split strategies.  Keeping that logic in one place
guarantees a fair comparison and avoids copy-paste drift between notebooks.

Reference
---------
"Analyzing & Forecasting River Morphological Evolution Using Machine
Learning & Spatiotemporal Neural Models: A Case Study on the Padma River"
Chapter 5 – Methodology.
"""

from __future__ import annotations

import os

# ── Silence TensorFlow / XLA / CUDA noise BEFORE importing tensorflow ──────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")

import glob
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("absl").setLevel(logging.FATAL)

import tensorflow as tf  # noqa: E402
from tensorflow.keras import backend as K, layers, models  # noqa: E402
from tensorflow.keras.callbacks import (  # noqa: E402
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam  # noqa: E402

import cv2  # noqa: E402
import rasterio  # noqa: E402
from skimage.transform import resize  # noqa: E402
from sklearn.metrics import precision_score, recall_score  # noqa: E402

tf.get_logger().setLevel("FATAL")

# =============================================================================
# 0.  GLOBAL DEFAULTS  (mirror the thesis training configuration, § 5.9)
# =============================================================================

IMG_HEIGHT: int = 256
IMG_WIDTH: int = 256
N_COMPONENTS: int = 3  # keep the 3 largest connected water components
BINARY_THRESHOLD: float = 0.5
LEARNING_RATE: float = 1e-4
CLIPNORM: float = 1.0
DEFAULT_EPOCHS: int = 200
EARLY_STOP_PATIENCE: int = 20
REDUCE_LR_PATIENCE: int = 7


# =============================================================================
# 1.  LOSSES AND METRICS
# =============================================================================


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


# ── NumPy (evaluation) counterparts ───────────────────────────────────────────


def iou_np(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = BINARY_THRESHOLD) -> float:
    yt = (y_true > threshold).astype(np.float32)
    yp = (y_pred > threshold).astype(np.float32)
    inter = np.sum(yt * yp)
    union = np.sum(yt) + np.sum(yp) - inter
    return float(inter / (union + 1e-6))


def dice_np(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = BINARY_THRESHOLD) -> float:
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
    """ΔA = A_pred − A_true, in km² (equation 5.6)."""
    true_area = np.sum((y_true > threshold).astype(np.float32)) * pixel_area_km2
    pred_area = np.sum((y_pred > threshold).astype(np.float32)) * pixel_area_km2
    return float(pred_area - true_area)


# =============================================================================
# 2.  DATA LOADING AND PREPROCESSING
# =============================================================================


def keep_largest_n_components_cv2(mask: np.ndarray, n: int = N_COMPONENTS):
    """
    Retain only the ``n`` largest connected components of a binary mask.

    This removes salt-and-pepper noise and isolated ponds so the network
    focuses on the main river channel (thesis § 5.9).
    """
    binary_mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    sorted_idx = np.argsort(areas)[::-1]
    n_keep = min(n, len(sorted_idx))

    cleaned = np.zeros_like(mask)
    for idx in sorted_idx[:n_keep]:
        cleaned[labels == (idx + 1)] = mask.max()
    return cleaned


def load_and_preprocess_image(
    filepath: str, apply_cleaning: bool = True, n_components: int = N_COMPONENTS
) -> np.ndarray:
    """
    Load a GeoTIFF water mask and turn it into a 256×256 clean binary array.

    Steps: read → optional connected-component cleaning → resize (nearest)
    → threshold at 0.5.
    """
    with rasterio.open(filepath) as src:
        img = src.read(1)
    if apply_cleaning:
        img = keep_largest_n_components_cv2(img, n=n_components)
    resized = resize(
        img,
        (IMG_HEIGHT, IMG_WIDTH),
        mode="constant",
        preserve_range=True,
        anti_aliasing=False,
    )
    return (resized > 0.5).astype(np.float32)


def get_pixel_area_km2(reference_tif: str) -> float:
    """
    Compute the ground area (km²) covered by a single 256×256 pixel.

    Handles both EPSG:4326 (geographic) and projected CRSs.
    """
    with rasterio.open(reference_tif) as src:
        bounds = src.bounds
        if src.crs and src.crs.to_epsg() == 4326:
            centre_lat = (bounds.top + bounds.bottom) / 2
            m_per_deg_lon = 111_320 * np.cos(np.radians(centre_lat))
            width_m = (bounds.right - bounds.left) * m_per_deg_lon
            height_m = (bounds.top - bounds.bottom) * 111_320
            total_km2 = (width_m * height_m) / 1e6
        else:
            total_km2 = ((bounds.right - bounds.left) * (bounds.top - bounds.bottom)) / 1e6
    return total_km2 / (IMG_HEIGHT * IMG_WIDTH)


def build_catalog(data_dir: str, pattern: str = "*.tif") -> pd.DataFrame:
    """
    Scan ``data_dir`` for GeoTIFFs and return a year-sorted DataFrame.

    Columns: ``filepath``, ``filename``, ``year``.
    """
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    records = []
    for filepath in files:
        match = re.search(r"(\d{4})", os.path.basename(filepath))
        if match:
            records.append(
                {
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "year": int(match.group(1)),
                }
            )
    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


def load_image_stack(data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """Load and preprocess every GeoTIFF in ``data_dir`` into a stack."""
    df = build_catalog(data_dir)
    images, years = [], []
    for _, row in df.iterrows():
        images.append(load_and_preprocess_image(row["filepath"]))
        years.append(row["year"])
    return images, years


# =============================================================================
# 3.  SEQUENCE GENERATION  (sliding window, § 5.1)
# =============================================================================


def create_sequences(
    images: Sequence[np.ndarray],
    years: Sequence[int],
    seq_len: int,
    horizon: int = 1,
    stride: int = 1,
):
    """
    Build overlapping ``(X, y)`` windows with a stride of 1.

    Parameters
    ----------
    images   : list of (H, W) binary frames, chronological order
    years    : matching year labels
    seq_len  : number of input frames ``L``
    horizon  : prediction horizon (thesis uses 1)
    stride   : window stride (thesis uses 1)

    Returns
    -------
    X, y, input_years, target_years : np.ndarrays
    """
    X, y, in_years, tgt_years = [], [], [], []
    stop = len(images) - seq_len - horizon + 1
    for i in range(0, stop, stride):
        X.append(images[i : i + seq_len])
        y.append(images[i + seq_len + horizon - 1])
        in_years.append(years[i : i + seq_len])
        tgt_years.append(years[i + seq_len + horizon - 1])
    return (np.array(X), np.array(y), np.array(in_years), np.array(tgt_years))


# =============================================================================
# 4.  STRICT TEMPORAL SPLIT  (§ 5.2)
# =============================================================================


def prepare_split(X_all, y_all, target_years_all, input_years_all, cutoff_year: int):
    """
    Leakage-proof temporal split.

    Training/validation use samples whose *target* and *latest input* years
    are ≤ ``cutoff_year``; testing uses targets strictly after the cutoff.
    The last 15 % of the (chronologically ordered) training samples become
    the validation set.

    Returns
    -------
    X_tr, y_tr, X_val, y_val, X_test, y_test, target_years_test
    (all arrays already expanded with a trailing channel axis)
    """
    max_input_year = np.array([iy.max() for iy in input_years_all])

    train_mask = (target_years_all <= cutoff_year) & (max_input_year <= cutoff_year)
    test_mask = target_years_all > cutoff_year

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    ty_train = target_years_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    ty_test = target_years_all[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(
            f"Empty split! Train={len(X_train)}, Test={len(X_test)} " f"(cutoff={cutoff_year})"
        )

    order = np.argsort(ty_train)
    X_train, y_train, ty_train = X_train[order], y_train[order], ty_train[order]

    split = int(len(X_train) * 0.85)
    X_tr, y_tr = X_train[:split], y_train[:split]
    X_val, y_val = X_train[split:], y_train[split:]

    if len(X_val) == 0:
        raise ValueError("Empty validation set!")

    overlap = set(ty_train.tolist()) & set(ty_test.tolist())
    assert not overlap, f"DATA LEAK! Overlapping years: {overlap}"

    def _add_channel(a):
        return np.expand_dims(a, axis=-1)

    return (
        _add_channel(X_tr),
        _add_channel(y_tr),
        _add_channel(X_val),
        _add_channel(y_val),
        _add_channel(X_test),
        _add_channel(y_test),
        ty_test,
    )


# =============================================================================
# 5.  MODEL ARCHITECTURES
# =============================================================================


def _adam():
    return Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM)


def build_convlstm(seq_len: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """Architecture 1 – standalone ConvLSTM (thesis Figure 5.1)."""
    inputs = layers.Input(shape=(seq_len, *input_shape))

    x = layers.TimeDistributed(layers.Conv2D(32, 3, padding="same", activation="relu"))(inputs)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.2))(x)

    x = layers.TimeDistributed(layers.Conv2D(64, 3, padding="same", activation="relu"))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.MaxPooling2D(2))(x)
    x = layers.TimeDistributed(layers.Dropout(0.2))(x)

    x = layers.ConvLSTM2D(
        128,
        (3, 3),
        padding="same",
        return_sequences=True,
        dropout=0.3,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(
        64,
        (3, 3),
        padding="same",
        return_sequences=False,
        dropout=0.3,
        kernel_regularizer=tf.keras.regularizers.l2(1e-3),
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name=f"ConvLSTM_seq{seq_len}")
    model.compile(optimizer=_adam(), loss=combined_loss, metrics=[dice_coefficient, iou_metric])
    return model


def _conv_bn(x, filters, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def build_unet_lstm(seq_len: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """
    Architecture 2 – U-Net with a ConvLSTM bottleneck (thesis Figure 5.2).

    The encoder is applied per-frame (TimeDistributed); the bottleneck
    models temporal dependencies; the decoder restores resolution using
    skip connections.
    """
    inputs = layers.Input(shape=(seq_len, *input_shape))
    td = layers.TimeDistributed

    # ── Encoder ──────────────────────────────────────────────────────────────
    e1 = td(layers.Conv2D(32, 3, padding="same", activation="relu"))(inputs)
    e1 = td(layers.BatchNormalization())(e1)
    e1 = td(layers.Conv2D(32, 3, padding="same", activation="relu"))(e1)
    e1 = td(layers.BatchNormalization())(e1)
    p1 = td(layers.MaxPooling2D(2))(e1)
    p1 = td(layers.Dropout(0.15))(p1)

    e2 = td(layers.Conv2D(64, 3, padding="same", activation="relu"))(p1)
    e2 = td(layers.BatchNormalization())(e2)
    e2 = td(layers.Conv2D(64, 3, padding="same", activation="relu"))(e2)
    e2 = td(layers.BatchNormalization())(e2)
    p2 = td(layers.MaxPooling2D(2))(e2)
    p2 = td(layers.Dropout(0.2))(p2)

    e3 = td(layers.Conv2D(128, 3, padding="same", activation="relu"))(p2)
    e3 = td(layers.BatchNormalization())(e3)
    e3 = td(layers.Conv2D(128, 3, padding="same", activation="relu"))(e3)
    e3 = td(layers.BatchNormalization())(e3)
    p3 = td(layers.MaxPooling2D(2))(e3)
    p3 = td(layers.Dropout(0.2))(p3)

    # ── Bottleneck (spatial conv + temporal ConvLSTM) ────────────────────────
    b = td(layers.Conv2D(256, 3, padding="same", activation="relu"))(p3)
    b = td(layers.BatchNormalization())(b)
    b = layers.ConvLSTM2D(256, (3, 3), padding="same", return_sequences=True, dropout=0.3)(b)
    b = layers.BatchNormalization()(b)
    b = layers.ConvLSTM2D(128, (3, 3), padding="same", return_sequences=False, dropout=0.3)(b)
    b = layers.BatchNormalization()(b)

    take_last = layers.Lambda(lambda t: t[:, -1, ...], name="take_last_frame")

    # ── Decoder ──────────────────────────────────────────────────────────────
    d3 = layers.UpSampling2D(2)(b)
    d3 = layers.Concatenate()([d3, take_last(e3)])
    d3 = _conv_bn(d3, 128, 0.2)
    d3 = _conv_bn(d3, 128)

    d2 = layers.UpSampling2D(2)(d3)
    d2 = layers.Concatenate()([d2, take_last(e2)])
    d2 = _conv_bn(d2, 64, 0.2)
    d2 = _conv_bn(d2, 64)

    d1 = layers.UpSampling2D(2)(d2)
    d1 = layers.Concatenate()([d1, take_last(e1)])
    d1 = _conv_bn(d1, 32)

    d1 = layers.Conv2D(16, 3, padding="same", activation="relu")(d1)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d1)

    model = models.Model(inputs, outputs, name=f"UNetLSTM_seq{seq_len}")
    model.compile(optimizer=_adam(), loss=combined_loss, metrics=[dice_coefficient, iou_metric])
    return model


class AttentionGate(layers.Layer):
    """
    Additive attention gate (Oktay et al., 2018) used on U-Net skip
    connections.  ``g`` is the gating (decoder) signal and ``s`` the skip
    feature map; the gate highlights the spatially relevant regions.
    """

    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.Wg = layers.Conv2D(filters, 1, padding="same")
        self.Ws = layers.Conv2D(filters, 1, padding="same")
        self.psi = layers.Conv2D(1, 1, padding="same")
        self.relu = layers.Activation("relu")
        self.sigmoid = layers.Activation("sigmoid")

    def call(self, inputs):
        g, s = inputs
        x = self.relu(self.Wg(g) + self.Ws(s))
        x = self.sigmoid(self.psi(x))
        return s * x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"filters": self.filters})
        return cfg


def build_attention_unet_convlstm(seq_len: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """
    Architecture 3 – Attention U-Net with ConvLSTM bottleneck
    (thesis Figure 5.3).  The best model for yearly and quarterly horizons.
    """
    inputs = layers.Input(shape=(seq_len, *input_shape))
    td = layers.TimeDistributed

    # ── Encoder ──────────────────────────────────────────────────────────────
    e1 = td(layers.Conv2D(32, 3, padding="same", activation="relu"))(inputs)
    e1 = td(layers.BatchNormalization())(e1)
    e1 = td(layers.Conv2D(32, 3, padding="same", activation="relu"))(e1)
    e1 = td(layers.BatchNormalization())(e1)
    p1 = td(layers.MaxPooling2D(2))(e1)
    p1 = td(layers.Dropout(0.15))(p1)

    e2 = td(layers.Conv2D(64, 3, padding="same", activation="relu"))(p1)
    e2 = td(layers.BatchNormalization())(e2)
    e2 = td(layers.Conv2D(64, 3, padding="same", activation="relu"))(e2)
    e2 = td(layers.BatchNormalization())(e2)
    p2 = td(layers.MaxPooling2D(2))(e2)
    p2 = td(layers.Dropout(0.2))(p2)

    e3 = td(layers.Conv2D(128, 3, padding="same", activation="relu"))(p2)
    e3 = td(layers.BatchNormalization())(e3)
    p3 = td(layers.MaxPooling2D(2))(e3)
    p3 = td(layers.Dropout(0.2))(p3)

    # ── Bottleneck ───────────────────────────────────────────────────────────
    b = td(layers.Conv2D(256, 3, padding="same", activation="relu"))(p3)
    b = td(layers.BatchNormalization())(b)
    b = layers.ConvLSTM2D(256, (3, 3), padding="same", return_sequences=True, dropout=0.3)(b)
    b = layers.BatchNormalization()(b)
    b = layers.ConvLSTM2D(128, (3, 3), padding="same", return_sequences=False, dropout=0.3)(b)
    b = layers.BatchNormalization()(b)

    take_last = layers.Lambda(lambda t: t[:, -1, ...])

    # ── Decoder with attention-gated skip connections ───────────────────────
    d3 = layers.UpSampling2D(2)(b)
    a3 = AttentionGate(64)([d3, take_last(e3)])
    d3 = layers.Concatenate()([d3, a3])
    d3 = _conv_bn(d3, 128, 0.2)

    d2 = layers.UpSampling2D(2)(d3)
    a2 = AttentionGate(32)([d2, take_last(e2)])
    d2 = layers.Concatenate()([d2, a2])
    d2 = _conv_bn(d2, 64, 0.2)

    d1 = layers.UpSampling2D(2)(d2)
    a1 = AttentionGate(16)([d1, take_last(e1)])
    d1 = layers.Concatenate()([d1, a1])
    d1 = _conv_bn(d1, 32)

    d1 = layers.Conv2D(16, 3, padding="same", activation="relu")(d1)
    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d1)

    model = models.Model(inputs, outputs, name=f"AttentionUNetConvLSTM_seq{seq_len}")
    model.compile(optimizer=_adam(), loss=combined_loss, metrics=[dice_coefficient, iou_metric])
    return model


def build_swin_st(seq_len: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)):
    """
    Architecture 4 – Swin Spatio-Temporal Transformer (thesis Figure 5.4).

    A lightweight Swin-style hierarchical encoder (window attention +
    patch merging) shared across frames, a temporal fusion head, and a
    U-Net-style convolutional decoder.  Implemented with Keras layers so
    it runs on the same TensorFlow stack as the other models.
    """
    inputs = layers.Input(shape=(seq_len, *input_shape))

    def patch_embed(x):
        return layers.Conv2D(64, 4, strides=4, padding="same")(x)

    def window_attention(x, dim, heads):
        """Simplified shifted-window attention: local 3×3 conv mixing
        followed by global channel attention (efficient proxy for Swin)."""
        y = layers.Conv2D(dim, 3, padding="same", activation="gelu")(x)
        # MultiHeadAttention expects 3-D input (batch, seq, channels);
        # flatten the spatial dims, apply attention, then reshape back.
        h, w = y.shape[1], y.shape[2]
        y_3d = layers.Reshape((h * w, dim))(y)
        attn = layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads, dropout=0.1)(
            y_3d, y_3d
        )
        attn = layers.Reshape((h, w, dim))(attn)
        return layers.Add()([x, attn])

    # Encode every frame with the shared spatial encoder.
    frames = [layers.Lambda(lambda t, i=i: t[:, i, ...])(inputs) for i in range(seq_len)]

    encoded = []
    for f in frames:
        x = patch_embed(f)  # (64, 64, 64)
        x = window_attention(x, 64, 2)  # stage 1
        skip1 = x
        x = layers.MaxPooling2D(2)(x)  # (32, 32, 64)
        x = layers.Conv2D(128, 3, padding="same", activation="gelu")(x)
        x = window_attention(x, 128, 4)
        skip2 = x
        x = layers.MaxPooling2D(2)(x)  # (16, 16, 128)
        x = layers.Conv2D(256, 3, padding="same", activation="gelu")(x)
        x = window_attention(x, 256, 8)
        encoded.append((x, skip1, skip2))

    # ── Temporal fusion across the stacked feature maps ─────────────────────
    # Each frame is summarised to a 256-d vector, then a temporal Conv1D +
    # self-attention fuse the sequence (Conv1D requires (batch, time, feat)).
    h_f, w_f = encoded[-1][0].shape[1], encoded[-1][0].shape[2]
    frame_vecs = [layers.Reshape((1, 256))(layers.GlobalAveragePooling2D()(e[0])) for e in encoded]
    stacked = layers.Concatenate(axis=1)(frame_vecs)  # (batch, T, 256)
    t = layers.Conv1D(256, 3, padding="same", activation="gelu")(stacked)
    t = layers.MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.1)(t, t)
    t = layers.GlobalAveragePooling1D()(t)  # aggregate over time -> (batch, 256)
    t = layers.Dense(h_f * w_f * 256)(t)
    t = layers.Reshape((h_f, w_f, 256))(t)

    # ── U-Net style decoder ─────────────────────────────────────────────────
    d = layers.UpSampling2D(2)(t)  # 32
    d = layers.Concatenate()([d, encoded[-1][2]])
    d = _conv_bn(d, 128)

    d = layers.UpSampling2D(2)(d)  # 64
    d = layers.Concatenate()([d, encoded[-1][1]])
    d = _conv_bn(d, 64)

    d = layers.UpSampling2D(2)(d)  # 128
    d = _conv_bn(d, 32)
    d = layers.UpSampling2D(2)(d)  # 256
    d = _conv_bn(d, 16)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d)
    model = models.Model(inputs, outputs, name=f"SwinST_seq{seq_len}")
    model.compile(optimizer=_adam(), loss=combined_loss, metrics=[dice_coefficient, iou_metric])
    return model


def build_vit_st(
    seq_len: int, input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), patch_size: int = 16, embed_dim: int = 128
):
    """
    Architecture 5 – ViT / ViViT spatio-temporal model (thesis Figure 5.5).

    Per-frame patch tokenisation → spatial transformer → global average
    pooling → temporal transformer → dense projection → convolutional
    decoder.
    """
    inputs = layers.Input(shape=(seq_len, *input_shape))
    h_out = input_shape[0] // patch_size
    w_out = input_shape[1] // patch_size
    n_patches = h_out * w_out

    # Shared spatial encoder weights (reused across frames so all frames are
    # embedded by the same patch-embedding + attention sub-network).
    patch_embed = layers.Conv2D(
        embed_dim, patch_size, strides=patch_size, padding="valid", name="patch_embed"
    )
    token_norm = layers.LayerNormalization(name="token_norm")
    spatial_mha = layers.MultiHeadAttention(
        num_heads=4, key_dim=embed_dim // 4, dropout=0.1, name="spatial_mha"
    )
    post_norm = layers.LayerNormalization(name="post_norm")
    token_pool = layers.GlobalAveragePooling1D(name="token_pool")

    # Encode every frame with the shared spatial transformer.
    frames = [layers.Lambda(lambda t, i=i: t[:, i, ...])(inputs) for i in range(seq_len)]
    frame_tokens_list = []
    for f in frames:
        x = patch_embed(f)
        x = layers.Reshape((n_patches, embed_dim))(x)
        x = token_norm(x)
        x = spatial_mha(x, x)
        x = post_norm(x)
        x = token_pool(x)  # (B, embed_dim)
        frame_tokens_list.append(layers.Reshape((1, embed_dim))(x))

    # (B, T, embed_dim) -- TimeDistributed-compatible, shared weights.
    frame_tokens = layers.Concatenate(axis=1)(frame_tokens_list)

    # ── Temporal transformer ────────────────────────────────────────────────
    t = layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim // 4, dropout=0.1)(
        frame_tokens, frame_tokens
    )
    t = layers.LayerNormalization()(t)
    t = layers.Dense(256, activation="gelu")(t)
    t = layers.GlobalAveragePooling1D()(t)  # (B, 256)
    t = layers.Dense(h_out * w_out * embed_dim, activation="gelu")(t)
    t = layers.Reshape((h_out, w_out, embed_dim))(t)

    # ── Convolutional decoder ────────────────────────────────────────────────
    d = layers.Conv2DTranspose(128, 2, strides=2, padding="same", activation="relu")(t)  # 32
    d = layers.Conv2DTranspose(64, 2, strides=2, padding="same", activation="relu")(d)  # 64
    d = layers.Conv2DTranspose(32, 2, strides=2, padding="same", activation="relu")(d)  # 128
    d = layers.Conv2DTranspose(16, 2, strides=2, padding="same", activation="relu")(d)  # 256
    d = _conv_bn(d, 16)

    outputs = layers.Conv2D(1, 1, padding="same", activation="sigmoid")(d)
    model = models.Model(inputs, outputs, name=f"ViT_ST_seq{seq_len}")
    model.compile(optimizer=_adam(), loss=combined_loss, metrics=[dice_coefficient, iou_metric])
    return model


# Registry: name → builder, so notebooks can stay generic.
MODEL_REGISTRY: Dict[str, Callable[[int], "tf.keras.Model"]] = {
    "convlstm": build_convlstm,
    "unet_lstm": build_unet_lstm,
    "attention_unet_convlstm": build_attention_unet_convlstm,
    "swin_st": build_swin_st,
    "vit_st": build_vit_st,
}


def build_model(name: str, seq_len: int):
    """Build and compile a model from the registry by its string name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. " f"Options: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](seq_len)


# =============================================================================
# 6.  TRAINING CALLBACKS
# =============================================================================


class StructuredTrainingLogger(Callback):
    """Compact, aligned one-line-per-epoch training logger."""

    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self.best_val_loss = np.inf

    def on_train_begin(self, logs=None):
        header = (
            f"{'Ep':>4s}/{'Tot':<4s} | {'Loss':>8s} | {'VLoss':>8s} | "
            f"{'Dice':>6s} | {'VDice':>6s} | {'IoU':>6s} | "
            f"{'VIoU':>6s} | {'LR':>9s} | Note"
        )
        print("-" * len(header))
        print(header)
        print("-" * len(header))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        vloss = logs.get("val_loss", 0)
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        note = ""
        if vloss < self.best_val_loss:
            self.best_val_loss = vloss
            note = "saved"
        print(
            f"{epoch + 1:>4d}/{self.total_epochs:<4d} | "
            f"{logs.get('loss', 0):>8.4f} | {vloss:>8.4f} | "
            f"{logs.get('dice_coefficient', 0):>6.4f} | "
            f"{logs.get('val_dice_coefficient', 0):>6.4f} | "
            f"{logs.get('iou_metric', 0):>6.4f} | "
            f"{logs.get('val_iou_metric', 0):>6.4f} | {lr:>9.2e} | {note}"
        )

    def on_train_end(self, logs=None):
        print("-" * 85)
        print(f"  Training finished  |  Best val_loss: {self.best_val_loss:.4f}")
        print("-" * 85)


def create_callbacks(
    model_name: str,
    checkpoint_dir: str = "checkpoints",
    epochs: int = DEFAULT_EPOCHS,
    patience: int = EARLY_STOP_PATIENCE,
):
    """Standard callback set (checkpoint, early stop, LR schedule, logger)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.keras")
    return [
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, mode="min", verbose=0),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=0
        ),
        StructuredTrainingLogger(total_epochs=epochs),
    ]


# =============================================================================
# 7.  EVALUATION
# =============================================================================


def evaluate_model(
    model,
    X_test,
    y_test,
    target_years_test,
    pixel_area_km2: float,
    model_label: str,
    setup_name: str,
    include_persistence: bool = True,
) -> pd.DataFrame:
    """
    Evaluate a trained model (and optionally a persistence baseline) on the
    test set, returning a tidy per-sample DataFrame with IoU, Dice,
    Precision, Recall and signed area difference.
    """
    pred = model.predict(X_test, verbose=0)
    comparisons = [(model_label, pred)]
    if include_persistence:
        comparisons.append(("Persistence", X_test[:, -1, :, :, :]))

    rows = []
    for name, predictions in comparisons:
        for i in range(len(y_test)):
            yt = y_test[i, :, :, 0]
            yp = predictions[i, :, :, 0]
            yt_flat = (yt.flatten() > BINARY_THRESHOLD).astype(int)
            yp_flat = (yp.flatten() > BINARY_THRESHOLD).astype(int)
            rows.append(
                {
                    "Setup": setup_name,
                    "Model": name,
                    "Year": target_years_test[i],
                    "IoU": iou_np(yt, yp),
                    "Dice": dice_np(yt, yp),
                    "Precision": precision_score(yt_flat, yp_flat, zero_division=0),
                    "Recall": recall_score(yt_flat, yp_flat, zero_division=0),
                    "Area_Diff_km2": calculate_area_difference(yt, yp, pixel_area_km2),
                }
            )
    return pd.DataFrame(rows)


def summarise_results(df_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-sample results into per (Setup, Model) means."""
    cols = ["IoU", "Dice", "Precision", "Recall", "Area_Diff_km2"]
    agg = df_results.groupby(["Setup", "Model"])[cols].mean().round(4)
    agg["Abs_Area_Diff_km2"] = (
        df_results.groupby(["Setup", "Model"])["Area_Diff_km2"]
        .apply(lambda s: s.abs().mean())
        .round(4)
    )
    return agg.reset_index()


# =============================================================================
# 8.  EXPERIMENT CONFIGURATION
# =============================================================================


@dataclass
class ExperimentConfig:
    """Encapsulates one experimental setup (resolution + temporal cutoff)."""

    resolution: str  # "yearly" | "quarterly" | "bimonthly"
    sequence_lengths: List[int] = field(default_factory=lambda: [4, 5, 6])
    cutoff_year: int = 2015
    test_label: str = "Test: 2016-2025"
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = 4


# Presets matching thesis § 5.1 / § 5.2.
EXPERIMENT_PRESETS: Dict[str, ExperimentConfig] = {
    "yearly_setup1": ExperimentConfig("yearly", [4, 5, 6], 2015, "2016-2025"),
    "yearly_setup2": ExperimentConfig("yearly", [4, 5, 6], 2020, "2021-2025"),
    "quarterly_setup1": ExperimentConfig("quarterly", [6, 8, 10], 2015, "2015-Q1 onwards"),
    "quarterly_setup2": ExperimentConfig("quarterly", [6, 8, 10], 2020, "2020-Q1 onwards"),
    "bimonthly_setup1": ExperimentConfig("bimonthly", [6, 9, 12], 2015, "long-range"),
    "bimonthly_setup2": ExperimentConfig("bimonthly", [6, 9, 12], 2020, "medium-range"),
}


def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpu() -> None:
    """Enable memory growth on all available GPUs (or warn if none)."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU configured: {len(gpus)} device(s)")
        except RuntimeError as exc:  # pragma: no cover
            print(f"GPU error: {exc}")
    else:
        print("No GPU found – running on CPU")
