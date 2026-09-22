"""Training, evaluation and the public API surface for the thesis models.

TensorFlow-dependent helpers shared by every notebook in ``models/``: the
structured epoch logger, the standard callback set, the evaluation and
summarisation utilities, seeding and GPU configuration.

The reviewed building blocks live in sibling modules and are re-exported here,
so the historical import style

    from utils.model_utils import create_sequences, iou_np, build_model

keeps working unchanged:

* :mod:`utils.model_losses`        -- BCE/Dice objectives and Keras metrics
* :mod:`utils.model_architectures` -- the five architectures and the registry
* :mod:`utils.pipeline_utils`      -- data pipeline, sequences, temporal split
* :mod:`utils.metrics_numpy`       -- NumPy IoU / Dice / area metrics

Reference: "Analyzing & Forecasting River Morphological Evolution Using Machine
Learning & Spatiotemporal Neural Models" -- Chapter 5 (Methodology).
"""

from __future__ import annotations

import logging
import os

from .tf_env import silence_tf

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from sklearn.metrics import precision_score, recall_score

from .metrics_numpy import (
    BINARY_THRESHOLD,
    calculate_area_difference,
    dice_np,
    iou_np,
)
from .model_architectures import (
    MODEL_REGISTRY,
    AttentionGate,
    build_attention_unet_convlstm,
    build_convlstm,
    build_model,
    build_swin_st,
    build_unet_lstm,
    build_vit_st,
)
from .model_losses import (
    bce_loss,
    combined_loss,
    dice_coefficient,
    dice_loss,
    iou_metric,
)
from .pipeline_utils import (
    CLIPNORM,
    DEFAULT_EPOCHS,
    EARLY_STOP_PATIENCE,
    EXPERIMENT_PRESETS,
    IMG_HEIGHT,
    IMG_WIDTH,
    LEARNING_RATE,
    N_COMPONENTS,
    REDUCE_LR_PATIENCE,
    ExperimentConfig,
    build_catalog,
    create_sequences,
    get_pixel_area_km2,
    keep_largest_n_components_cv2,
    load_and_preprocess_image,
    load_image_stack,
    prepare_split,
    seed_numpy,
)

silence_tf()

logger = logging.getLogger(__name__)

__all__ = [
    "BINARY_THRESHOLD",
    "CLIPNORM",
    "DEFAULT_EPOCHS",
    "EARLY_STOP_PATIENCE",
    "EXPERIMENT_PRESETS",
    "IMG_HEIGHT",
    "IMG_WIDTH",
    "LEARNING_RATE",
    "MODEL_REGISTRY",
    "N_COMPONENTS",
    "REDUCE_LR_PATIENCE",
    "AttentionGate",
    "ExperimentConfig",
    "StructuredTrainingLogger",
    "bce_loss",
    "build_attention_unet_convlstm",
    "build_catalog",
    "build_convlstm",
    "build_model",
    "build_swin_st",
    "build_unet_lstm",
    "build_vit_st",
    "calculate_area_difference",
    "combined_loss",
    "configure_gpu",
    "create_callbacks",
    "create_sequences",
    "dice_coefficient",
    "dice_loss",
    "dice_np",
    "evaluate_model",
    "get_pixel_area_km2",
    "iou_metric",
    "iou_np",
    "keep_largest_n_components_cv2",
    "load_and_preprocess_image",
    "load_image_stack",
    "prepare_split",
    "seed_everything",
    "seed_numpy",
    "summarise_results",
]


class StructuredTrainingLogger(Callback):
    """Compact, aligned one-line-per-epoch training logger.

    Emits through the :mod:`logging` framework (logger ``utils.model_utils``)
    instead of ``print`` so that long training runs can be captured, filtered
    and timestamped by the caller's logging configuration.
    """

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
        logger.info("-" * len(header))
        logger.info(header)
        logger.info("-" * len(header))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        vloss = logs.get("val_loss", 0)
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        note = ""
        if vloss < self.best_val_loss:
            self.best_val_loss = vloss
            note = "saved"
        logger.info(
            "%4d/%-4d | %8.4f | %8.4f | %6.4f | %6.4f | %6.4f | %6.4f | %9.2e | %s",
            epoch + 1,
            self.total_epochs,
            logs.get("loss", 0),
            vloss,
            logs.get("dice_coefficient", 0),
            logs.get("val_dice_coefficient", 0),
            logs.get("iou_metric", 0),
            logs.get("val_iou_metric", 0),
            lr,
            note,
        )

    def on_train_end(self, logs=None):
        logger.info("-" * 85)
        logger.info("  Training finished  |  Best val_loss: %.4f", self.best_val_loss)
        logger.info("-" * 85)


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


def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training.

    Seeds both the NumPy global RNG (used by the data pipeline) and the
    TensorFlow global RNG (used for weight initialisation and dropout), which
    together guarantee byte-identical runs for a fixed seed.
    """
    seed_numpy(seed)
    tf.random.set_seed(seed)


def configure_gpu() -> None:
    """Enable memory growth on all available GPUs (or log that none was found)."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("GPU configured: %d device(s)", len(gpus))
        except RuntimeError as exc:  # pragma: no cover
            logger.warning("GPU configuration failed: %s", exc)
    else:
        logger.info("No GPU found - running on CPU")
