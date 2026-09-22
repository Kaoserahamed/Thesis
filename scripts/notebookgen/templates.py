"""Inline-code templates for the self-contained training notebooks.

Every ``%%TOKEN%%`` placeholder is substituted by :mod:`notebookgen.writers`;
``%%CODE%%`` and ``%%CALLBACKS%%`` are replaced with extracted ``utils/`` code.
"""

from __future__ import annotations

from .extract import _banner, grab
from .metadata import ARCHITECTURES, ARCH_ORDER


def shared_pipeline_code() -> str:
    """Losses, metrics, data pipeline, sequences, split and evaluation."""
    parts = [_banner("Losses and metrics  (thesis equations 5.6-5.9)")]
    for fn in ["dice_coefficient", "dice_loss", "bce_loss", "combined_loss", "iou_metric"]:
        parts.append(grab(fn))
    parts.append(_banner("NumPy metrics used at evaluation time"))
    for fn in ["iou_np", "dice_np", "calculate_area_difference"]:
        parts.append(grab(fn))
    parts.append(_banner("Data loading and preprocessing"))
    for fn in [
        "keep_largest_n_components_cv2",
        "load_and_preprocess_image",
        "get_pixel_area_km2",
        "build_catalog",
        "load_image_stack",
    ]:
        parts.append(grab(fn))
    parts.append(_banner("Sequence generation  (sliding window, section 5.1)"))
    parts.append(grab("create_sequences"))
    parts.append(_banner("Strict temporal split  (section 5.2)"))
    parts.append(grab("prepare_split"))
    parts.append(_banner("Evaluation helpers"))
    parts.append(grab("evaluate_model"))
    parts.append(grab("summarise_results"))
    return "\n\n".join(parts)


PRELUDE = """# ============================================================================
# Environment, imports and configuration
# ============================================================================
# Everything this notebook needs is defined inline so the file is stand-alone.
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_strict_conv_algorithm_picker=false")

import re
import glob
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("FATAL")

from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import (
    Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

import rasterio
import cv2
from skimage.transform import resize
from sklearn.metrics import precision_score, recall_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Reproducibility -------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)
for _gpu in tf.config.experimental.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

# ---- Configuration (thesis section 5.9) ------------------------------------
DATA_DIR         = os.environ.get("%%ENVVAR%%", os.path.join("data", "raw", "%%RES%%"))
IMG_HEIGHT       = 256
IMG_WIDTH        = 256
N_COMPONENTS     = 3
BINARY_THRESHOLD = 0.5
LEARNING_RATE    = 1e-4
CLIPNORM         = 1.0
DEFAULT_EPOCHS   = 200
EARLY_STOP_PATIENCE = 20
REDUCE_LR_PATIENCE  = 7
BATCH_SIZE       = 4

RESOLUTION       = "%%RES%%"
SEQUENCE_LENGTHS = %%SEQLENS%%
MODEL_LABEL      = "%%LABEL%%"
MODEL_KEY        = "%%KEY%%"
SETUP_CONFIGS = {
    "Setup 1": {"cutoff_year": 2015, "test_label": "2016-2025"},
    "Setup 2": {"cutoff_year": 2020, "test_label": "2021-2025"},
}
OUTPUT_DIR = os.path.join("outputs", RESOLUTION)
CKPT_DIR   = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

print("Data directory :", DATA_DIR)
print("Resolution     :", RESOLUTION, "| sequence lengths:", SEQUENCE_LENGTHS)
print("Model          :", MODEL_LABEL, "(", MODEL_KEY, ")")
"""

LOAD_DATA = """# ============================================================================
# Load the %%LOWER%% water-mask time series
# ============================================================================
images, years = load_image_stack(DATA_DIR)
print(f"Loaded {len(images)} %%PERIOD%% frames: {years[0]} ... {years[-1]}")
print("Frame shape :", images[0].shape)

# Ground area of one 256x256 pixel (km2), derived from the GeoTIFF georeferencing
PIXEL_AREA_KM2 = get_pixel_area_km2(build_catalog(DATA_DIR).iloc[0]["filepath"])
print(f"Pixel ground area : {PIXEL_AREA_KM2:.6f} km2")
"""


ARCH_CELL = """# ============================================================================
# Model architecture -- %%LABEL%%   (report %%SECTION%%, %%FIGURE%%)
# ============================================================================
%%CODE%%
# Sanity check: build the network for the first sequence length.
%%BUILDER%%(SEQUENCE_LENGTHS[0]).summary()
"""

TRAIN_CELL = """# ============================================================================
# Training callbacks and the multi-configuration training loop
# ============================================================================
%%CALLBACKS%%

# ---- Train every (setup, sequence-length) combination ----------------------
all_results = []
histories = {}

for setup_name, cfg in SETUP_CONFIGS.items():
    for seq_len in SEQUENCE_LENGTHS:
        tag = f"{MODEL_KEY}_L{seq_len}_{setup_name.replace(' ', '')}"
        print()
        print("=" * 78)
        print(f"  {MODEL_LABEL} | {RESOLUTION} | {setup_name} | L={seq_len}")
        print("=" * 78)

        X_all, y_all, in_years, tgt_years = create_sequences(images, years, seq_len)
        X_tr, y_tr, X_val, y_val, X_test, y_test, ty_test = prepare_split(
            X_all, y_all, tgt_years, in_years, cfg["cutoff_year"])
        print(f"  windows -> train={len(X_tr)}  val={len(X_val)}  test={len(X_test)}")

        tf.keras.backend.clear_session()
        model = %%BUILDER%%(seq_len)
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=DEFAULT_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=create_callbacks(tag, checkpoint_dir=CKPT_DIR),
            verbose=0,
        )
        histories[(setup_name, seq_len)] = hist

        df_eval = evaluate_model(
            model, X_test, y_test, ty_test,
            PIXEL_AREA_KM2, MODEL_LABEL, setup_name)
        df_eval["L"] = seq_len
        all_results.append(df_eval)

df_all = pd.concat(all_results, ignore_index=True)
df_summary = summarise_results(df_all)
df_by_len = (df_all.groupby(["Setup", "L"])
             [["IoU", "Dice", "Precision", "Recall", "Area_Diff_km2"]]
             .mean().round(4).reset_index())

print()
print("=== Mean metrics per (setup, sequence length) ===")
df_by_len
"""

SUMMARY_CELL = """# ============================================================================
# Comparative summary (mean over all test windows)
# ============================================================================
print("Mean metrics per setup (compare with docs/predefence_report.md section 7):")
df_summary
"""

VIZ_CELL = """# ============================================================================
# Visualisations: training curves and sequence-length comparison
# ============================================================================
# -- 1. Training / validation loss curves ------------------------------------
n = len(histories)
fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n), squeeze=False)
for ax, ((setup_name, seq_len), hist) in zip(axes[:, 0], histories.items()):
    ax.plot(hist.history["loss"], label="train")
    ax.plot(hist.history["val_loss"], label="validation")
    ax.set_title(f"{setup_name} | L={seq_len} | loss")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_KEY}_training_curves.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# -- 2. IoU vs sequence length for each setup --------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
for setup_name in df_by_len["Setup"].unique():
    sub = df_by_len[df_by_len["Setup"] == setup_name]
    ax.plot(sub["L"], sub["IoU"], marker="o", label=setup_name)
ax.set_xlabel("sequence length L")
ax.set_ylabel("mean IoU")
ax.set_title(f"{MODEL_LABEL} -- {RESOLUTION}: IoU vs L")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_KEY}_iou_vs_L.png"),
            dpi=150, bbox_inches="tight")
plt.show()
"""


# ---------------------------------------------------------------------------
# 5.  Builder helpers
# ---------------------------------------------------------------------------


def architecture_code(arch_key: str) -> str:
    """Inline the optimiser, helpers and the architecture builder."""
    spec = ARCHITECTURES[arch_key]
    parts = [_banner("Optimiser  (thesis section 5.9)"), grab("_adam")]
    for helper in spec["helpers"]:
        parts.append(grab(helper))
    parts.append(grab(spec["builder"]))
    return "\n\n".join(parts)


def callbacks_code() -> str:
    """Inline the training callbacks."""
    return "\n\n".join([grab("StructuredTrainingLogger"), grab("create_callbacks")])


ARCH_REGISTRY_SNIPPET = """

# ---- Registry of all five architectures under comparison -------------------
ARCH_REGISTRY = {
    "convlstm": ("ConvLSTM", build_convlstm),
    "unet_lstm": ("U-Net + LSTM", build_unet_lstm),
    "attention_unet_convlstm": ("Attention U-Net + ConvLSTM",
                                build_attention_unet_convlstm),
    "swin_st": ("Swin Spatio-Temporal Transformer", build_swin_st),
    "vit_st": ("Vision Transformer (ViViT)", build_vit_st),
}
"""


def all_architecture_code() -> str:
    """Inline every architecture builder for the comparison notebook."""
    parts = [_banner("Optimiser  (thesis section 5.9)"), grab("_adam")]
    seen = []
    for spec in ARCHITECTURES.values():
        for helper in spec["helpers"]:
            if helper not in seen:
                seen.append(helper)
    for helper in seen:
        parts.append(grab(helper))
    for key in ARCH_ORDER:
        parts.append(grab(ARCHITECTURES[key]["builder"]))
    return "\n\n".join(parts) + "\n" + ARCH_REGISTRY_SNIPPET
