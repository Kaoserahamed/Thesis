"""Inline-code templates for the best-model prediction notebooks."""

from __future__ import annotations

PRED_PRELUDE = """# ============================================================================
# Environment, imports and configuration  (best-model final prediction)
# ============================================================================
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

np.random.seed(42)
tf.random.set_seed(42)
for _gpu in tf.config.experimental.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

# ---- Configuration ---------------------------------------------------------
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

RESOLUTION        = "%%RES%%"
SEQ_LEN           = %%SEQLEN%%
MODEL_LABEL       = "%%LABEL%%"
MODEL_KEY         = "%%KEY%%"
T_FORECAST        = %%TFC%%
FORECAST_LABELS   = %%FLABELS%%
SEASONAL_CHANNELS = %%SEASONAL%%
OUTPUT_DIR        = os.path.join("outputs", RESOLUTION, "forecast")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Data directory :", DATA_DIR)
print("Best model     :", MODEL_LABEL, "(", MODEL_KEY, ") | L =", SEQ_LEN)
print("Forecast       :", T_FORECAST, "steps ->",
      FORECAST_LABELS[0], "...", FORECAST_LABELS[-1])
"""

PRED_DATA = """# ============================================================================
# Load the full %%LOWER%% water-mask record
# ============================================================================
images, years = load_image_stack(DATA_DIR)
print(f"Loaded {len(images)} %%PERIOD%% frames: {years[0]} ... {years[-1]}")
print("Frame shape :", images[0].shape)

PIXEL_AREA_KM2 = get_pixel_area_km2(build_catalog(DATA_DIR).iloc[0]["filepath"])
print(f"Pixel ground area : {PIXEL_AREA_KM2:.6f} km2")
"""


PRED_FORECAST = '''# ============================================================================
# Train the best model on the full record and roll the forecast forward
# ============================================================================
%%CALLBACKS%%
%%FORECAST_HELPERS%%

N_CHANNELS = 3 if SEASONAL_CHANNELS else 1


def to_cube(frames):
    """Stack frames into a (T, H, W, C) tensor (see utils/forecast_utils.py)."""
    return frames_to_cube(frames, seasonal=SEASONAL_CHANNELS)


def make_sequences(cube, seq_len, horizon=1):
    """Sliding-window sequences (stride 1); the target is the next frame."""
    return make_forecast_sequences(cube, seq_len, horizon=horizon)


cube = to_cube(images)
X, y = make_sequences(cube, SEQ_LEN)
print("Training windows :", X.shape)

tf.keras.backend.clear_session()
model = %%BUILDER%%(SEQ_LEN, input_shape=(IMG_HEIGHT, IMG_WIDTH, N_CHANNELS))
model.fit(X, y, epochs=DEFAULT_EPOCHS, batch_size=BATCH_SIZE,
          callbacks=create_callbacks(MODEL_KEY, checkpoint_dir=OUTPUT_DIR),
          verbose=0)

# ---- Autoregressive roll-out -----------------------------------------------
window = cube[-SEQ_LEN:][None, ...]
baseline = float((images[-1] > BINARY_THRESHOLD).sum()) * PIXEL_AREA_KM2
areas, masks = [], []

for step in range(T_FORECAST):
    pred = model.predict(window, verbose=0)[0, :, :, 0]
    mask = (pred > BINARY_THRESHOLD).astype(np.float32)
    masks.append(mask)
    areas.append(float(mask.sum()) * PIXEL_AREA_KM2)

    if SEASONAL_CHANNELS:
        phase = 2.0 * np.pi * (((len(images) + step) % 6) / 6.0)
        H, W = pred.shape
        nxt = np.stack([pred,
                        np.full((H, W), np.sin(phase), dtype=np.float32),
                        np.full((H, W), np.cos(phase), dtype=np.float32)],
                       axis=-1)[None, ...]
    else:
        nxt = pred[None, ..., None]
    window = np.concatenate([window[:, 1:, ...], nxt], axis=1)

masks = np.stack(masks, axis=0)
df_fc = pd.DataFrame({"Period": FORECAST_LABELS, "Area_km2": np.round(areas, 2)})
df_fc["Delta_km2"] = np.round(df_fc["Area_km2"] - baseline, 2)
df_fc.to_csv(os.path.join(OUTPUT_DIR, "forecast_areas.csv"), index=False)

print()
print(df_fc.to_string(index=False))
print()
print(f"Baseline (last observed) : {baseline:,.2f} km2")
print(f"Final forecast           : {areas[-1]:,.2f} km2")
print(f"Net change               : {areas[-1] - baseline:+,.2f} km2 "
      f"({100 * (areas[-1] - baseline) / baseline:+.1f}%)")
'''

PRED_VIZ = """# ============================================================================
# Forecast area trend, change-frequency (risk) map and sample masks
# ============================================================================
# -- 1. Forecast water-area trend --------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(areas)), areas, marker="o", color="tab:blue")
ax.axhline(baseline, color="grey", ls="--", label="baseline (last observed)")
ax.set_xticks(range(len(areas)))
ax.set_xticklabels(FORECAST_LABELS, rotation=90)
ax.set_ylabel("water area (km²)")
ax.set_title(f"{RESOLUTION} forecast -- {MODEL_LABEL} (L={SEQ_LEN})")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "forecast_area_trend.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# -- 2. Water-occurrence frequency (erosion / accretion risk map) ------------
freq = masks.mean(axis=0)
fig, ax = plt.subplots(figsize=(6.5, 6))
im = ax.imshow(freq, cmap="viridis", vmin=0, vmax=1)
ax.set_title("Forecast water-occurrence frequency (risk map)")
ax.axis("off")
plt.colorbar(im, fraction=0.046, shrink=0.85)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "forecast_risk_map.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# -- 3. A few forecast masks -------------------------------------------------
n_show = min(6, T_FORECAST)
idx = np.linspace(0, T_FORECAST - 1, n_show, dtype=int)
fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3.4))
for ax, i in zip(np.atleast_1d(axes), idx):
    ax.imshow(masks[i], cmap="Blues", vmin=0, vmax=1)
    ax.set_title(FORECAST_LABELS[i], fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "forecast_samples.png"),
            dpi=150, bbox_inches="tight")
plt.show()
"""
