"""
Self-contained notebook generator for the River Morphology thesis repository.
============================================================================

This script writes **stand-alone** Jupyter notebooks for every architecture at
every temporal resolution plus the best-model final-prediction notebooks.

Why this exists
---------------
The original notebooks that produced the thesis results were flat, single-cell
files.  To make the experiments reproducible and readable, this generator emits
well-structured, *self-contained* notebooks: each notebook carries the full
pipeline inline -- configuration, preprocessing, losses/metrics, sequence
generation, the strict temporal split, the model architecture, the training
loop and the evaluation -- so a reader can run one file end-to-end without
chasing shared modules.

The architecture and pipeline code is **not** re-typed by hand.  It is extracted
directly from :mod:`utils.model_utils` at generation time (via :mod:`ast`), so
the inlined code is guaranteed to match the reviewed, report-accurate
implementation.  ``utils/model_utils.py`` therefore remains the single source of
truth for the algorithms, while the notebooks stay self-contained.

Run from the repository root::

    python scripts/generate_notebooks.py

It (re)writes:

    models/yearly/01_yearly_convlstm.ipynb
    models/yearly/02_yearly_unet_lstm.ipynb
    models/yearly/03_yearly_attention_unet_convlstm.ipynb
    models/yearly/04_yearly_swin_transformer.ipynb
    models/yearly/05_yearly_vit.ipynb
    models/quarterly/01_quarterly_convlstm.ipynb
    models/quarterly/02_quarterly_unet_lstm.ipynb
    models/quarterly/03_quarterly_attention_unet_convlstm.ipynb
    models/quarterly/04_quarterly_swin_transformer.ipynb
    models/quarterly/05_quarterly_vit.ipynb
    models/bimonthly/01_bimonthly_convlstm.ipynb
    models/bimonthly/02_bimonthly_unet_lstm.ipynb
    models/bimonthly/03_bimonthly_attention_unet_convlstm.ipynb
    models/bimonthly/04_bimonthly_swin_transformer.ipynb
    models/bimonthly/05_bimonthly_vit.ipynb
    models/00_{yearly,quarterly,bimonthly}_model_comparison.ipynb
    analysis/02_long_term_prediction.ipynb      (yearly, best model)
    analysis/03_short_term_prediction.ipynb     (bi-monthly, best model)
    analysis/04_quarterly_prediction.ipynb      (quarterly, best model)

Every resolution carries the same five notebooks, numbered ``01``..``05`` in the
canonical architecture order, so the three folders line up one-to-one:

    01 ConvLSTM | 02 U-Net + LSTM | 03 Attention U-Net + ConvLSTM
    04 Swin ST  | 05 ViT

Reference: ``docs/predefence_report.md`` -- Chapter 5 (architectures,
§5.1-§5.9) and Chapter 7 (results, §7.1-§7.5).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
MODEL_UTILS = ROOT / "utils" / "model_utils.py"


# ---------------------------------------------------------------------------
# 1.  Extract reviewed source from utils/model_utils.py (single source of truth)
# ---------------------------------------------------------------------------


def _load_mu_lines() -> list:
    return MODEL_UTILS.read_text(encoding="utf-8").splitlines(keepends=True)


_MU_SRC = MODEL_UTILS.read_text(encoding="utf-8")
_MU_TREE = ast.parse(_MU_SRC)
_MU_LINES = _load_mu_lines()


def grab(name: str) -> str:
    """Return the source of a top-level function/class in model_utils.py."""
    for node in _MU_TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                start = node.lineno - 1
                for dec in node.decorator_list:
                    start = min(start, dec.lineno - 1)
                end = node.end_lineno
                return "".join(_MU_LINES[start:end]).rstrip() + "\n"
    raise KeyError(f"'{name}' not found in {MODEL_UTILS.name}")


# ---------------------------------------------------------------------------
# 2.  Notebook cell helpers
# ---------------------------------------------------------------------------


def md(text: str) -> dict:
    body = text.strip("\n") + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": body.splitlines(keepends=True)}


def code(text: str) -> dict:
    body = text.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": body.splitlines(keepends=True),
    }


def write_notebook(path: Path, cells: list) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 3.  Experiment metadata (mirrors docs/predefence_report.md §5.1 / §7)
# ---------------------------------------------------------------------------

RESOLUTIONS: Dict[str, Dict[str, Any]] = {
    "yearly": {
        "label": "Yearly",
        "sequence_lengths": [4, 5, 6],
        "period_name": "year",
        "env_var": "YEARLY_DIR",
        "best_key": "attention_unet_convlstm",
        "best_note": (
            "Best architecture for the yearly horizon (report §7.1): "
            "**Attention U-Net + ConvLSTM**, Setup 1, L=5 -- "
            "IoU = 0.7005, Dice = 0.8236."
        ),
    },
    "quarterly": {
        "label": "Quarterly",
        "sequence_lengths": [6, 8, 10],
        "period_name": "quarter",
        "env_var": "QUARTERLY_DIR",
        "best_key": "attention_unet_convlstm",
        "best_note": (
            "Best architecture for the quarterly horizon (report §7.2): "
            "**Attention U-Net + ConvLSTM**, Setup 1 -- "
            "IoU = 0.6956, Dice = 0.8139."
        ),
    },
    "bimonthly": {
        "label": "Bi-monthly",
        "sequence_lengths": [6, 9, 12],
        "period_name": "bi-month",
        "env_var": "BIMONTHLY_DIR",
        "best_key": "unet_lstm",
        "best_note": (
            "Best architecture for the bi-monthly horizon (report §7.3): "
            "**U-Net + LSTM**, Setup 2, L=9 -- "
            "IoU = 0.7791, Dice = 0.8701."
        ),
    },
}

# Every architecture, its Keras builder, the helper symbols that must travel
# with it, the report section that specifies it, and the delivery file stem.
ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "convlstm": {
        "label": "ConvLSTM",
        "builder": "build_convlstm",
        "helpers": [],
        "section": "§5.5",
        "figure": "Figure 5.1",
        "summary": (
            "A standalone spatiotemporal recurrent network: per-frame "
            "convolutional feature extraction followed by stacked "
            "ConvLSTM2D layers that model temporal evolution and a "
            "lightweight decoder that restores the full resolution."
        ),
        "short": "convlstm",
    },
    "unet_lstm": {
        "label": "U-Net + LSTM",
        "builder": "build_unet_lstm",
        "helpers": ["_conv_bn"],
        "section": "§5.6",
        "figure": "Figure 5.2",
        "summary": (
            "An encoder-decoder network whose encoder is applied per "
            "frame and whose ConvLSTM bottleneck captures temporal "
            "dependencies; skip connections preserve fine river "
            "boundaries. Best model for the bi-monthly horizon."
        ),
        "short": "unet_lstm",
    },
    "attention_unet_convlstm": {
        "label": "Attention U-Net + ConvLSTM",
        "builder": "build_attention_unet_convlstm",
        "helpers": ["_conv_bn", "AttentionGate"],
        "section": "§5.7",
        "figure": "Figure 5.3",
        "summary": (
            "A U-Net with a ConvLSTM bottleneck and additive "
            "attention-gated skip connections (Oktay et al., 2018) "
            "that focus the decoder on the spatially relevant regions. "
            "Best model for the yearly and quarterly horizons."
        ),
        "short": "attention_unet_convlstm",
    },
    "swin_st": {
        "label": "Swin Spatio-Temporal Transformer",
        "builder": "build_swin_st",
        "helpers": ["_conv_bn"],
        "section": "§5.8",
        "figure": "Figure 5.4",
        "summary": (
            "A hierarchical window-attention encoder shared across "
            "frames, a temporal-fusion head and a U-Net-style "
            "convolutional decoder."
        ),
        "short": "swin_transformer",
    },
    "vit_st": {
        "label": "Vision Transformer (ViViT)",
        "builder": "build_vit_st",
        "helpers": ["_conv_bn"],
        "section": "§5.9",
        "figure": "Figure 5.5",
        "summary": (
            "Per-frame patch tokenisation with a shared spatial "
            "transformer, a temporal transformer across frames, and a "
            "convolutional decoder that reconstructs the mask."
        ),
        "short": "vit",
    },
}

# Canonical presentation order of the five architectures.  Notebooks are numbered
# 01..05 in exactly this order, identically for all three resolutions, so the
# yearly/, quarterly/ and bimonthly/ folders stay parallel:
#
#     01_convlstm  02_unet_lstm  03_attention_unet_convlstm
#     04_swin_transformer  05_vit
ARCH_ORDER: list = [
    "convlstm",
    "unet_lstm",
    "attention_unet_convlstm",
    "swin_st",
    "vit_st",
]
assert list(ARCHITECTURES) == ARCH_ORDER, "ARCHITECTURES must follow ARCH_ORDER"


def arch_number(arch_key: str) -> int:
    """1-based position of an architecture in the canonical order (1..5)."""
    return ARCH_ORDER.index(arch_key) + 1


def stem_for(res_key: str, arch_key: str) -> str:
    """Notebook file stem, e.g. ``03_yearly_attention_unet_convlstm``."""
    return f"{arch_number(arch_key):02d}_{res_key}_{ARCHITECTURES[arch_key]['short']}"


SETUP_CONFIGS = [
    ("Setup 1", 2015, "2016-2025"),
    ("Setup 2", 2020, "2021-2025"),
]


# ---------------------------------------------------------------------------
# 4.  Inlined code templates (assembled from the model_utils.py source)
# ---------------------------------------------------------------------------


def _banner(title: str) -> str:
    bar = "# " + "=" * 76
    return bar + "\n# " + title + "\n" + bar + "\n"


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


def _title_markdown(res_key: str, arch_key: str) -> str:
    res, arch = RESOLUTIONS[res_key], ARCHITECTURES[arch_key]
    return (
        f"# {arch['label']} — {res['label']} Resolution\n\n"
        f"**Architecture {arch_number(arch_key)} of {len(ARCH_ORDER)}** "
        f"(`{stem_for(res_key, arch_key)}.ipynb`)\n\n"
        "**Stand-alone training notebook.** This file inlines the complete "
        "pipeline — configuration, preprocessing, losses/metrics, sequence "
        "generation, the strict temporal split, the `"
        + arch["label"]
        + "` architecture, training and evaluation — so it can "
        "be executed end-to-end without importing any project module.\n\n"
        f"- **Architecture:** `{arch['label']}` ({arch['section']}, "
        f"{arch['figure']}): {arch['summary']}\n"
        f"- **Temporal resolution:** {res['label']} ({res['period_name']} composite)\n"
        f"- **Sequence lengths L:** {res['sequence_lengths']}\n"
        "- **Temporal split:** Setup 1 (train ≤ 2015 / test 2016–2025) and "
        "Setup 2 (train ≤ 2020 / test 2021–2025)\n"
        "- **Loss:** `L_BCE + L_Dice`; optimiser Adam(lr=1e-4, clipnorm=1.0)\n"
        "- **Evaluation:** IoU, Dice, Precision, Recall and signed area "
        "difference ΔA (km²)\n\n"
        f"> {res['best_note']}\n"
    )


def model_notebook(res_key: str, arch_key: str) -> None:
    """Write one self-contained architecture notebook."""
    res, arch = RESOLUTIONS[res_key], ARCHITECTURES[arch_key]
    lower = res["label"].lower()

    def fill(t: str) -> str:
        for token, value in (
            ("%%RES%%", res_key),
            ("%%LOWER%%", lower),
            ("%%PERIOD%%", res["period_name"]),
            ("%%ENVVAR%%", res["env_var"]),
            ("%%SEQLENS%%", repr(res["sequence_lengths"])),
            ("%%LABEL%%", arch["label"]),
            ("%%KEY%%", arch_key),
            ("%%BUILDER%%", arch["builder"]),
            ("%%SECTION%%", arch["section"]),
            ("%%FIGURE%%", arch["figure"]),
        ):
            t = t.replace(token, value)
        t = t.replace("%%CODE%%", architecture_code(arch_key))
        t = t.replace("%%CALLBACKS%%", callbacks_code())
        return t

    cells = [
        md(_title_markdown(res_key, arch_key)),
        md(
            "## 1. Configuration and imports\n\n"
            "Environment setup, imports, random seeds and the experiment "
            "configuration (thesis §5.9)."
        ),
        code(fill(PRELUDE)),
        md(
            "## 2. Preprocessing, losses, metrics and data pipeline\n\n"
            "These helpers are copied verbatim from the reviewed "
            "`utils/model_utils.py`, keeping this notebook fully self-contained: "
            "connected-component cleaning, water-mask loading, the BCE + Dice "
            "loss (equations 5.7–5.9), the IoU/Dice/area metrics (equations "
            "5.2–5.6), sliding-window sequences (§5.1) and the leakage-proof "
            "temporal split (§5.2)."
        ),
        code(shared_pipeline_code()),
        md(
            f"## 3. Load the {lower} water-mask time series\n\n"
            "Every GeoTIFF in the data directory is read, cleaned and resized "
            "to 256×256 binary masks."
        ),
        code(fill(LOAD_DATA)),
        md(
            f"## 4. Model architecture — {arch['label']}\n\n"
            f"Definition of the {arch['label']} network ({arch['section']}, "
            f"{arch['figure']}), inlined below."
        ),
        code(fill(ARCH_CELL)),
        md(
            "## 5. Training across sequence lengths and temporal setups\n\n"
            "The architecture is trained for every sequence length "
            f"{res['sequence_lengths']} under both temporal setups, with "
            "`ModelCheckpoint`, `EarlyStopping` and `ReduceLROnPlateau`."
        ),
        code(fill(TRAIN_CELL)),
        md("## 6. Comparative summary"),
        code(fill(SUMMARY_CELL)),
        md("## 7. Visualisations"),
        code(fill(VIZ_CELL)),
    ]
    path = ROOT / "models" / res_key / (stem_for(res_key, arch_key) + ".ipynb")
    write_notebook(path, cells)


# ---------------------------------------------------------------------------
# 6.  Best-model final-prediction notebooks
# ---------------------------------------------------------------------------

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

N_CHANNELS = 3 if SEASONAL_CHANNELS else 1


def to_cube(frames):
    """Stack frames into a (T, H, W, C) tensor, optionally adding sin/cos
    channels that encode the position inside the annual cycle."""
    stack = np.stack(frames, axis=0).astype(np.float32)
    if not SEASONAL_CHANNELS:
        return np.expand_dims(stack, -1)
    T, H, W = stack.shape
    cube = np.empty((T, H, W, 3), dtype=np.float32)
    for t in range(T):
        phase = 2.0 * np.pi * ((t % 6) / 6.0)
        cube[t, ..., 0] = stack[t]
        cube[t, ..., 1] = np.sin(phase)
        cube[t, ..., 2] = np.cos(phase)
    return cube


def make_sequences(cube, seq_len, horizon=1):
    """Sliding-window sequences (stride 1); the target is the next frame."""
    X, y = [], []
    for i in range(len(cube) - seq_len - horizon + 1):
        X.append(cube[i:i + seq_len])
        y.append(cube[i + seq_len + horizon - 1, ..., 0])
    return np.array(X), np.expand_dims(np.array(y), -1)


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


FORECASTS: Dict[str, Dict[str, Any]] = {
    "yearly": {
        "stem": "02_long_term_prediction",
        "seq_len": 5,
        "seasonal": False,
        "steps": 15,
        "labels": [str(y) for y in range(2026, 2041)],
        "title": "Long-Term Morphological Forecast (2026–2040)",
        "report": "section 7.4",
    },
    "quarterly": {
        "stem": "04_quarterly_prediction",
        "seq_len": 8,
        "seasonal": False,
        "steps": 16,
        "labels": [str(2026 + i // 4) + " Q" + str(i % 4 + 1) for i in range(16)],
        "title": "Quarterly Morphological Forecast (2026 Q1–2029 Q4)",
        "report": "section 7.2",
    },
    "bimonthly": {
        "stem": "03_short_term_prediction",
        "seq_len": 9,
        "seasonal": True,
        "steps": 18,
        "labels": [str(2026 + i // 6) + " P" + str(i % 6 + 1) for i in range(18)],
        "title": "Short-Term Morphological Forecast (2026 P1–2028 P6)",
        "report": "section 7.5",
    },
}


def _prediction_title(res_key: str) -> str:
    res, fc = RESOLUTIONS[res_key], FORECASTS[res_key]
    best = ARCHITECTURES[res["best_key"]]
    extra = " and seasonal sin/cos channels" if fc["seasonal"] else ""
    return (
        f"# {fc['title']}\n\n"
        f"**Best model for the {res['label'].lower()} horizon** "
        f"(report {fc['report']}): `{best['label']}` ({best['section']}) with "
        f"sequence length L={fc['seq_len']}{extra}.\n\n"
        "This stand-alone notebook trains the winning architecture on the "
        "complete available record and rolls the prediction forward "
        f"autoregressively to produce the {fc['steps']}-step forecast "
        f"({fc['labels'][0]} … {fc['labels'][-1]}). It writes the forecast "
        "area table, the area-trend figure and the erosion/accretion risk "
        "map to `outputs/`.\n\n"
        f"> {res['best_note']}\n"
    )


def final_prediction_notebook(res_key: str) -> None:
    """Write the best-model final-prediction notebook for one resolution."""
    res, fc = RESOLUTIONS[res_key], FORECASTS[res_key]
    lower = res["label"].lower()

    def fill(t: str) -> str:
        for token, value in (
            ("%%RES%%", res_key),
            ("%%LOWER%%", lower),
            ("%%PERIOD%%", res["period_name"]),
            ("%%ENVVAR%%", res["env_var"]),
            ("%%LABEL%%", ARCHITECTURES[res["best_key"]]["label"]),
            ("%%KEY%%", res["best_key"]),
            ("%%BUILDER%%", ARCHITECTURES[res["best_key"]]["builder"]),
            ("%%SEQLEN%%", str(fc["seq_len"])),
            ("%%TFC%%", str(fc["steps"])),
            ("%%FLABELS%%", repr(fc["labels"])),
            ("%%SEASONAL%%", "True" if fc["seasonal"] else "False"),
        ):
            t = t.replace(token, value)
        t = t.replace("%%CALLBACKS%%", callbacks_code())
        return t

    cells = [
        md(_prediction_title(res_key)),
        md(
            "## 1. Configuration and imports\n\n"
            "Environment setup, imports and the forecast configuration."
        ),
        code(fill(PRED_PRELUDE)),
        md(
            "## 2. Inlined pipeline (preprocessing, losses, metrics, split)\n\n"
            "The shared helpers, copied verbatim from `utils/model_utils.py`."
        ),
        code(shared_pipeline_code()),
        md(f"## 3. Load the full {lower} water-mask record"),
        code(fill(PRED_DATA)),
        md(
            f"## 4. Best model architecture\n\n"
            f"The {ARCHITECTURES[res['best_key']]['label']} network, inlined."
        ),
        code(architecture_code(res["best_key"])),
        md("## 5. Train on the full record and forecast autoregressively"),
        code(fill(PRED_FORECAST)),
        md("## 6. Forecast figures and risk map"),
        code(fill(PRED_VIZ)),
    ]
    path = ROOT / "analysis" / (fc["stem"] + ".ipynb")
    write_notebook(path, cells)


# ---------------------------------------------------------------------------
# 6b.  Self-contained architecture-comparison notebooks
# ---------------------------------------------------------------------------

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


COMP_TRAIN = """# ============================================================================
# Benchmark every architecture across setups and sequence lengths
# ============================================================================
%%CALLBACKS%%

rows = []
for arch_key, (arch_label, builder) in ARCH_REGISTRY.items():
    for setup_name, cfg in SETUP_CONFIGS.items():
        for seq_len in SEQUENCE_LENGTHS:
            tag = f"{arch_key}_L{seq_len}_{setup_name.replace(' ', '')}"
            print()
            print("=" * 78)
            print(f"  {arch_label} | {RESOLUTION} | {setup_name} | L={seq_len}")
            print("=" * 78)

            X_all, y_all, in_years, tgt_years = create_sequences(images, years, seq_len)
            X_tr, y_tr, X_val, y_val, X_test, y_test, ty_test = prepare_split(
                X_all, y_all, tgt_years, in_years, cfg["cutoff_year"])

            tf.keras.backend.clear_session()
            model = builder(seq_len)
            model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                      epochs=DEFAULT_EPOCHS, batch_size=BATCH_SIZE,
                      callbacks=create_callbacks(tag, checkpoint_dir=CKPT_DIR),
                      verbose=0)

            df = evaluate_model(model, X_test, y_test, ty_test,
                                PIXEL_AREA_KM2, arch_label, setup_name)
            df["L"] = seq_len
            rows.append(df)

df_all = pd.concat(rows, ignore_index=True)
df_summary = summarise_results(df_all)
print()
print("=== Mean metrics per architecture and setup ===")
df_summary
"""

COMP_VIZ = """# ============================================================================
# Architecture comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
df_summary.pivot(index="Model", columns="Setup", values="IoU").plot(
    kind="bar", ax=ax)
ax.set_ylabel("mean IoU")
ax.set_title(f"{RESOLUTION}: IoU by architecture and setup")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, f"{RESOLUTION}_architecture_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# Best architecture per setup (highest mean IoU)
best = (df_summary.sort_values("IoU", ascending=False)
        .groupby("Setup", as_index=False).first())
print("Best architecture per setup:")
best
"""


def comparison_notebook(res_key: str) -> None:
    """Write the self-contained all-architecture comparison notebook."""
    res = RESOLUTIONS[res_key]

    def fill(t: str) -> str:
        for token, value in (
            ("%%RES%%", res_key),
            ("%%LOWER%%", res["label"].lower()),
            ("%%PERIOD%%", res["period_name"]),
            ("%%ENVVAR%%", res["env_var"]),
            ("%%SEQLENS%%", repr(res["sequence_lengths"])),
            ("%%LABEL%%", "All architectures"),
            ("%%KEY%%", "comparison"),
        ):
            t = t.replace(token, value)
        t = t.replace("%%CALLBACKS%%", callbacks_code())
        return t

    title = (
        f"# Full Model Comparison — {res['label']} Resolution\n\n"
        "**Stand-alone benchmark notebook.** Trains and evaluates **all five** "
        "spatiotemporal architectures — ConvLSTM, U-Net + LSTM, Attention "
        "U-Net + ConvLSTM, Swin ST and ViT — under identical preprocessing, "
        "sequence lengths, loss and strict temporal splitting, so the "
        "comparison is provably fair. Everything needed is inlined below.\n\n"
        f"- **Temporal resolution:** {res['label']} ({res['period_name']} composite)\n"
        f"- **Sequence lengths L:** {res['sequence_lengths']}\n"
        "- **Temporal split:** Setup 1 (train ≤ 2015 / test 2016–2025) and "
        "Setup 2 (train ≤ 2020 / test 2021–2025)\n"
        "- **Loss:** `L_BCE + L_Dice`; optimiser Adam(lr=1e-4, clipnorm=1.0)\n\n"
        f"> {res['best_note']}\n"
    )

    cells = [
        md(title),
        md("## 1. Configuration and imports"),
        code(fill(PRELUDE)),
        md("## 2. Preprocessing, losses, metrics and data pipeline"),
        code(shared_pipeline_code()),
        md(f"## 3. Load the {res['label'].lower()} water-mask time series"),
        code(fill(LOAD_DATA)),
        md("## 4. All five architectures (inlined)"),
        code(all_architecture_code()),
        md("## 5. Benchmark every architecture"),
        code(fill(COMP_TRAIN)),
        md("## 6. Comparison figures"),
        code(fill(COMP_VIZ)),
    ]
    path = ROOT / "models" / ("00_" + res_key + "_model_comparison.ipynb")
    write_notebook(path, cells)


# ---------------------------------------------------------------------------
# 7.  Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Generating self-contained notebooks ...")
    for res_key in RESOLUTIONS:
        comparison_notebook(res_key)
    for res_key in RESOLUTIONS:
        for arch_key in ARCH_ORDER:
            model_notebook(res_key, arch_key)
    for res_key in FORECASTS:
        final_prediction_notebook(res_key)
    print("Done.")


if __name__ == "__main__":
    main()
