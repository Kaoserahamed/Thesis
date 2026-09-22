"""
Visualization Utilities for River Morphology Prediction
=======================================================

Reusable plotting helpers shared by the analysis, model-evaluation and
prediction notebooks:

* training-curve plots (loss / Dice / IoU)
* sequence-length comparison bar charts
* actual-vs-predicted side-by-side panels with an error overlay
* temporal area-trend plots and seasonal heat-maps
* erosion / accretion frequency and instability maps
* interactive Folium risk maps (composite erosion + accretion + instability)

These helpers keep the notebooks focused on *narrative* while the plotting
boilerplate lives here.  They are deliberately dependency-light: matplotlib,
seaborn, numpy and (optionally) folium.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-safe; notebooks can override
import matplotlib.pyplot as plt  # noqa: E402

from .metrics_numpy import dice_np, iou_np  # noqa: E402  (TensorFlow-free)

logger = logging.getLogger(__name__)

try:
    import seaborn as sns

    sns.set_theme(style="whitegrid")
except ImportError:  # pragma: no cover
    sns = None

FIG_DPI = 150
COLOR_EROSION = "#d73027"
COLOR_ACCRETION = "#4575b4"
COLOR_STABLE = "#f7f7f7"


def _ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, out_path, dpi: int = FIG_DPI) -> None:
    """Save and close a figure, creating parent directories as needed."""
    out_path = Path(out_path)
    _ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved figure -> %s", out_path.name)


def plot_training_curves(history, setup_name: str, seq_len: int, out_path: Optional[str] = None):
    """
    Plot loss, Dice and IoU curves for a single trained model.

    Parameters
    ----------
    history    : Keras History object (or dict of lists)
    setup_name : label for the title
    seq_len    : input sequence length used
    out_path   : optional path to save the figure
    """
    h = history.history if hasattr(history, "history") else history

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    pairs = [
        ("loss", "val_loss", "Loss"),
        ("dice_coefficient", "val_dice_coefficient", "Dice"),
        ("iou_metric", "val_iou_metric", "IoU"),
    ]
    for ax, (train_key, val_key, label) in zip(axes, pairs):
        if train_key in h:
            ax.plot(h[train_key], label=f"Train {label}")
        if val_key in h:
            ax.plot(h[val_key], label=f"Val {label}")
        ax.set_title(f"{setup_name} | seq={seq_len} | {label}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Training Curves – {setup_name} (SeqLen={seq_len})", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_sequence_comparison(
    df_summary,
    setup_name: str,
    metrics=("IoU", "Dice", "Precision", "Recall"),
    out_path: Optional[str] = None,
):
    """
    Grouped bar chart comparing sequence lengths for one model/setup.

    ``df_summary`` must contain columns ``SeqLen`` and the metric names.
    """
    sub = df_summary
    seq_lens = sorted(sub["SeqLen"].unique())

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(metrics))
    width = 0.8 / max(len(seq_lens), 1)

    for i, sl in enumerate(seq_lens):
        rows = sub[sub["SeqLen"] == sl]
        if rows.empty:
            continue
        vals = [rows.iloc[0].get(m, 0) for m in metrics]
        offset = (i - len(seq_lens) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=f"seq={sl}")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{setup_name} – Sequence Length Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_predictions(
    y_true,
    y_pred,
    years,
    setup_name: str,
    model_label: str,
    n_show: int = 6,
    out_path: Optional[str] = None,
):
    """
    Side-by-side Actual | Predicted | Difference panels for a few samples.

    The difference panel encodes  R = false positive,  G = false negative,
    B = true positive.
    """
    n_show = min(n_show, len(y_true))
    indices = np.linspace(0, len(y_true) - 1, n_show, dtype=int)

    fig, axes = plt.subplots(n_show, 3, figsize=(15, 4 * n_show))
    if n_show == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        actual = np.asarray(y_true[idx]).squeeze()
        predicted = (np.asarray(y_pred[idx]).squeeze() > 0.5).astype(np.float32)

        diff = np.zeros((*actual.shape, 3))
        diff[:, :, 1] = actual * (1 - predicted)  # FN green
        diff[:, :, 0] = predicted * (1 - actual)  # FP red
        diff[:, :, 2] = actual * predicted  # TP blue

        year = years[idx]
        axes[row, 0].imshow(actual, cmap="gray")
        axes[row, 0].set_title(f"Actual – {year}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(predicted, cmap="gray")
        axes[row, 1].set_title(
            f"Predicted – {year}\nIoU={iou_np(actual, predicted):.4f}  "
            f"Dice={dice_np(actual, predicted):.4f}"
        )
        axes[row, 1].axis("off")

        axes[row, 2].imshow(diff)
        axes[row, 2].set_title("Difference (R=FP, G=FN, B=TP)")
        axes[row, 2].axis("off")

    fig.suptitle(
        f"{setup_name} | {model_label} | Actual vs Predicted", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    return fig


def plot_area_trend(
    years: Sequence[int],
    areas: Sequence[float],
    unit: str = "km²",
    title: str = "Water Area Over Time",
    out_path: Optional[str] = None,
):
    """Simple line + fill plot of water area over time."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(years, areas, alpha=0.25, color="steelblue")
    ax.plot(years, areas, "o-", color="steelblue", lw=2, ms=5)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Water Area ({unit})")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    return fig


def compute_change_frequencies(predicted_stack: np.ndarray, baseline: Optional[np.ndarray] = None):
    """
    Compute erosion and accretion frequency maps from a stack of binary
    predictions (thesis § 5.11).

    Convention: 1 = land, 0 = water.
      erosion   :  L_t = 1 (land)  ->  L_{t+1} = 0 (water)
      accretion :  L_t = 0 (water) ->  L_{t+1} = 1 (land)

    If ``baseline`` is given, transitions are computed against the baseline
    for every frame; otherwise consecutive frames are compared.

    Returns
    -------
    erosion_freq, accretion_freq, instability : float arrays in [0, 1]
    """
    stack = (np.asarray(predicted_stack) > 0.5).astype(np.uint8)
    T = stack.shape[0]

    if baseline is not None:
        base = (np.asarray(baseline) > 0.5).astype(np.uint8)
        erosion = np.zeros(base.shape, dtype=np.float32)
        accretion = np.zeros(base.shape, dtype=np.float32)
        for t in range(T):
            erosion += ((base == 1) & (stack[t] == 0)).astype(np.float32)
            accretion += ((base == 0) & (stack[t] == 1)).astype(np.float32)
        erosion /= T
        accretion /= T
    else:
        erosion = np.zeros(stack.shape[1:], dtype=np.float32)
        accretion = np.zeros(stack.shape[1:], dtype=np.float32)
        for t in range(T - 1):
            erosion += ((stack[t] == 1) & (stack[t + 1] == 0)).astype(np.float32)
            accretion += ((stack[t] == 0) & (stack[t + 1] == 1)).astype(np.float32)
        erosion /= max(T - 1, 1)
        accretion /= max(T - 1, 1)

    f_water = 1.0 - stack.mean(axis=0)
    instability = 1.0 - np.abs(2 * f_water - 1)
    return erosion, accretion, instability


def plot_risk_maps(
    erosion: np.ndarray,
    accretion: np.ndarray,
    instability: np.ndarray,
    title: str = "Morphological Risk",
    out_path: Optional[str] = None,
):
    """Three-panel figure: erosion frequency, accretion frequency, instability."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(erosion, cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title("Erosion Frequency (land→water)")
    plt.colorbar(im0, ax=axes[0], fraction=0.03, pad=0.02)

    im1 = axes[1].imshow(accretion, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Accretion Frequency (water→land)")
    plt.colorbar(im1, ax=axes[1], fraction=0.03, pad=0.02)

    im2 = axes[2].imshow(instability, cmap="magma", vmin=0, vmax=1)
    axes[2].set_title("Instability Index")
    plt.colorbar(im2, ax=axes[2], fraction=0.03, pad=0.02)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout()

    if out_path:
        save_figure(fig, out_path)
    return fig


def create_folium_risk_map(
    erosion: np.ndarray,
    accretion: np.ndarray,
    instability: np.ndarray,
    centre: Sequence[float] = (23.743, 89.67),
    zoom: int = 9,
    bounds: Optional[Sequence[Sequence[float]]] = None,
    out_path: Optional[str] = None,
):
    """
    Build an interactive Folium map overlaying erosion, accretion and
    instability layers on a satellite basemap (thesis § 5.11).

    ``bounds`` should be ``[[south, west], [north, east]]``.  When omitted the
    layers are added as image overlays over the given ``centre`` using
    normalised pixel coordinates.

    Returns the Folium ``Map`` object (and optionally saves it to HTML).
    """
    try:
        import folium
        from folium.raster_layers import ImageOverlay
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "folium is required for interactive risk maps. " "Install with `pip install folium`."
        ) from exc

    def _to_rgba(layer, cmap_name):
        norm = (layer - np.nanmin(layer)) / (np.nanmax(layer) - np.nanmin(layer) + 1e-9)
        cmap = plt.get_cmap(cmap_name)
        rgba = (cmap(norm) * 255).astype(np.uint8)
        rgba[:, :, 3] = (norm * 220).astype(np.uint8)  # alpha by intensity
        return rgba

    m = folium.Map(location=list(centre), zoom_start=zoom, tiles="Esri.WorldImagery")

    if bounds is None:
        s, w = centre[0] - 0.35, centre[1] - 0.35
        n, e = centre[0] + 0.35, centre[1] + 0.35
    else:
        (s, w), (n, e) = bounds

    ImageOverlay(
        _to_rgba(erosion, "Reds"), bounds=[[s, w], [n, e]], opacity=0.6, name="Erosion frequency"
    ).add_to(m)
    ImageOverlay(
        _to_rgba(accretion, "Blues"),
        bounds=[[s, w], [n, e]],
        opacity=0.6,
        name="Accretion frequency",
    ).add_to(m)
    ImageOverlay(
        _to_rgba(instability, "magma"),
        bounds=[[s, w], [n, e]],
        opacity=0.6,
        name="Instability index",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    if out_path:
        out_file = Path(out_path)
        _ensure_dir(out_file.parent)
        m.save(str(out_file))
        logger.info("saved risk map -> %s", out_file.name)
    return m
