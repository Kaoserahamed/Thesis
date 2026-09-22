"""
Data-pipeline and configuration utilities for river morphology prediction.

Every function and dataclass in this module is pure NumPy / pandas / GeoTIFF
I/O -- **no TensorFlow dependency** -- so they can be imported and unit-tested
in the fast CI lane without a deep-learning runtime.

Contents
--------
* Shared training constants (image size, learning rate, etc.)
* Geospatial helpers that used to be trapped inside ``model_utils``
  (``keep_largest_n_components_cv2``, ``load_and_preprocess_image``,
  ``build_catalog``, ``load_image_stack`` ...)
* Sequence generation (``create_sequences``) and the strict temporal split
  (``prepare_split``)
* Experiment configuration (``ExperimentConfig`` / ``EXPERIMENT_PRESETS``)

The generic GeoTIFF I/O helpers (``load_geotiff``, ``save_geotiff``,
``normalize_array`` ...) live in :mod:`utils.data_utils` and are re-exported
here, so this module is a single import surface for the TensorFlow-free data
pipeline without duplicating any implementation.

Reference
---------
"Analyzing & Forecasting River Morphological Evolution Using Machine
Learning & Spatiotemporal Neural Models: A Case Study on the Padma River"
Chapter 5 - Methodology.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
from skimage.transform import resize

from .data_utils import (  # noqa: F401
    calculate_water_area,
    compute_statistics,
    create_water_mask,
    extract_year_from_filename,
    get_file_list,
    load_config,
    load_geotiff,
    load_temporal_sequence,
    normalize_array,
    save_geotiff,
    validate_path,
    validate_safe_pattern,
)
from .metrics_numpy import BINARY_THRESHOLD  # noqa: F401

IMG_HEIGHT: int = 256
IMG_WIDTH: int = 256
N_COMPONENTS: int = 3  # keep the 3 largest connected water components
LEARNING_RATE: float = 1e-4
CLIPNORM: float = 1.0
DEFAULT_EPOCHS: int = 200
EARLY_STOP_PATIENCE: int = 20
REDUCE_LR_PATIENCE: int = 7

__all__ = [
    "BINARY_THRESHOLD",
    "IMG_HEIGHT",
    "IMG_WIDTH",
    "N_COMPONENTS",
    "LEARNING_RATE",
    "CLIPNORM",
    "DEFAULT_EPOCHS",
    "EARLY_STOP_PATIENCE",
    "REDUCE_LR_PATIENCE",
    "ExperimentConfig",
    "EXPERIMENT_PRESETS",
    "build_catalog",
    "create_sequences",
    "get_pixel_area_km2",
    "keep_largest_n_components_cv2",
    "load_and_preprocess_image",
    "load_image_stack",
    "prepare_split",
    "seed_numpy",
    "seed_numpy",
    "calculate_water_area",
    "compute_statistics",
    "create_water_mask",
    "extract_year_from_filename",
    "get_file_list",
    "load_config",
    "load_geotiff",
    "load_temporal_sequence",
    "normalize_array",
    "save_geotiff",
]


def keep_largest_n_components_cv2(mask: np.ndarray, n: int = N_COMPONENTS):
    """Retain only the *n* largest connected components of a binary mask.

    This removes salt-and-pepper noise and isolated ponds so the network
    focuses on the main river channel (thesis section 5.9).
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
    """Load a GeoTIFF water mask and turn it into a 256x256 clean binary array.

    Steps: read -> optional connected-component cleaning -> resize (nearest)
    -> binarise at :data:`BINARY_THRESHOLD`.
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
    return (resized > BINARY_THRESHOLD).astype(np.float32)


def get_pixel_area_km2(reference_tif: str) -> float:
    """Compute the ground area (km2) covered by a single 256x256 pixel.

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
    """Scan *data_dir* for GeoTIFFs and return a year-sorted DataFrame.

    Columns: ``filepath``, ``filename``, ``year``.  Files whose name contains
    no 4-digit year are skipped.

    The column schema is always returned -- even when the directory is empty or
    contains nothing usable -- so callers can rely on ``df["year"]`` without a
    guard clause.
    """
    data_dir = validate_path(data_dir, "data_dir")
    pattern = validate_safe_pattern(pattern)

    files = sorted(glob.glob(os.path.join(str(data_dir), pattern)))
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

    if not records:
        return pd.DataFrame(columns=["filepath", "filename", "year"])

    return pd.DataFrame(records).sort_values("year").reset_index(drop=True)


def load_image_stack(data_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """Load and preprocess every GeoTIFF in *data_dir* into a stack.

    Raises
    ------
    FileNotFoundError
        If *data_dir* does not exist, or exists but holds no GeoTIFF whose
        filename contains a 4-digit year.  Failing loudly here is deliberate:
        a silently empty stack only surfaces much later as a confusing error
        inside the sequence/split code (or, worse, as a model trained on
        nothing).
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory does not exist: {data_dir!r}")

    df = build_catalog(data_dir)
    if df.empty:
        raise FileNotFoundError(
            f"No year-tagged GeoTIFFs found in {data_dir!r}. Expected filenames "
            f"containing a 4-digit year, e.g. 'Padma_2015.tif'."
        )

    images, years = [], []
    for _, row in df.iterrows():
        images.append(load_and_preprocess_image(row["filepath"]))
        years.append(int(row["year"]))
    return images, years


def create_sequences(
    images: Sequence[np.ndarray],
    years: Sequence[int],
    seq_len: int,
    horizon: int = 1,
    stride: int = 1,
):
    """Build overlapping ``(X, y)`` windows with a stride of 1.

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
    if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
        raise ValueError("seq_len must be a positive integer")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("stride must be a positive integer")
    if len(images) != len(years):
        raise ValueError("images and years must have the same length")

    X, y, in_years, tgt_years = [], [], [], []
    stop = len(images) - seq_len - horizon + 1
    for i in range(0, stop, stride):
        X.append(images[i : i + seq_len])
        y.append(images[i + seq_len + horizon - 1])
        in_years.append(years[i : i + seq_len])
        tgt_years.append(years[i + seq_len + horizon - 1])
    return (np.array(X), np.array(y), np.array(in_years), np.array(tgt_years))


def prepare_split(X_all, y_all, target_years_all, input_years_all, cutoff_year: int):
    """Leakage-proof temporal split.

    Training/validation use samples whose *target* and *latest input* years
    are <= ``cutoff_year``; testing uses targets strictly after the cutoff.
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
            f"Empty split! Train={len(X_train)}, Test={len(X_test)} (cutoff={cutoff_year})"
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


def seed_numpy(seed: int = 42) -> None:
    """Seed the NumPy global RNG used by the data pipeline.

    Split out from :func:`utils.model_utils.seed_everything` so that the
    NumPy half of the reproducibility guarantee can be unit-tested without a
    deep-learning runtime.
    """
    np.random.seed(seed)


@dataclass
class ExperimentConfig:
    """Encapsulates one experimental setup (resolution + temporal cutoff)."""

    resolution: str  # "yearly" | "quarterly" | "bimonthly"
    sequence_lengths: List[int] = field(default_factory=lambda: [4, 5, 6])
    cutoff_year: int = 2015
    test_label: str = "Test: 2016-2025"
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = 4


EXPERIMENT_PRESETS: Dict[str, ExperimentConfig] = {
    "yearly_setup1": ExperimentConfig("yearly", [4, 5, 6], 2015, "2016-2025"),
    "yearly_setup2": ExperimentConfig("yearly", [4, 5, 6], 2020, "2021-2025"),
    "quarterly_setup1": ExperimentConfig("quarterly", [6, 8, 10], 2015, "2015-Q1 onwards"),
    "quarterly_setup2": ExperimentConfig("quarterly", [6, 8, 10], 2020, "2020-Q1 onwards"),
    "bimonthly_setup1": ExperimentConfig("bimonthly", [6, 9, 12], 2015, "long-range"),
    "bimonthly_setup2": ExperimentConfig("bimonthly", [6, 9, 12], 2020, "medium-range"),
}
