# Preprocessing — Water Mask Cleaning & Gap Filling

This folder contains the preprocessing pipeline that transforms raw exported water-mask GeoTIFFs into the cleaned, gap-filled binary masks used for model training and evaluation.

> **Full technical details** are in the **pre-defence report**: [`docs/predefence_report.md`](docs/predefence_report.md) — see §4.3 *Data Preprocessing* (water detection, gap-filling process) and §5.1 *Sequence Generation Strategy*. Additional detail is in [`docs/methodology.md`](docs/methodology.md).

---

## Pipeline Overview

```
raw GeoTIFFs (60 m, EPSG:4326)
        │
        ▼
   1. Load & scan
        │
        ▼
   2. Water-mask cleaning
      • Keep top-3 largest connected components
      • Resize to 256×256
        │
        ▼
   3. Gap filling  (7 methods benchmarked)
      • Best: BiConvLSTM  (IoU = 0.7366)
        │
        ▼
   cleaned binary masks  (1 = water, 0 = land)
```

The main preprocessing notebook is:

| File | Purpose |
|------|---------|
| `preprocessing/01_gap_filling_methods.ipynb` | Full gap-filling pipeline for yearly water masks (1988–2025) — loads raw masks, applies connected-component cleaning, runs all 7 gap-filling methods, produces method comparison table |

---

## 1. Water-Mask Cleaning

After loading each raw GeoTIFF, two cleaning steps are applied before gap filling:

### 1.1 Top-3 Largest Connected Components

Water bodies in the study area form a single dominant river system; isolated noise pixels and small false-positive water patches from clouds, shadows, or classification errors appear as scattered small components. To remove these:

1. The binary mask is labelled using connected-component analysis (4-connectivity).
2. Components are sorted by pixel count (area) in descending order.
3. Only the **top 3 largest components** are retained; all smaller components are set to 0 (land).
4. The cleaned mask is resized to **256 × 256** pixels to standardise the spatial dimensions for model input.

This step is implemented in `utils/model_utils.py` via `keep_largest_n_components_cv2()` and is applied identically to all temporal resolutions.

### 1.2 Mask Convention

| Value | Meaning |
|-------|---------|
| `1` | Water |
| `0` | Land |
| `255` / `NaN` | No-data |

The same gap-filling logic is independently replicated and benchmarked in [`analysis/01_gap_filling_comparison.ipynb`](analysis/01_gap_filling_comparison.ipynb), which removes 20 % of years and evaluates all seven methods against ground truth to reproduce Table 4.2 of the thesis.

---

## 2. Gap-Filling Methods

Seven gap-filling methods are implemented and benchmarked. The thesis artificially removes **20 % of years** (7 randomly selected years: **2002, 2005, 2010, 2015, 2019, 2020, 2024**) from the 38-year yearly time series to simulate real-world missing-data scenarios, then evaluates each method on the held-out years.

| # | Method | Type | IoU | Dice | Precision | Recall |
|---|--------|------|-----|------|-----------|--------|
| 1 | **Mean Composite** | Simple temporal reduction | 0.6573 | 0.7926 | 0.8535 | 0.7403 |
| 2 | **Median Composite** | Simple temporal reduction | 0.6573 | 0.7926 | 0.8535 | 0.7403 |
| 3 | **Linear Interpolation** | Temporal interpolation | 0.6548 | 0.7910 | 0.8913 | 0.7193 |
| 4 | **Spline Interpolation** | Cubic spline (temporal) | 0.6733 | 0.8042 | 0.8082 | 0.8012 |
| 5 | **SDF (Signed Distance Field)** | Geometric + temporal interpolation | 0.7290 | 0.8429 | 0.8741 | 0.8161 |
| 6 | **Weighted Temporal (IDW)** | Inverse-distance weighted fusion | 0.6724 | 0.8035 | 0.8306 | 0.7790 |
| 7 | **BiConvLSTM** | Bidirectional ConvLSTM (deep learning) | **0.7366** | **0.8477** | 0.8457 | 0.8511 |

**BiConvLSTM** is the best-performing method and is used as the default gap-filling method in the production pipeline (configured in `config.yaml` under `preprocessing.gap_filling.method`).

### Method descriptions

1. **Mean / Median Composite** — fills a missing year using the pixel-wise mean or median of all adjacent available yearly masks. Robust but naive.

2. **Linear Interpolation** — for each pixel, linearly interpolates between the two nearest available years in the time series.

3. **Spline Interpolation** — cubic spline interpolation along the temporal axis for each pixel independently, producing smoother temporal transitions than linear interpolation.

4. **SDF Interpolation** — converts each binary water mask to a signed distance field (continuous), performs temporal interpolation in the continuous SDF space, then re-thresholds to recover a binary mask. Maintains water-body geometry and topology better than pixel-wise interpolation.

5. **Weighted Temporal (IDW)** — fills a missing year using inverse-distance-weighted averaging of all available years, where weights are proportional to `1 / |t − t_missing|`.

6. **BiConvLSTM** — a bidirectional Convolutional LSTM trained on 128×128 spatiotemporal patches with masked sequence learning (20 % missingness simulation). Uses binary cross-entropy loss. Captures complex spatiotemporal dynamics and achieves the highest IoU (0.7366).

---

## 3. Notebook: `01_gap_filling_methods.ipynb`

### What it does

1. **Loads** all yearly water-mask GeoTIFFs from the input directory (reads from the `YEARLY_DIR` environment variable, or falls back to `data/raw/yearly`).
2. **Cleans** each mask: keeps top-3 connected components, resizes to 256×256.
3. **Simulates missingness** by dropping the 7 designated years.
4. **Runs all 7 gap-filling methods** on the missing years.
5. **Evaluates** each method using IoU, Dice, Precision, Recall.
6. **Outputs** the gap-filled masks and a comparison table matching Table 4.2 of the thesis.

### Configuration

The notebook reads paths from environment variables (same convention as all other notebooks):

```bash
set YEARLY_DIR=path/to/Yearly_WaterMasks_1988_2025
```

If `YEARLY_DIR` is not set, it falls back to `data/raw/yearly`.

### Outputs

- Gap-filled yearly water masks (one GeoTIFF per filled year).
- A CSV / printed table comparing all 7 methods.
- Visualisation of original vs. gap-filled masks for selected years.

---

## 4. Configuration (`config.yaml`)

The preprocessing pipeline is configured in [`config/config.yaml`](config/config.yaml):

```yaml
preprocessing:
  gap_filling:
    enabled: true
    method: "biconvlstm"      # Options: mean, median, linear, spline, sdf, weighted, biconvlstm
    neighbour_count: 3

  cloud_masking:
    enabled: true
    landsat_qa_bits:
      cloud_shadow: 3
      cloud: 4
    sentinel2_qa_bits:
      opaque_clouds: 10
      cirrus_clouds: 11

  normalization:
    enabled: true
    method: "min_max"         # Options: min_max, z_score
    min_max_range: [0, 1]

  resize:
    target_size: [256, 256]   # Final mask dimensions
```

---

## 5. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `YEARLY_DIR` | `data/raw/yearly` | Directory containing yearly water-mask GeoTIFFs |
| `QUARTERLY_DIR` | `data/raw/quarterly` | Directory containing quarterly water-mask GeoTIFFs |
| `BIMONTHLY_DIR` | `data/raw/bimonthly` | Directory containing bi-monthly water-mask GeoTIFFs |

Set these before running any preprocessing or model notebook. See [`docs/repository_guide.md`](docs/repository_guide.md) and the root [`README.md`](../README.md) for details.

## 6. Sequence Generation (Post-Preprocessing)

After gap filling, the cleaned binary masks are converted into supervised learning sequences using a sliding-window approach (see pre-defence report §5.1):

| Resolution | Sequence lengths (L) | Stride |
|------------|---------------------|--------|
| Yearly | {4, 5, 6} | 1 year |
| Quarterly | {6, 8, 10} | 1 quarter |
| Bi-monthly | {6, 9, 12} | 1 bi-month |

For each sequence of L consecutive observations, the next time step is the prediction target. Overlapping windows (stride = 1) are used to maximise data usage.

---

## 7. References

- Pre-defence report: [`docs/predefence_report.md`](docs/predefence_report.md) — §4.3 *Data Preprocessing*, §5.1 *Sequence Generation Strategy*.
- Methodology: [`docs/methodology.md`](docs/methodology.md).
- Repository guide: [`docs/repository_guide.md`](docs/repository_guide.md).
- Root README: [`../README.md`](../README.md).
- Shared utilities: [`utils/model_utils.py`](../utils/model_utils.py) — `keep_largest_n_components_cv2()`, `create_sequences()`, `prepare_split()`.

