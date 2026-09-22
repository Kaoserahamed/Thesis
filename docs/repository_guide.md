# Repository Guide

This document explains how the repository is organised, what each notebook does, and the order in
which to run them. It is the map that connects the **thesis report** (`docs/predefence_report.md`)
to the **code**.

---

## 1. The three layers of the repository

```
┌──────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — NOTEBOOKS (self-contained, one per experiment)                 │
│  data_collection/  preprocessing/  models/  analysis/  results/           │
│  each model / analysis notebook inlines the full pipeline                 │
└───────────────▲──────────────────────────────────────────────────────────┘
                │ source of truth for the inlined code
┌───────────────┴──────────────────────────────────────────────────────────┐
│  LAYER 2 — SHARED LIBRARIES (reviewed implementations)                    │
│  utils/model_utils.py   utils/visualization_utils.py   utils/data_utils.py│
└───────────────▲──────────────────────────────────────────────────────────┘
                │ extracted via ast
┌───────────────┴──────────────────────────────────────────────────────────┐
│  LAYER 1 — GENERATOR                                                      │
│  scripts/generate_notebooks.py  →  writes every self-contained notebook   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Why this matters:** the thesis trains the *same* five architectures at three temporal resolutions
under *identical* preprocessing and splitting. `utils/model_utils.py` holds the reviewed
implementations; `scripts/generate_notebooks.py` inlines them into **self-contained** notebooks, so
the comparison cannot silently drift and each notebook can be read and run on its own.

---

## 2. Run order (end-to-end reproduction)

| Step | Notebook | Produces |
|------|----------|----------|
| 1 | `data_collection/01_yearly_data_collection.ipynb` | Yearly water masks 1988–2025 (GeoTIFF) |
| 2 | `data_collection/02_bimonthly_data_collection.ipynb` | Bi-monthly composites (BM1–BM6) |
| 3 | `data_collection/03_quarterly_data_collection.ipynb` | Quarterly composites (Q1–Q4) |
| 4 | `preprocessing/01_gap_filling_methods.ipynb` *or* `analysis/01_gap_filling_comparison.ipynb` | Gap-filled masks + method comparison (Table 4.2) |
| 5 | `models/00_yearly_model_comparison.ipynb` | Best yearly model (Table 7.x) |
| 6 | `models/00_quarterly_model_comparison.ipynb` | Best quarterly model |
| 7 | `models/00_bimonthly_model_comparison.ipynb` | Best bi-monthly model |
| 8 | `analysis/02_long_term_prediction.ipynb` | Yearly 2026–2040 forecast + risk map (Figure 7.1) |
| 9 | `analysis/04_quarterly_prediction.ipynb` | Quarterly 2026 Q1–2029 Q4 forecast |
| 10 | `analysis/03_short_term_prediction.ipynb` | Bi-monthly forecast 2026 P1–2028 P6 (§ 7.5) |
| 11 | `results/01_statistical_analysis.ipynb` | Morphological statistics (Chapter 6) |

---

## 3. Notebook catalogue

### `data_collection/`
Google Earth Engine export scripts. Each notebook filters scenes, masks clouds, computes MNDWI
(and Sentinel-1 VV fallback), builds a median composite, and exports a 60 m GeoTIFF in EPSG:4326.

| File | Purpose |
|------|---------|
| `01_yearly_data_collection.ipynb` | Annual composites, Landsat 1988–2014 + Sentinel 2015–2025 |
| `02_bimonthly_data_collection.ipynb` | Six two-month composites per year |
| `03_quarterly_data_collection.ipynb` | Four quarterly composites per year |

### `preprocessing/` and `analysis/01_...`
Gap-filling. The thesis defines seven methods; `analysis/01_gap_filling_comparison.ipynb` removes
20 % of the years and benchmarks all seven, reproducing Table 4.2.

### `models/`
Every architecture × temporal resolution. The `00_*` notebooks run the **full comparison** (all five
architectures); the numbered notebooks train a **single** architecture. Model notebooks live in the
`yearly/`, `quarterly/` and `bimonthly/` sub-folders.

| # | File | Architecture | Resolution |
|---|------|--------------|------------|
| — | `00_yearly_model_comparison.ipynb` | all five | yearly |
| — | `00_quarterly_model_comparison.ipynb` | all five | quarterly |
| — | `00_bimonthly_model_comparison.ipynb` | all five | bi-monthly |
| 1 | `yearly/01_yearly_convlstm.ipynb` | ConvLSTM | yearly |
| 2 | `yearly/02_yearly_unet_lstm.ipynb` | U-Net+LSTM | yearly |
| 3 | `yearly/03_yearly_attention_unet_convlstm.ipynb` | Attention U-Net+ConvLSTM | yearly |
| 4 | `yearly/04_yearly_swin_transformer.ipynb` | Swin | yearly |
| 5 | `yearly/05_yearly_vit.ipynb` | ViT | yearly |
| 1 | `quarterly/01_quarterly_convlstm.ipynb` | ConvLSTM | quarterly |
| 2 | `quarterly/02_quarterly_unet_lstm.ipynb` | U-Net+LSTM | quarterly |
| 3 | `quarterly/03_quarterly_attention_unet_convlstm.ipynb` | Attention U-Net+ConvLSTM | quarterly |
| 4 | `quarterly/04_quarterly_swin_transformer.ipynb` | Swin | quarterly |
| 5 | `quarterly/05_quarterly_vit.ipynb` | ViViT | quarterly |
| 1 | `bimonthly/01_bimonthly_convlstm.ipynb` | ConvLSTM | bi-monthly |
| 2 | `bimonthly/02_bimonthly_unet_lstm.ipynb` | U-Net+LSTM | bi-monthly |
| 3 | `bimonthly/03_bimonthly_attention_unet_convlstm.ipynb` | Attention U-Net+ConvLSTM | bi-monthly |
| 4 | `bimonthly/04_bimonthly_swin_transformer.ipynb` | Swin | bi-monthly |
| 5 | `bimonthly/05_bimonthly_vit.ipynb` | ViT | bi-monthly |

The numbering is identical for every resolution — **`01` ConvLSTM, `02` U-Net+LSTM,
`03` Attention U-Net+ConvLSTM, `04` Swin ST, `05` ViT** — so the three folders line up
one-to-one and any architecture can be compared across resolutions by its number.

All of these notebooks are **self-contained**: they inline the complete pipeline (preprocessing,
losses, metrics, sequences, split, architecture, training, evaluation) and run without importing any
project module. They are generated by `scripts/generate_notebooks.py`, which extracts the reviewed
code from `utils/model_utils.py`.

### `analysis/`
Cross-cutting experiments that use the *best* model per resolution.

| File | Purpose |
|------|---------|
| `01_gap_filling_comparison.ipynb` | Seven-method gap-filling benchmark (Table 4.2) |
| `02_long_term_prediction.ipynb` | 2026–2040 autoregressive forecast + risk maps (Fig. 7.1) |
| `03_short_term_prediction.ipynb` | 18-period bi-monthly forecast with seasonal channels (§ 7.5) |
| `04_quarterly_prediction.ipynb` | 16-period quarterly forecast (2026 Q1–2029 Q4) |

### `results/`
| File | Purpose |
|------|---------|
| `01_statistical_analysis.ipynb` | Areal dynamics, erosion/accretion, channel migration, WOF, Markov |

---

## 4. The shared library (`utils/`)

### `model_utils.py`
Everything model-related:

- **Losses/metrics** — `dice_coefficient`, `dice_loss`, `bce_loss`, `combined_loss`, `iou_metric`,
  plus NumPy variants `iou_np`, `dice_np`, `calculate_area_difference`.
- **Data pipeline** — `load_and_preprocess_image`, `keep_largest_n_components_cv2`,
  `build_catalog`, `load_image_stack`, `get_pixel_area_km2`.
- **Sequence/temporal** — `create_sequences`, `prepare_split` (leakage-proof split).
- **Architectures** — `build_convlstm`, `build_unet_lstm`, `build_attention_unet_convlstm`,
  `build_swin_st`, `build_vit_st`, and `build_model(name, seq_len)` via `MODEL_REGISTRY`.
- **Training** — `create_callbacks`, `StructuredTrainingLogger`, `seed_everything`,
  `configure_gpu`.
- **Evaluation** — `evaluate_model`, `summarise_results`.
- **Config** — `ExperimentConfig`, `EXPERIMENT_PRESETS`.

### `visualization_utils.py`
`plot_training_curves`, `plot_sequence_comparison`, `plot_predictions`, `plot_area_trend`,
`compute_change_frequencies`, `plot_risk_maps`, `create_folium_risk_map`, `save_figure`.

### `data_utils.py`
GeoTIFF I/O (`load_geotiff`, `save_geotiff`), temporal loading, normalisation, water-mask helpers.

---

## 5. Configuration & conventions

### Paths
Notebooks read the dataset directories from environment variables (falling back to
`data/raw/<resolution>/`):

| Variable | Default |
|----------|---------|
| `YEARLY_DIR` | `data/raw/yearly` |
| `QUARTERLY_DIR` | `data/raw/quarterly` |
| `BIMONTHLY_DIR` | `data/raw/bimonthly` |

Outputs are written to `outputs/<resolution>/checkpoints/` (best model weights, `.keras`) and
`outputs/<resolution>/forecast/` (forecast CSVs and figures from the prediction notebooks).

### Mask convention
`1 = water`, `0 = land`, `255`/`NaN` = no-data.

### Temporal split
| Setup | Trains/validates | Tests |
|-------|------------------|-------|
| Setup 1 | targets ≤ 2015 | 2016–2025 |
| Setup 2 | targets ≤ 2020 | 2021–2025 |

`prepare_split()` asserts there is **no** year overlap between train/val and test.

### Sequence lengths
- Yearly: `{4, 5, 6}`
- Quarterly: `{6, 8, 10}`
- Bi-monthly: `{6, 9, 12}`

---

## 6. Regenerating notebooks

If you modify `scripts/generate_notebooks.py` (templates) or the shared library, regenerate the
self-contained notebooks:

```bash
python scripts/generate_notebooks.py
```

It (re)writes every self-contained notebook: `models/00_*_model_comparison.ipynb`,
`models/{yearly,quarterly,bimonthly}/*.ipynb` (five architectures per resolution) and the best-model
prediction notebooks `analysis/02`–`04`. The `data_collection/`, `preprocessing/`,
`analysis/01_gap_filling_comparison.ipynb` and `results/` notebooks are **not** overwritten.

---

## 7. Testing & CI

### Test suite

`tests/` holds 91 pytest tests covering the shared library:

| File | Covers |
| --- | --- |
| `test_data_utils.py` | GeoTIFF I/O round-trip, normalisation, statistics, water masks, year extraction |
| `test_visualization.py` | `compute_change_frequencies` (erosion/accretion/instability) and the plotting helpers (headless `Agg`) |
| `test_metrics.py` | IoU, Dice, area difference `ΔA`, threshold behaviour |
| `test_data_pipeline.py` | `create_sequences`, the leakage-proof `prepare_split`, `EXPERIMENT_PRESETS`, `seed_everything` |
| `test_model_builders.py` | all five architectures build, compile, and run a forward pass |

Two pytest markers keep the suite fast: `slow` (model construction) and `requires_tensorflow`
(needs a working TF runtime). Run them locally with:

```bash
# fast unit lane (no TensorFlow, seconds)
pytest -m "not slow"

# model lane (builds all five architectures)
pytest -m "requires_tensorflow"

# everything, with coverage
pytest --cov=utils --cov-report=term-missing
```

### Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request to `main` and has four jobs:

1. **`lint`** — `black --check` (line length 100), `flake8`, `mypy`, and
   `python scripts/validate_notebooks.py`, which asserts every generated notebook is valid
   nbformat JSON, has fully parseable code cells, is self-contained (no `model_utils` imports and
   no leftover `%%` template tokens), and that every helper it calls is defined in the same file.
2. **`test`** — installs dependencies, runs `pip check` to validate dependency consistency, then
   runs a two-entry matrix (`fast` → `-m "not slow"`, `models` → `-m "requires_tensorflow"`) with
   coverage reported to Codecov and JUnit XML uploaded as an artifact.
3. **`build`** — `python -m build` produces the sdist and wheel,
   `scripts/verify_dist.py` asserts every shared-library module is present and byte-compiles, and
   the wheel is installed with `--no-deps` to confirm the metadata is valid.
4. **`audit`** — `pip-audit -r requirements.txt` for known-vulnerable dependencies (advisory: the
   step does not fail the build, since third-party transitive pins are outside this project's
   control).

Reproduce the lint job locally before pushing:

```bash
black --check --line-length 100 utils tests scripts
flake8 utils tests scripts
mypy utils tests scripts
python scripts/validate_notebooks.py
```

Or run the whole pipeline at once with the cross-platform helper:

```bash
python scripts/ci_local.py            # lint + notebooks + fast tests
python scripts/ci_local.py --build    # also build & verify the wheel
python scripts/ci_local.py --all      # also run the TensorFlow model lane
```

---

## 8. Archive

`archive/` holds the original flattened uploads:

- `archive/notebooks_root/` — the original top-level notebooks before reorganisation.
- `archive/legacy_notebooks/` — the original `05_yearly_prediction.ipynb` and
  `06_short_term_prediction.ipynb` (superseded by `analysis/02` and `analysis/03`).
- `archive/reports_root/` — the original pre-defence report PDF (`Predefence_report (1).pdf`);
  the final `Thesis_Paper.pdf` is kept once, in [`docs/`](../docs/).
- `archive/bimonthly_river_prediction_map (2).html` — an original interactive map.

These are kept for provenance and are not part of the runnable pipeline.

---

*Last Updated: September 22, 2026*