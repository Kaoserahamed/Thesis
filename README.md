# Analyzing & Forecasting River Morphological Evolution Using Machine Learning & Spatiotemporal Neural Models

## A Case Study on the Padma River

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()
[![Platform](https://img.shields.io/badge/platform-Google%20Earth%20Engine-green.svg)]()
[![License](https://img.shields.io/badge/license-Academic-lightgrey.svg)]()

**Shahjalal University of Science and Technology — Department of Computer Science and Engineering**

| Author | Registration No. |
|--------|------------------|
| Md. Kaoser Ahamed Anik | 2020331019 |
| S. S. Mahmud Turza | 2020331039 |

**Supervisor:** Md. Shadmim Hasan Sifat, Lecturer, Department of CSE, SUST

---

## 🌊 Overview

This repository contains the complete implementation of a framework that **forecasts river
morphological evolution** from freely available satellite imagery using spatiotemporal deep
learning. The study covers a **38-year time series (1987–2025)** of binary water masks for the
Padma River, Bangladesh, built from Landsat (5 TM, 7 ETM+, 8 OLI) and Sentinel (Sentinel-2 MSI,
Sentinel-1 SAR) imagery via **Google Earth Engine**.

Five spatiotemporal architectures are compared — ConvLSTM, U-Net+LSTM, Attention U-Net+ConvLSTM,
Swin Transformer and a ViT-based model — across **yearly, quarterly and bi-monthly** temporal
resolutions under two strict temporal evaluation setups.

## 🗺️ Interactive Prediction Map

An interactive **long-term prediction map** (2026–2040) is deployed via GitHub Pages:
➡️ **[https://kaoserahamed.github.io/Thesis](https://kaoserahamed.github.io/Thesis)**

The map visualizes predicted water-mask evolution across all three temporal resolutions
(yearly, quarterly, bi-monthly). Source: `docs/index.html` (auto-deployed from the `docs/`
folder on every push to `main`).

---

## 🖼️ Visual Summary

<div align="center">

**Study Area Selection**
<br>
<a href="https://github.com/user-attachments/assets/f5c83b4b-2fa8-4373-8283-36be2e14b36e">
  <img src="https://github.com/user-attachments/assets/f5c83b4b-2fa8-4373-8283-36be2e14b36e" alt="StudyArea" width="500" height="auto" />
</a>

**Water Mask**
<br>
<a href="https://github.com/user-attachments/assets/8f2f1069-cb65-4e2d-8f65-342b39f09940">
  <img src="https://github.com/user-attachments/assets/8f2f1069-cb65-4e2d-8f65-342b39f09940" alt="Water Mask" width="500" height="auto" />
</a>

**Analysis** (seasonal profiles & cross-validation)
<br>
<a href="https://github.com/user-attachments/assets/952322a9-c775-4bca-9bf6-b9d4d51b2c11">
  <img src="https://github.com/user-attachments/assets/952322a9-c775-4bca-9bf6-b9d4d51b2c11" alt="fig2_seasonal_profiles_cv" width="500" height="auto" />
</a>

**Prediction** (long-term morphological forecast 2026–2040)
<br>
<a href="https://github.com/user-attachments/assets/da72745f-53894e56-9561-b1e26dc73a3f">
  <img src="https://github.com/user-attachments/assets/da72745f-53894e56-9561-b1e26dc73a3f" alt="Prediction" width="500" height="auto" />
</a>

</div>

---

## 🎯 Key Results

### Gap-filling (reconstructing missing annual masks)

| Method | IoU | Dice | Precision | Recall |
|--------|-----|------|-----------|--------|
| Mean Composite | 0.6573 | 0.7926 | 0.8535 | 0.7403 |
| Median Composite | 0.6573 | 0.7926 | 0.8535 | 0.7403 |
| Linear Interpolation | 0.6548 | 0.7910 | 0.8913 | 0.7193 |
| Spline Interpolation | 0.6733 | 0.8042 | 0.8082 | 0.8012 |
| SDF | 0.7290 | 0.8429 | 0.8741 | 0.8161 |
| Weighted Temporal | 0.6724 | 0.8035 | 0.8306 | 0.7790 |
| **BiConvLSTM** | **0.7366** | **0.8477** | 0.8457 | 0.8511 |

### Best architecture per temporal resolution

| Resolution | Best model | Setup | IoU | Dice |
|------------|-----------|-------|-----|------|
| Yearly | **Attention U-Net+ConvLSTM** | Setup 1 (10-yr) | **0.7005** | **0.8236** |
| Quarterly | **Attention U-Net+ConvLSTM** | Setup 1 (long-range) | **0.6956** | 0.8139 |
| Bi-monthly | **U-Net+LSTM** | Setup 2 (medium-range) | **0.7791** | **0.8701** |

### Key findings

- **Hybrid CNN-LSTM models consistently outperform pure transformers** for river morphology
  prediction; ViT/ViViT reached only IoU ≈ 0.34–0.38.
- Mean annual centreline migration is **255.4 m yr⁻¹**, peaking at **1,485.5 m yr⁻¹** during the
  catastrophic 1998–1999 flood.
- The 2026–2040 forecast projects an expansion of **≈ 290 km² (+56 %)** over the 2025 baseline,
  dominated by accretion-rather-than-erosion transitions.
- A spatially explicit **erosion / accretion / instability risk map** was generated for disaster
  preparedness.

**Live demo:** <https://kaoserahamed.github.io/Thesis/>

---

## 📁 Repository Structure

```
ThesisFinal/
│
├── config/
│   └── config.yaml                  # Study area, data, model & metric configuration
│
├── docs/
│   ├── predefence_report.md         # Full thesis report (Markdown, converted from PDF)
│   ├── methodology.md               # Methodology summary
│   ├── repository_guide.md          # How every folder/notebook fits together
│   └── Thesis_Paper.pdf             # Complete thesis document
│
├── utils/                           # Shared Python libraries (single source of truth)
│   ├── __init__.py
│   ├── model_utils.py               # Architectures, losses, metrics, splits, callbacks
│   ├── data_utils.py                # GeoTIFF I/O and preprocessing helpers
│   └── visualization_utils.py       # Plots and interactive Folium risk maps
│
├── scripts/
│   ├── generate_notebooks.py        # Regenerates all self-contained notebooks (inlines model_utils.py)
│   ├── validate_notebooks.py        # Asserts every notebook is valid, parseable & self-contained
│   ├── verify_dist.py               # Asserts the built wheel contains + compiles every module
│   └── ci_local.py                  # Runs the CI quality gates locally (cross-platform)
│
├── data_collection/                 # Google Earth Engine export notebooks
│   ├── 01_yearly_data_collection.ipynb
│   ├── 02_bimonthly_data_collection.ipynb
│   └── 03_quarterly_data_collection.ipynb
│
├── preprocessing/
│   └── 01_gap_filling_methods.ipynb
│
├── models/                          # Architecture experiments (self-contained)
│   ├── 00_yearly_model_comparison.ipynb      # all 5 architectures, one notebook
│   ├── 00_quarterly_model_comparison.ipynb
│   ├── 00_bimonthly_model_comparison.ipynb
│   ├── yearly/                      # 5 architectures @ yearly resolution
│   │   ├── 01_yearly_convlstm.ipynb
│   │   ├── 02_yearly_unet_lstm.ipynb
│   │   ├── 03_yearly_attention_unet_convlstm.ipynb   # best yearly
│   │   ├── 04_yearly_swin_transformer.ipynb
│   │   └── 05_yearly_vit.ipynb
│   ├── quarterly/                   # 5 architectures @ quarterly resolution
│   │   ├── 01_quarterly_convlstm.ipynb
│   │   ├── 02_quarterly_unet_lstm.ipynb
│   │   ├── 03_quarterly_attention_unet_convlstm.ipynb # best quarterly
│   │   ├── 04_quarterly_swin_transformer.ipynb
│   │   └── 05_quarterly_vit.ipynb
│   └── bimonthly/                   # 5 architectures @ bi-monthly resolution
│       ├── 01_bimonthly_convlstm.ipynb
│       ├── 02_bimonthly_unet_lstm.ipynb               # best bi-monthly
│       ├── 03_bimonthly_attention_unet_convlstm.ipynb
│       ├── 04_bimonthly_swin_transformer.ipynb
│       └── 05_bimonthly_vit.ipynb
│
│   # Numbering is identical in all three folders:
│   # 01 ConvLSTM | 02 U-Net+LSTM | 03 Attention U-Net+ConvLSTM | 04 Swin ST | 05 ViT
│
├── analysis/                        # Cross-cutting analysis notebooks
│   ├── 01_gap_filling_comparison.ipynb
│   ├── 02_long_term_prediction.ipynb
│   ├── 03_short_term_prediction.ipynb
│   └── 04_quarterly_prediction.ipynb
│
├── results/
│   └── 01_statistical_analysis.ipynb
│
├── tests/                           # pytest suite (91 tests) for the shared library
│   ├── conftest.py                  # Shared fixtures (masks, synthetic sequences)
│   ├── test_data_utils.py           # GeoTIFF I/O, normalisation, statistics, masks
│   ├── test_data_pipeline.py        # create_sequences, prepare_split, presets, seeding
│   ├── test_metrics.py              # IoU, Dice, area difference, thresholds
│   ├── test_model_builders.py       # all five architectures build & forward-pass
│   └── test_visualization.py        # change-frequency maps & plotting helpers
│
├── .github/workflows/ci.yml         # CI: black, flake8, mypy, notebook validation, tests
│
├── visualizations/                  # Generated interactive maps (HTML)
│
├── data/                            # Data storage (not tracked in git)
│   ├── raw/{yearly,quarterly,bimonthly}/
│   └── predictions/
│
├── archive/                         # Original flattened notebooks & reports (reference)
│
├── requirements.txt                 # Runtime dependencies
├── requirements-dev.txt             # Test/lint dependencies (used by CI)
├── requirements-collection.txt      # Google Earth Engine (data collection only)
├── pyproject.toml                   # Packaging, black, pytest & coverage config
├── setup.cfg                        # flake8 + mypy config
└── README.md
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Authenticate Google Earth Engine (data collection only)

```bash
earthengine authenticate
```

### 3. Regenerate the notebooks (optional)

```bash
python scripts/generate_notebooks.py     # (re)write the self-contained notebooks
python scripts/validate_notebooks.py     # sanity-check them (valid, parseable, self-contained)
```

### 4. Run the test suite (optional)

```bash
pip install -r requirements-dev.txt
pytest -m "not slow"                     # fast lane, no TensorFlow
pytest -m "requires_tensorflow"          # builds all five architectures
```

### 5. Point the notebooks at your data

Every model/analysis notebook resolves the dataset paths from environment variables
(falling back to `data/raw/<resolution>/`):

```bash
set YEARLY_DIR=path/to/yearly_masks
set QUARTERLY_DIR=path/to/quarterly_masks
set BIMONTHLY_DIR=path/to/bimonthly_masks
```

---

## 🔬 Methodology Summary

### Data pipeline

1. **Collection** — yearly / quarterly / bi-monthly composites exported from GEE as 60 m GeoTIFFs
   in EPSG:4326.
2. **Water detection** — `MNDWI = (Green − SWIR1) / (Green + SWIR1) > 0`; Sentinel-1 VV < −16 dB
   as cloud-independent fallback.
3. **Cloud masking** — Landsat `QA_PIXEL` bits 3–4; Sentinel-2 `QA60` bits 10–11.
4. **Gap filling** — 7 methods; BiConvLSTM is best (IoU = 0.7366).
5. **Connected-component cleaning** — keep the 3 largest water components, resize to 256×256.

### Models

The architectures are implemented in `utils/model_utils.py` and **inlined** into each self-contained
notebook by `scripts/generate_notebooks.py`, so every model notebook runs on its own.

| Registry key | Architecture |
|--------------|--------------|
| `convlstm` | Standalone ConvLSTM |
| `unet_lstm` | U-Net with ConvLSTM bottleneck |
| `attention_unet_convlstm` | Attention-gated U-Net with ConvLSTM |
| `swin_st` | Swin spatio-temporal transformer |
| `vit_st` | ViT / ViViT spatio-temporal model |

- **Loss:** `L_Total = L_BCE + L_Dice`
- **Optimiser:** Adam, lr = 1e-4, clipnorm = 1.0
- **Sequence lengths:** yearly {4,5,6}, quarterly {6,8,10}, bi-monthly {6,9,12}
- **Strict temporal split:** Setup 1 (train ≤ 2015 / test 2016–2025), Setup 2 (train ≤ 2020 /
  test 2021–2025)
- **Callbacks:** ModelCheckpoint, EarlyStopping (patience 20), ReduceLROnPlateau (patience 7)

### Evaluation metrics

IoU, Dice, Precision, Recall and the signed area difference `ΔA = A_pred − A_true` (km²).

---

## 📖 Usage

```bash
# Data collection
jupyter notebook data_collection/01_yearly_data_collection.ipynb

# Gap-filling comparison
jupyter notebook analysis/01_gap_filling_comparison.ipynb

# Train one architecture (e.g. yearly Attention U-Net+ConvLSTM = 03)
jupyter notebook models/yearly/03_yearly_attention_unet_convlstm.ipynb

# Train another architecture at another resolution (e.g. bi-monthly ViT = 05)
jupyter notebook models/bimonthly/05_bimonthly_vit.ipynb

# Full yearly architecture comparison
jupyter notebook models/00_yearly_model_comparison.ipynb

# Long-term forecast + risk map (2026–2040)
jupyter notebook analysis/02_long_term_prediction.ipynb

# Statistical analysis of river dynamics
jupyter notebook results/01_statistical_analysis.ipynb
```

---

## 🧩 Design Principles

- **Single source of truth** — every architecture, loss, metric and split lives in
  `utils/model_utils.py`, so all notebooks and all three temporal resolutions are provably
  consistent.
- **No data leakage** — `prepare_split()` asserts that no test year appears in training.
- **Reproducibility** — `seed_everything(42)` is called before every model build; identical
  splits are reused across architectures.
- **Self-contained notebooks** — each model and analysis notebook inlines the full pipeline and can
  be run on its own; `scripts/generate_notebooks.py` regenerates them from `utils/model_utils.py`.

---

## 🧪 Testing & CI

The shared library is covered by 91 pytest tests, and every push runs
**black + flake8 + mypy + notebook validation + package build + tests** on GitHub Actions
(`.github/workflows/ci.yml`).

```bash
pip install -r requirements-dev.txt

pytest -m "not slow"                     # fast unit lane (no TensorFlow)
pytest -m "requires_tensorflow"          # builds all five architectures
pytest --cov=utils --cov-report=term-missing

python scripts/ci_local.py               # run the whole gate set locally
python scripts/ci_local.py --build --all # ... including build + model lane
```

Individual gates (identical to the CI `lint` job):

```bash
black --check --line-length 100 utils tests scripts
flake8 utils tests scripts
mypy utils tests scripts
python scripts/validate_notebooks.py     # notebooks: valid, parseable, self-contained
python -m build && python scripts/verify_dist.py dist   # package builds & wheel is intact
```

`scripts/validate_notebooks.py` guarantees every generated notebook is valid nbformat JSON, has
parseable code cells, does not import `model_utils` (it is self-contained), and defines every
helper it calls.

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`docs/predefence_report.md`](docs/predefence_report.md) | Full thesis report (Markdown) |
| [`docs/methodology.md`](docs/methodology.md) | Methodology summary |
| [`docs/repository_guide.md`](docs/repository_guide.md) | How every folder/notebook fits together |
| `docs/Thesis_Paper.pdf` | Complete thesis document |

---

## 📄 Citation

```bibtex
@thesis{anik_turza_2026_river,
  title  = {Analyzing & Forecasting River Morphological Evolution Using Machine Learning
            & Spatiotemporal Neural Models: A Case Study on the Padma River},
  author = {Md. Kaoser Ahamed Anik and S. S. Mahmud Turza},
  school = {Shahjalal University of Science and Technology},
  year   = {2026}
}
```

---

## 🙏 Acknowledgments

- **Google Earth Engine** for planetary-scale geospatial processing.
- **USGS** for Landsat imagery and **ESA** for Sentinel imagery.
- Department of CSE, SUST for facilities and support.

---

**Last Updated:** September 22, 2026