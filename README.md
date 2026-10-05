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

## 🔬 Ablation Study

To validate the design choices behind the best-performing model (Attention U-Net + ConvLSTM), we
conducted a systematic ablation study varying one factor at a time:

| Factor | Variants Tested | Finding |
|--------|-----------------|---------|
| **Loss Function** | Combined (BCE + Dice), Dice-only, BCE-only | Combined loss provides best balance between pixel-level accuracy and region-level overlap |
| **Sequence Length** | L ∈ {3, 5, 7, 10} | Performance peaks at L=5–7; longer sequences may introduce noise |
| **Dropout Regularization** | With (0.2) vs. without | Dropout prevents overfitting given limited training data (<40 observations) |
| **Learning Rate** | 1e-3, 1e-4, 1e-5 | 1e-4 provides stable convergence; 1e-3 causes instability, 1e-5 too slow |
| **Attention Mechanism** | With vs. without attention gates | Attention gates provide measurable IoU/Dice improvements |

**Key findings:** The baseline configuration (combined loss, L=5, dropout=0.2, LR=1e-4, with
attention) represents a well-balanced choice across all tested dimensions. Removing any component
degrades performance, confirming each design decision contributes meaningfully to the final model.

All ablation experiments are tracked in MLflow with reproducible seeding and visualized in
`analysis/05_ablation_study.ipynb`. Results are saved to `outputs/ablation/ablation_results.csv`
for further analysis.

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
│   ├── 04_quarterly_prediction.ipynb
│   └── 05_ablation_study.ipynb      # Systematic model configuration sensitivity analysis
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
├── requirements.lock                # Exact transitive Python 3.11 CI lockfile
├── requirements-collection.txt      # Google Earth Engine (data collection only)
├── pyproject.toml                   # Packaging, black, pytest & coverage config
├── setup.cfg                        # flake8 + mypy config
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (tested on 3.10 and 3.11)
- **pip** 20.0 or newer
- **Git** for cloning the repository
- **Optional:** Docker for containerized development

### Quick Start (Fresh Clone to Running Tests)

The fastest path from a fresh clone to verified working tests:

```bash
# 1. Clone the repository
git clone https://github.com/Kaoserahamed/Thesis_v02.git
cd Thesis_v02

# 2. Create a virtual environment (recommended)
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate

# 3. Install dependencies from the committed lockfile
python -m pip install --upgrade pip
python -m pip install -r requirements.lock

# 4. Run the fast test suite to verify installation
python -m pytest tests/ -m "not slow" --ignore=tests/test_model_builders.py --cov=utils --cov-fail-under=70

# 5. Optional: Run the full test suite including TensorFlow model builds
python -m pytest tests/ --cov=utils --cov-report=term-missing
```

**Expected output:** All tests should pass with ≥70% coverage on the fast lane, confirming that
the repository is correctly installed and functional.

### 1. Install dependencies

For **runtime-only** (notebooks, training, inference):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
```

For **complete local development** (includes testing, linting, type checking):

```bash
pip install -r requirements-dev.txt
```

The CI pipeline uses the committed [requirements.lock](requirements.lock) for
reproducible Python 3.11 builds. This lockfile pins all transitive dependencies
to ensure consistent behavior across environments.

### 2. Refresh the lockfile (maintainers only)

After updating `requirements.txt` or `requirements-dev.txt`, regenerate the lockfile:

```bash
python -m pip install pip-tools
pip-compile --output-file=requirements.lock requirements-dev.txt
```

**Important:** Review lockfile changes as you would any dependency update. The lockfile
diff shows exactly which packages and versions changed, making security and compatibility
reviews straightforward.

### 3. Verify the build (comprehensive quality check)

Run the complete build verification workflow that CI uses:

```bash
# Build the distributable package
python -m pip install build
python -m build --outdir dist

# Verify the wheel contains all modules
python scripts/verify_dist.py dist

# Run the fast test lane with coverage gate
python -m pytest tests/ -m "not slow" --ignore=tests/test_model_builders.py \
  --cov=utils --cov=scripts --cov-fail-under=70

# Run the model-building test lane (requires TensorFlow)
python -m pytest tests/ -m "requires_tensorflow"
```

Or run all quality gates at once:

```bash
python scripts/ci_local.py --all
```

### Container and VS Code setup

The repository includes a CPU-oriented [Dockerfile](Dockerfile) and a
[devcontainer](.devcontainer/devcontainer.json) for reproducible development environments.

**Build and run the Docker image:**

```bash
docker build -t river-morphology-thesis .
docker run --rm river-morphology-thesis
```

The default command runs the local quality gates (lint, type-check, tests).

**Run JupyterLab in the container:**

```bash
docker run --rm -p 8888:8888 -v $(pwd):/workspace river-morphology-thesis \
  jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

Then open the URL shown in the terminal (includes the token).

**VS Code Dev Containers:**

1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the repository in VS Code
3. Click the green button in the bottom-left corner
4. Select "Reopen in Container"

The container automatically installs `requirements-dev.txt` and configures Python,
pytest, linting, and Jupyter extensions.

### Environment variables and data paths

Copy [.env.example](.env.example) to `.env` if you need custom data directories:

```bash
cp .env.example .env
```

**Key environment variables:**

- `YEARLY_DIR`, `QUARTERLY_DIR`, `BIMONTHLY_DIR` — paths to GeoTIFF time series
- `MLFLOW_TRACKING_URI` — MLflow server URL (defaults to `file:./mlruns`)
- `MLFLOW_EXPERIMENT_NAME` — experiment name for grouping runs
- `THESIS_DISK_FREE_GB` — minimum free disk space threshold
- `THESIS_ERROR_WEBHOOK_URL` — optional webhook for error notifications

**Important:** `.env` is ignored by Git. Never place Earth Engine credentials or
service-account JSON in it. For data collection notebooks, authenticate separately:

```bash
pip install -r requirements-collection.txt
earthengine authenticate
```

### 4. Authenticate Google Earth Engine (data collection only)

```bash
earthengine authenticate
```

### 5. Regenerate the notebooks (optional)

```bash
python scripts/generate_notebooks.py     # (re)write the self-contained notebooks
python scripts/validate_notebooks.py     # sanity-check them (valid, parseable, self-consistent)
```

## 🔄 Reproduce Results from Scratch

The following workflow demonstrates full reproducibility from a fresh clone:

### Step 1: Environment setup

```bash
# Clone and install
git clone https://github.com/Kaoserahamed/Thesis_v02.git
cd Thesis_v02
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.lock
```

### Step 2: Verify installation

```bash
# Run fast tests to confirm everything installed correctly
pytest -m "not slow" --ignore=tests/test_model_builders.py --cov=utils --cov-fail-under=70
```

### Step 3: Data preparation

Option A: Use existing data (if you have GeoTIFFs):

```bash
# Set environment variables pointing to your data
export YEARLY_DIR=/path/to/yearly_geotiffs
export QUARTERLY_DIR=/path/to/quarterly_geotiffs
export BIMONTHLY_DIR=/path/to/bimonthly_geotiffs
```

Option B: Generate data from Google Earth Engine (requires authentication):

```bash
pip install -r requirements-collection.txt
earthengine authenticate
jupyter notebook data_collection/01_yearly_data_collection.ipynb
# Run the notebook to export GeoTIFFs from GEE
```

### Step 4: Run experiments

Execute a single tracked training run with a preset configuration:

Execute a single tracked training run with a preset configuration:

```bash
python scripts/run_experiment.py \
  --preset yearly_setup1 \
  --model attention_unet_convlstm \
  --seq-len 5 \
  --seed 42
```

Or train via notebooks (self-contained, no imports from `utils/`):

```bash
jupyter nbconvert --to notebook --execute \
  models/yearly/03_yearly_attention_unet_convlstm.ipynb \
  --output executed_yearly_attention_unet_convlstm.ipynb
```

### Step 5: Verify results

All models use deterministic seeding (`np.random.seed(42)`, `tf.random.set_seed(42)`)
and the experiment presets defined in `utils/pipeline_utils.py`. MLflow tracks runs
locally in `./mlruns` by default (ignored by Git).

**Expected IoU for Attention U-Net + ConvLSTM (yearly, Setup 1, L=5):** ~0.70

To view tracked experiments:

```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

### Quick verification without full training

For a fast wiring check (1 epoch, small synthetic data):

```bash
python scripts/smoke_train.py
```

This confirms TensorFlow, the model builder, and the training loop work correctly
without waiting for full convergence.

### Troubleshooting

**Import errors:** Ensure you activated the virtual environment and installed `requirements.lock`.

**Test failures:** The test suite requires specific package versions. If tests fail after
manual dependency changes, regenerate the lockfile:

```bash
pip-compile --output-file=requirements.lock requirements-dev.txt
pip install -r requirements.lock
```

**CUDA/GPU issues:** The default installation uses `tensorflow-cpu`. For GPU training,
replace it:

```bash
pip uninstall tensorflow-cpu
pip install "tensorflow[and-cuda]>=2.15,<2.20"
```

**Out of memory:** Reduce `BATCH_SIZE` in the notebook configuration cells (default is 4).

---

## 🧪 Testing & CI

The shared library (`utils/`, `scripts/`) is covered by comprehensive pytest tests.
Every push to `main` or pull request runs the full quality gate on GitHub Actions.

## 🧪 Testing & CI

The shared library (`utils/`, `scripts/`) is covered by comprehensive pytest tests.
Every push to `main` or pull request runs the full quality gate on GitHub Actions.

### Run tests locally

**Fast test lane** (unit tests, no TensorFlow model builds, <30s):

```bash
pip install -r requirements-dev.txt
pytest -m "not slow" --ignore=tests/test_model_builders.py \
  --cov=utils --cov=scripts --cov-fail-under=70
```

**Model test lane** (builds all 5 architectures, requires TensorFlow):

```bash
pytest -m "requires_tensorflow"
```

**Full test suite with coverage report:**

```bash
pytest --cov=utils --cov=scripts --cov-report=term-missing --cov-report=html
# Open htmlcov/index.html to view detailed coverage
```

**Run all local quality gates** (lint, type-check, tests, notebook validation):

```bash
python scripts/ci_local.py
```

**Include build verification and slow tests:**

```bash
python scripts/ci_local.py --build --all
```

### CI pipeline structure

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs four parallel jobs:

1. **Lint** — black, flake8, mypy, notebook validation
2. **Test** — split into fast (70% coverage gate) and model (TensorFlow) lanes
3. **Build** — create wheel, verify contents, test installation
4. **Audit** — pip-audit for known vulnerabilities

On successful `main` branch CI, a fifth job deploys the prediction map to GitHub Pages.

### Individual quality gates

Run the same checks CI uses:

```bash
# Code formatting
black --check --line-length 100 utils tests scripts

# Linting
flake8 utils tests scripts

# Type checking
mypy utils tests scripts

# Notebook validation (structure, no external imports, self-contained)
python scripts/validate_notebooks.py

# Package build
python -m build --outdir dist
python scripts/verify_dist.py dist
```

### Test markers

Tests are organized by execution time and dependencies:

- `@pytest.mark.slow` — long-running tests (model training, large data)
- `@pytest.mark.requires_tensorflow` — tests that import TensorFlow
- No marker — fast unit tests (<1s per test)

**Skip slow tests:**

```bash
pytest -m "not slow"
```

**Run only TensorFlow tests:**

```bash
pytest -m "requires_tensorflow"
```

### Continuous integration behavior

- **Pull requests:** Run all gates, do not deploy
- **Push to main:** Run all gates, deploy docs to GitHub Pages on success
- **Dependabot PRs:** Automatic merge if all tests pass (requires maintainer approval)

---

## 🔬 Methodology Summary

### Data pipeline

1. **Collection** — yearly / quarterly / bi-monthly composites exported from GEE as 60 m GeoTIFFs
   in EPSG:4326
2. **Water detection** — `MNDWI = (Green − SWIR1) / (Green + SWIR1) > 0`; Sentinel-1 VV < −16 dB
   as cloud-independent fallback
3. **Cloud masking** — Landsat `QA_PIXEL` bits 3–4; Sentinel-2 `QA60` bits 10–11
4. **Gap filling** — 7 methods compared; BiConvLSTM achieves best IoU (0.7366)
5. **Connected-component cleaning** — keep the 3 largest water components, resize to 256×256

### Models

All architectures are implemented in `utils/model_architectures.py` and inlined into
self-contained notebooks by `scripts/generate_notebooks.py`.

| Registry key | Architecture |
|--------------|--------------|
| `convlstm` | Standalone ConvLSTM |
| `unet_lstm` | U-Net with ConvLSTM bottleneck |
| `attention_unet_convlstm` | Attention-gated U-Net with ConvLSTM (best for yearly/quarterly) |
| `swin_st` | Swin spatio-temporal transformer |
| `vit_st` | ViT / ViViT spatio-temporal model |

**Training configuration:**

- **Loss:** `L_Total = L_BCE + L_Dice` (validated by ablation study)
- **Optimizer:** Adam, lr = 1e-4, clipnorm = 1.0
- **Sequence lengths:** yearly {4,5,6}, quarterly {6,8,10}, bi-monthly {6,9,12}
- **Temporal split:** Setup 1 (train ≤ 2015 / test 2016–2025), Setup 2 (train ≤ 2020 / test 2021–2025)
- **Callbacks:** ModelCheckpoint, EarlyStopping (patience 20), ReduceLROnPlateau (patience 7)
- **Reproducibility:** Fixed seeds (42) for NumPy and TensorFlow

### Evaluation metrics

- **Spatial overlap:** IoU, Dice coefficient
- **Classification:** Precision, Recall
- **Area change:** Signed difference ΔA = A_pred − A_true (km²)

### Experiment tracking

MLflow tracks all experiments with reproducible configurations:

```python
from utils.experiment_tracking import tracked_run, enable_keras_autolog, log_metrics

with tracked_run(run_name="attention-unet-yearly-setup1"):
    enable_keras_autolog()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val))
    log_metrics({"test_iou": iou, "test_dice": dice})
```

**Local tracking:** Runs write to `./mlruns` (Git-ignored)  
**Shared tracking:** Set `MLFLOW_TRACKING_URI` to a remote MLflow server

---

## 📖 Usage Examples

### Data collection

```bash
# Authenticate with Google Earth Engine
pip install -r requirements-collection.txt
earthengine authenticate

# Export yearly composites
jupyter notebook data_collection/01_yearly_data_collection.ipynb
```

### Analysis workflows

```bash
# Gap-filling method comparison
jupyter notebook analysis/01_gap_filling_comparison.ipynb

# Ablation study (model configuration sensitivity)
jupyter notebook analysis/05_ablation_study.ipynb

# Long-term forecast and risk mapping (2026–2040)
jupyter notebook analysis/02_long_term_prediction.ipynb
```

### Model training

```bash
# Single architecture training (yearly Attention U-Net+ConvLSTM)
jupyter notebook models/yearly/03_yearly_attention_unet_convlstm.ipynb

# Full architecture comparison (all 5 models, yearly resolution)
jupyter notebook models/00_yearly_model_comparison.ipynb

# Different resolution (bi-monthly U-Net+LSTM)
jupyter notebook models/bimonthly/02_bimonthly_unet_lstm.ipynb
```

### Command-line training

```bash
# Single tracked experiment with preset configuration
python scripts/run_experiment.py \
  --preset yearly_setup1 \
  --model attention_unet_convlstm \
  --seq-len 5 \
  --seed 42

# View tracked runs
mlflow ui  # http://localhost:5000
```

---

## 🧩 Design Principles

- **Single source of truth:** All architectures, losses, and metrics defined in `utils/` — 
  notebooks inline these for self-containment
- **No data leakage:** `prepare_split()` validates that test years never appear in training
- **Reproducibility:** Deterministic seeding (seed=42), committed lockfile, MLflow tracking
- **Self-contained notebooks:** Each can run independently after code inlining via 
  `scripts/generate_notebooks.py`
- **Comprehensive testing:** 70% coverage requirement, fast/slow test separation, CI on every push

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [`README.md`](README.md) | This file — overview, getting started, usage examples |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history and notable changes |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development setup, testing, code quality, commit guidelines |
| [`docs/predefence_report.md`](docs/predefence_report.md) | Full thesis report (Markdown) |
| [`docs/methodology.md`](docs/methodology.md) | Methodology summary |
| [`docs/repository_guide.md`](docs/repository_guide.md) | How every folder/notebook fits together |
| [`docs/deployment_and_reproducibility.md`](docs/deployment_and_reproducibility.md) | Deployment workflows, CI/CD, dependency management, reproducibility guarantees |
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