# Deployment and Reproducibility Guide

This document provides comprehensive guidance on deploying the river morphology prediction system, ensuring reproducible experiments, and maintaining the codebase over time.

## Table of Contents

- [Reproducibility Guarantees](#reproducibility-guarantees)
- [Dependency Management](#dependency-management)
- [CI/CD Pipeline](#cicd-pipeline)
- [Deployment Workflows](#deployment-workflows)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Lockfile Policy](#lockfile-policy)
- [Container Deployment](#container-deployment)
- [Maintenance Procedures](#maintenance-procedures)

---

## Reproducibility Guarantees

This repository implements multiple layers of reproducibility to ensure experiments can be faithfully reproduced:

### 1. Deterministic Seeding

All experiments use fixed random seeds across NumPy and TensorFlow:

```python
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)
```

This seeding is applied:
- In every model training notebook
- In `scripts/run_experiment.py`
- In the test suite (`tests/conftest.py`)
- Before any data shuffling or model initialization

### 2. Committed Lockfile

The repository includes `requirements.lock` that pins **all transitive dependencies** to exact versions tested in CI:

```bash
# Example from requirements.lock
numpy==1.24.3
tensorflow-cpu==2.15.0
pandas==2.0.3
# ... etc
```

This ensures that a fresh clone on any machine will install identical package versions, eliminating "works on my machine" issues.

### 3. Strict Temporal Splits

The `prepare_split()` function in `utils/pipeline_utils.py` enforces temporal splitting with validation:

```python
def prepare_split(X, y, target_years, input_years, cutoff_year):
    """
    Split sequences by target year with validation that prevents data leakage.
    
    Raises:
        ValueError: if any test year appears in the training set
    """
    # ... validation logic ensures no test year in train
```

Two standard evaluation setups are defined:
- **Setup 1:** Train ≤ 2015, Test 2016–2025 (10-year holdout)
- **Setup 2:** Train ≤ 2020, Test 2021–2025 (5-year holdout)

### 4. Experiment Presets

All training configurations are codified in `utils/pipeline_utils.py`:

```python
EXPERIMENT_PRESETS = {
    "yearly_setup1": {
        "cutoff_year": 2015,
        "sequence_lengths": [4, 5, 6],
        "resolution": "yearly",
        "data_dir_env": "YEARLY_DIR",
    },
    # ... more presets
}
```

This eliminates configuration drift between experiments.

### 5. MLflow Tracking

Every experiment can be tracked with full parameter and metric logging:

```python
with mlflow.start_run(run_name="attention-unet-yearly-setup1"):
    mlflow.log_params({
        "model": "attention_unet_convlstm",
        "seq_len": 5,
        "cutoff_year": 2015,
        "learning_rate": 1e-4,
        "seed": 42,
    })
    # ... training
    mlflow.log_metrics({"test_iou": 0.7005, "test_dice": 0.8236})
    mlflow.log_artifact("model_checkpoint.keras")
```

### Reproducing a Published Result

To reproduce the thesis results exactly:

```bash
# 1. Clone at the exact commit
git clone https://github.com/Kaoserahamed/Thesis_v02.git
cd Thesis_v02
git checkout <commit-hash-from-paper>

# 2. Install from committed lockfile
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.lock

# 3. Verify installation
pytest -m "not slow" --ignore=tests/test_model_builders.py --cov-fail-under=70

# 4. Set data paths
export YEARLY_DIR=/path/to/yearly_geotiffs
export MLFLOW_TRACKING_URI=file:./mlruns

# 5. Run the exact experiment
python scripts/run_experiment.py \
  --preset yearly_setup1 \
  --model attention_unet_convlstm \
  --seq-len 5 \
  --seed 42

# 6. Expected result: IoU ≈ 0.70, Dice ≈ 0.82
```

---

## Dependency Management

### Philosophy

The repository uses a **two-tier dependency strategy**:

1. **`requirements.txt`**: Broad compatibility ranges for runtime dependencies
2. **`requirements.lock`**: Exact pinned versions for CI and reproducible builds

This balances flexibility (users can use newer compatible versions) with reproducibility (CI and published results use exact versions).

### Updating Dependencies

#### For Users (Manual Updates)

```bash
# Update a specific package
pip install --upgrade tensorflow-cpu

# Regenerate lockfile from updated environment
pip freeze > requirements.lock
```

#### For Maintainers (Systematic Updates)

```bash
# 1. Update version ranges in requirements.txt
vim requirements.txt
# Change: tensorflow-cpu>=2.15,<2.20
# To:     tensorflow-cpu>=2.16,<2.21

# 2. Regenerate lockfile with pip-compile
pip install pip-tools
pip-compile --output-file=requirements.lock requirements-dev.txt

# 3. Install and test
pip install -r requirements.lock
pytest --cov=utils --cov-fail-under=70

# 4. Commit both files
git add requirements.txt requirements.lock
git commit -m "deps: update TensorFlow to 2.16.x range

Regenerated lockfile with pip-compile. All tests pass with new versions."
```

### Dependabot Configuration

The repository uses Dependabot for automated dependency updates:

- **Schedule:** Weekly on Mondays at 09:00 Bangladesh time
- **Python deps:** Up to 10 PRs, grouped by minor/patch versions
- **GitHub Actions:** Up to 5 PRs
- **Auto-review:** Assigned to repository maintainers

Dependabot PRs trigger full CI:
1. Lint and type-check pass
2. Full test suite passes (fast + model lanes)
3. Package builds successfully
4. Notebook validation passes

**If all gates pass**, the dependency update is safe to merge.

### Security Updates

For security-critical updates:

```bash
# Check for known vulnerabilities
pip install pip-audit
pip-audit -r requirements.txt

# Update affected package immediately
pip install --upgrade <vulnerable-package>
pip-compile --output-file=requirements.lock requirements-dev.txt

# Test and merge urgently
pytest --cov=utils --cov-fail-under=70
git add requirements.lock
git commit -m "deps(security): update <package> to address CVE-XXXX-YYYY"
```

---

## CI/CD Pipeline

### Pipeline Structure

The GitHub Actions workflow (`.github/workflows/ci.yml`) consists of 5 jobs:

```
┌─────────────────────────────────────────────────┐
│                   Trigger                        │
│    (push to main OR pull_request to main)       │
└──────────────┬──────────────────────────────────┘
               │
               ├─────────────────┬─────────────────┬────────────┬──────────────┐
               ↓                 ↓                 ↓            ↓              ↓
          ┌─────────┐       ┌────────┐      ┌─────────┐  ┌─────────┐   ┌──────────┐
          │  Lint   │       │  Test  │      │  Build  │  │  Audit  │   │  Deploy  │
          │  Job    │       │  Job   │      │   Job   │  │   Job   │   │   Job    │
          └─────────┘       └────────┘      └─────────┘  └─────────┘   └──────────┘
               │                 │                │            │              │
               │            ┌────┴────┐           │            │              │
               │            ↓         ↓           │            │              │
               │       ┌────────┐ ┌──────────┐   │            │              │
               │       │ Fast   │ │  Models  │   │            │              │
               │       │ Tests  │ │  Tests   │   │            │              │
               │       └────────┘ └──────────┘   │            │              │
               │                                  │            │              │
               └──────────────────────────────────┴────────────┘              │
                                                                               │
                              (on main branch only) ──────────────────────────┘
```

### Job Details

#### 1. Lint Job

Validates code quality and notebook structure:

```yaml
- black --check --line-length 100 utils tests scripts
- flake8 utils tests scripts
- mypy utils tests scripts
- python scripts/validate_notebooks.py
```

**Duration:** ~2 minutes

#### 2. Test Job (Matrix)

Runs two test lanes in parallel:

**Fast Lane:**
- Unit tests only (no model builds)
- 70% coverage requirement
- Duration: ~3 minutes

**Model Lane:**
- Builds all 5 architectures
- Smoke tests forward passes
- Duration: ~8 minutes

#### 3. Build Job

Verifies package can be built and distributed:

```bash
python -m build --outdir dist
python scripts/verify_dist.py dist
pip install --no-deps dist/*.whl
```

**Duration:** ~2 minutes

#### 4. Audit Job

Checks for known security vulnerabilities:

```bash
pip-audit -r requirements.txt
```

**Duration:** ~1 minute

#### 5. Deploy Job (main branch only)

Deploys `docs/` folder to GitHub Pages:

```yaml
- Configure GitHub Pages
- Upload docs/ as artifact
- Deploy to gh-pages branch
```

**Duration:** ~1 minute

### Total Pipeline Time

- **Pull Request:** ~10 minutes (no deploy)
- **Main Branch Push:** ~11 minutes (includes deploy)

### Running CI Locally

Replicate the entire CI pipeline on your machine:

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all gates
python scripts/ci_local.py

# Run with build and slow tests
python scripts/ci_local.py --build --all
```

### CI Troubleshooting

**Lint failures:**
```bash
# Auto-fix formatting
black utils tests scripts

# Check what flake8 complains about
flake8 utils tests scripts

# Fix type errors shown by mypy
mypy utils tests scripts
```

**Test failures:**
```bash
# Run failing test in isolation
pytest tests/test_module.py::test_function -v

# Run with full output
pytest tests/test_module.py -v -s

# Debug with breakpoint
# Add: import pdb; pdb.set_trace()
pytest tests/test_module.py --pdb
```

**Lockfile drift:**
```bash
# Regenerate lockfile
pip-compile --output-file=requirements.lock requirements-dev.txt

# Verify CI lockfile check passes
git diff --exit-code requirements.lock
```

---

## Deployment Workflows

### Local Development Deployment

For running notebooks locally with tracked experiments:

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.lock

# 2. Configure data paths
export YEARLY_DIR=/path/to/data/yearly
export QUARTERLY_DIR=/path/to/data/quarterly
export BIMONTHLY_DIR=/path/to/data/bimonthly

# 3. Configure MLflow (optional, defaults to ./mlruns)
export MLFLOW_TRACKING_URI=file:./mlruns
export MLFLOW_EXPERIMENT_NAME=river-morphology

# 4. Run experiments
python scripts/run_experiment.py --preset yearly_setup1 --model attention_unet_convlstm --seq-len 5

# 5. View results
mlflow ui  # http://localhost:5000
```

### Container Deployment

#### CPU-Only Container (Development/CI)

```bash
# Build
docker build -t river-morphology:latest .

# Run quality gates
docker run --rm river-morphology:latest

# Interactive shell
docker run --rm -it river-morphology:latest /bin/bash

# Mount data and run training
docker run --rm \
  -v /host/data:/data \
  -v $(pwd)/mlruns:/app/mlruns \
  -e YEARLY_DIR=/data/yearly \
  river-morphology:latest \
  python scripts/run_experiment.py --preset yearly_setup1 --model convlstm --seq-len 5
```

#### GPU Container (Production Training)

Create `Dockerfile.gpu`:

```dockerfile
FROM tensorflow/tensorflow:2.15.0-gpu

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "scripts/ci_local.py", "--all"]
```

Build and run:

```bash
docker build -f Dockerfile.gpu -t river-morphology:gpu .

docker run --rm --gpus all \
  -v /host/data:/data \
  -v $(pwd)/mlruns:/app/mlruns \
  -e YEARLY_DIR=/data/yearly \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  river-morphology:gpu \
  python scripts/run_experiment.py --preset yearly_setup1 --model attention_unet_convlstm --seq-len 5
```

### GitHub Pages Deployment

The prediction map is automatically deployed to GitHub Pages on every successful push to `main`:

**Manual deployment trigger:**

```bash
# Ensure docs/ contains latest prediction maps
python analysis/02_long_term_prediction.ipynb  # generates HTML maps

# Commit and push
git add docs/
git commit -m "docs: update prediction maps with 2026 data"
git push origin main

# CI will deploy to https://kaoserahamed.github.io/Thesis/
```

**Pages configuration:**
- Source: GitHub Actions deployment
- Branch: `gh-pages` (auto-created)
- Directory: `/` (root of gh-pages branch)

### Cloud Deployment (AWS/GCP/Azure)

For large-scale training on cloud GPU instances:

**1. Prepare cloud storage:**

```bash
# AWS S3
aws s3 sync data/raw/ s3://your-bucket/river-morphology/data/
aws s3 sync mlruns/ s3://your-bucket/river-morphology/mlruns/

# GCP Cloud Storage
gsutil -m rsync -r data/raw/ gs://your-bucket/river-morphology/data/
gsutil -m rsync -r mlruns/ gs://your-bucket/river-morphology/mlruns/
```

**2. Launch GPU instance:**

```bash
# AWS EC2 with GPU
aws ec2 run-instances \
  --image-id ami-0c9c942bd7bf113a2 \
  --instance-type p3.2xlarge \
  --key-name your-key \
  --security-group-ids sg-xxxxx \
  --user-data file://cloud-init.sh
```

**3. Cloud-init script (`cloud-init.sh`):**

```bash
#!/bin/bash
apt-get update
apt-get install -y python3-pip git

git clone https://github.com/Kaoserahamed/Thesis_v02.git /opt/river-morphology
cd /opt/river-morphology

python3 -m pip install -r requirements.lock

# Download data from S3
aws s3 sync s3://your-bucket/river-morphology/data/ /data/

# Run experiment
export YEARLY_DIR=/data/yearly
export MLFLOW_TRACKING_URI=s3://your-bucket/river-morphology/mlruns

python3 scripts/run_experiment.py \
  --preset yearly_setup1 \
  --model attention_unet_convlstm \
  --seq-len 5 \
  --seed 42

# Sync results back
aws s3 sync /opt/river-morphology/outputs/ s3://your-bucket/river-morphology/outputs/
aws s3 sync /opt/river-morphology/mlruns/ s3://your-bucket/river-morphology/mlruns/
```

---

## MLflow Experiment Tracking

### Local Tracking Setup

```bash
# Default: tracks to ./mlruns (Git-ignored)
python scripts/run_experiment.py --preset yearly_setup1 --model convlstm --seq-len 5

# View tracked experiments
mlflow ui
# Open http://localhost:5000
```

### Remote Tracking Server

For team collaboration, set up a shared MLflow server:

**Server setup:**

```bash
# Install MLflow server
pip install mlflow

# Run with PostgreSQL backend and S3 artifact store
mlflow server \
  --backend-store-uri postgresql://user:pass@db-host:5432/mlflow \
  --default-artifact-root s3://your-bucket/mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

**Client configuration:**

```bash
# Point experiments to remote server
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export MLFLOW_EXPERIMENT_NAME=river-morphology-team

python scripts/run_experiment.py --preset yearly_setup1 --model attention_unet_convlstm --seq-len 5
```

### Tracking Best Practices

**1. Use descriptive run names:**

```python
run_name = f"{model_key}_L{seq_len}_{setup_name}_seed{seed}"
with mlflow.start_run(run_name=run_name):
    # ...
```

**2. Log all hyperparameters:**

```python
mlflow.log_params({
    "model": "attention_unet_convlstm",
    "seq_len": 5,
    "cutoff_year": 2015,
    "learning_rate": 1e-4,
    "batch_size": 4,
    "dropout_rate": 0.2,
    "use_attention": True,
    "loss_function": "combined",
    "seed": 42,
})
```

**3. Log metrics at multiple stages:**

```python
# Training metrics
mlflow.log_metric("final_train_loss", history.history["loss"][-1])
mlflow.log_metric("final_val_loss", history.history["val_loss"][-1])
mlflow.log_metric("epochs_trained", len(history.history["loss"]))

# Test metrics
mlflow.log_metric("test_iou", test_iou)
mlflow.log_metric("test_dice", test_dice)
mlflow.log_metric("test_precision", test_precision)
mlflow.log_metric("test_recall", test_recall)
```

**4. Log artifacts:**

```python
# Model checkpoint
mlflow.log_artifact("checkpoints/best_model.keras", artifact_path="checkpoints")

# Evaluation plots
mlflow.log_artifact("outputs/iou_vs_L.png", artifact_path="plots")

# Result tables
mlflow.log_artifact("outputs/metrics.csv", artifact_path="results")
```

### MLflow Isolation and Network Independence

**Local-Only Operation:**

By default, MLflow operates entirely locally without network access:

```python
# Default tracking URI (no network required)
MLFLOW_TRACKING_URI = "file:./mlruns"

# All tracking data stored locally
# - Run metadata: ./mlruns/<experiment_id>/<run_id>/meta.yaml
# - Metrics: ./mlruns/<experiment_id>/<run_id>/metrics/
# - Artifacts: ./mlruns/<experiment_id>/<run_id>/artifacts/
```

**Test Suite Isolation:**

The test suite runs completely offline:

1. **No external services required** — tests use in-memory data and temporary directories
2. **MLflow defaults to local** — `MLFLOW_TRACKING_URI` defaults to `file:./mlruns`
3. **No network calls** — all dependencies installed during `pip install`
4. **Verify isolation** — run `python scripts/verify_test_isolation.py`

**Verification Script:**

```bash
# Verify tests can run without network
python scripts/verify_test_isolation.py

# Expected output:
# ✓ MLflow defaults to local file-based tracking
# ✓ No external service imports in tests
# ✓ Test fixtures use temporary/local storage
# ✓ Fast test suite passes without network
```

This isolation ensures:
- Tests run on CI without credentials
- Fresh clones work immediately
- Experiments are reproducible offline
- No accidental data leakage to external services

### Comparing Runs

```python
import mlflow
import pandas as pd

# Search runs by experiment
experiment = mlflow.get_experiment_by_name("river-morphology")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# Filter by tag or param
attention_runs = runs[runs["params.use_attention"] == "True"]

# Sort by metric
best_runs = runs.sort_values("metrics.test_iou", ascending=False).head(10)

print(best_runs[["run_id", "params.model", "params.seq_len", "metrics.test_iou"]])
```

---

## Lockfile Policy

### Why We Use Lockfiles

Lockfiles provide **exact reproducibility** by pinning every transitive dependency:

```
requirements.txt (what we want):
  tensorflow-cpu>=2.15,<2.20
  numpy>=1.24,<2.0

requirements.lock (what we actually get):
  tensorflow-cpu==2.15.0
  numpy==1.24.3
  six==1.16.0           # transitive dep of tensorflow
  protobuf==4.25.1      # transitive dep of tensorflow
  # ... 50+ more packages
```

### Lockfile Lifecycle

**Generation:**

```bash
pip-compile --output-file=requirements.lock requirements-dev.txt
```

**Installation:**

```bash
pip install -r requirements.lock
```

**Verification (CI):**

```bash
pip-compile --quiet --no-upgrade --output-file=requirements.lock requirements-dev.txt
git diff --exit-code requirements.lock  # Fails if lockfile is stale
```

### When to Update Lockfiles

1. **After changing requirements.txt or requirements-dev.txt**
2. **Weekly via Dependabot** (automatic PRs)
3. **For security patches** (urgent, manual)
4. **Before major releases** (ensure latest compatible versions)

### Lockfile Conflicts

If `git pull` causes lockfile conflicts:

```bash
# Accept the remote version
git checkout --theirs requirements.lock

# Reinstall
pip install -r requirements.lock

# Run tests to verify
pytest -m "not slow" --cov-fail-under=70
```

Or regenerate from scratch:

```bash
# Discard local changes
git checkout requirements.lock

# Pull latest
git pull

# Regenerate with latest pip-compile
pip-compile --output-file=requirements.lock requirements-dev.txt

# Commit if changed
git add requirements.lock
git commit -m "deps: regenerate lockfile with pip-compile"
```

---

## Container Deployment

### Docker Image Layers

The Dockerfile uses multi-stage pattern for efficiency:

```dockerfile
# Stage 1: Base with Python and system dependencies
FROM python:3.11-slim AS base
RUN apt-get update && apt-get install -y git

# Stage 2: Dependencies (cached layer)
FROM base AS dependencies
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Stage 3: Application code
FROM dependencies AS app
COPY . /app
WORKDIR /app
RUN pip install -e .

# Default command
CMD ["python", "scripts/ci_local.py"]
```

### Docker Compose for Development

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  jupyter:
    build: .
    command: jupyter lab --ip=0.0.0.0 --no-browser --allow-root
    ports:
      - "8888:8888"
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./mlruns:/app/mlruns
    environment:
      - YEARLY_DIR=/app/data/raw/yearly
      - QUARTERLY_DIR=/app/data/raw/quarterly
      - BIMONTHLY_DIR=/app/data/raw/bimonthly
      - MLFLOW_TRACKING_URI=file:/app/mlruns

  mlflow:
    image: python:3.11-slim
    command: mlflow ui --host 0.0.0.0 --backend-store-uri file:/mlruns
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
```

Usage:

```bash
docker-compose up jupyter  # JupyterLab on http://localhost:8888
docker-compose up mlflow   # MLflow UI on http://localhost:5000
```

---

## Maintenance Procedures

### Weekly Maintenance

```bash
# 1. Pull latest changes
git pull origin main

# 2. Check for Dependabot PRs
gh pr list --label dependencies

# 3. Review and merge passing PRs
gh pr merge <pr-number> --auto --squash

# 4. Update local environment
pip install -r requirements.lock

# 5. Run full test suite
pytest --cov=utils --cov-report=term-missing
```

### Monthly Maintenance

```bash
# 1. Audit dependencies
pip-audit -r requirements.txt

# 2. Update GitHub Actions
# Check .github/workflows/ci.yml for newer action versions

# 3. Review test coverage
pytest --cov=utils --cov-report=html
# Open htmlcov/index.html

# 4. Update documentation
# Check for outdated examples, broken links, etc.

# 5. Regenerate notebooks
python scripts/generate_notebooks.py
python scripts/validate_notebooks.py
```

### Quarterly Maintenance

```bash
# 1. Major dependency updates
pip-compile --upgrade --output-file=requirements.lock requirements-dev.txt
pip install -r requirements.lock
pytest --cov=utils --cov-fail-under=70

# 2. Review and update Docker base images
# Update FROM python:3.11-slim to newer Python version if available

# 3. Performance profiling
python -m cProfile -o profile.stats scripts/run_experiment.py ...
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# 4. Security audit
bandit -r utils/ scripts/
safety check -r requirements.txt
```

### Release Checklist

Before creating a new release:

- [ ] All tests passing on `main` branch
- [ ] Documentation up to date (README, docs/)
- [ ] Notebooks validated and executable
- [ ] Lockfile current and consistent
- [ ] CHANGELOG.md updated with user-facing changes
- [ ] Version bumped in `pyproject.toml`
- [ ] Git tag created: `git tag -a v1.0.0 -m "Release v1.0.0"`
- [ ] Tag pushed: `git push origin v1.0.0`
- [ ] GitHub Release created with release notes
- [ ] Docker image built and tagged
- [ ] PyPI package published (if applicable)

---

## Summary

This deployment and reproducibility guide ensures:

✅ **Experiments are reproducible** via lockfiles, seeding, and MLflow tracking  
✅ **Dependencies stay current** via Dependabot and regular audits  
✅ **CI catches regressions** before they reach main branch  
✅ **Deployments are automated** for GitHub Pages  
✅ **Containers provide isolation** for consistent environments  
✅ **Maintenance is systematic** with clear procedures

For questions or issues, refer to:
- [README.md](../README.md) for general usage
- [methodology.md](methodology.md) for scientific methods
- [repository_guide.md](repository_guide.md) for codebase navigation
