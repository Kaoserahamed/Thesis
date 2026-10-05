# Changelog

All notable changes to the River Morphology Prediction project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive ablation study notebook analyzing model configuration sensitivity
  - Loss function comparison (combined BCE+Dice vs. Dice-only vs. BCE-only)
  - Sequence length analysis (L ∈ {3, 5, 7, 10})
  - Dropout regularization impact
  - Learning rate sensitivity (1e-3, 1e-4, 1e-5)
  - Attention mechanism contribution
- Deployment and reproducibility documentation (`docs/deployment_and_reproducibility.md`)
  - Reproducibility guarantees (seeding, lockfiles, temporal splits)
  - Dependency management procedures
  - CI/CD pipeline documentation
  - MLflow tracking setup and best practices
  - Container deployment workflows
  - Maintenance checklists (weekly, monthly, quarterly)
- Enhanced README with fresh-clone quick start workflow
  - Step-by-step installation instructions
  - Build verification commands
  - Troubleshooting guide
  - Testing section with markers and CI structure
- Weekly Dependabot configuration for proactive dependency updates
  - Automatic PR creation and reviewer assignment
  - Semantic commit message prefixes
  - Grouped minor/patch updates

### Changed
- Raised CI coverage threshold from 50% to 70% for fast test lane
- Improved Dependabot schedule from monthly to weekly updates
- Enhanced README structure with clear sections for reproducibility
- Updated test documentation with detailed marker usage

### Fixed
- Clarified lockfile usage and regeneration procedures
- Improved Docker container documentation with volume mounting examples
- Enhanced environment variable configuration documentation

## [1.0.0] - 2026-09-22

### Added
- Five spatiotemporal deep learning architectures for river morphology prediction:
  - Standalone ConvLSTM
  - U-Net + LSTM
  - Attention U-Net + ConvLSTM (best for yearly/quarterly)
  - Swin Spatio-Temporal Transformer
  - Vision Transformer (ViViT)
- Three temporal resolutions: yearly, quarterly, bi-monthly
- Gap-filling comparison with 7 methods (BiConvLSTM achieves IoU 0.7366)
- Comprehensive test suite (17 test files, 1:3 test-to-source ratio)
- CI/CD pipeline with GitHub Actions:
  - Code quality gates (black, flake8, mypy)
  - Split test lanes (fast unit tests, TensorFlow model tests)
  - Lockfile verification
  - Automatic GitHub Pages deployment
- MLflow experiment tracking integration
- Interactive prediction map (2026–2040 forecast)
- Self-contained Jupyter notebooks with inlined utilities
- Docker and VS Code Dev Container support
- Structured JSON logging framework
- Health check registry
- Google Earth Engine data collection notebooks
- 38-year time series (1987–2025) dataset workflow

### Results
- Best yearly model: Attention U-Net+ConvLSTM (IoU 0.7005, Dice 0.8236)
- Best quarterly model: Attention U-Net+ConvLSTM (IoU 0.6956, Dice 0.8139)
- Best bi-monthly model: U-Net+LSTM (IoU 0.7791, Dice 0.8701)
- Mean annual centerline migration: 255.4 m/yr (peak: 1485.5 m/yr in 1998-1999)
- 2026–2040 forecast: +290 km² (+56%) expansion projected

### Documentation
- Complete thesis report (`docs/predefence_report.md`, `docs/Thesis_Paper.pdf`)
- Methodology summary (`docs/methodology.md`)
- Repository structure guide (`docs/repository_guide.md`)
- Comprehensive README with results tables and usage examples

[Unreleased]: https://github.com/Kaoserahamed/Thesis_v02/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Kaoserahamed/Thesis_v02/releases/tag/v1.0.0
