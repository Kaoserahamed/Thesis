# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
where versioning is applicable.

## [Unreleased]

### Added

- Ongoing improvements to experiment reproducibility, validation, and research documentation.

## [0.1.0] - 2026-09-22

### Added

- Five TensorFlow model architectures with a shared model registry and dispatcher.
- Yearly, quarterly, and bi-monthly data pipelines, temporal splitting, evaluation metrics,
  and experiment tracking through local or remote MLflow.
- Self-contained generated notebooks for data collection, analysis, model comparison, and
  forecasting experiments.
- Reproducible command-line experiment execution with fixed NumPy/TensorFlow seeds and
  resolution and architecture selection.
- Unit, model smoke, notebook validation, logging, and experiment tracking tests.
- CI checks for formatting, linting, type checking, notebook validation, dependency auditing,
  and test coverage.

### Documentation

- Added setup, dependency management, repository guidance, security, and reproducibility
  documentation.

[Unreleased]: https://github.com/Kaoserahamed/Thesis/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Kaoserahamed/Thesis/releases/tag/v0.1.0