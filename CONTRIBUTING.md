# Contributing to River Morphology Prediction

Thank you for your interest in contributing to this project! This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)

---

## Code of Conduct

This is an academic research project from Shahjalal University of Science and Technology. We expect all contributors to:

- Be respectful and professional
- Provide constructive feedback
- Focus on scientific rigor and reproducibility
- Acknowledge the work of others

## Getting Started

### Prerequisites

- **Python 3.8+** (tested on 3.10 and 3.11)
- **Git** for version control
- **pip** 20.0 or newer
- **Optional:** Docker for containerized development

### Initial Setup

1. **Fork and clone the repository:**

```bash
git clone https://github.com/YOUR-USERNAME/Thesis_v02.git
cd Thesis_v02
```

2. **Create a virtual environment:**

```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate
```

3. **Install dependencies from the committed lockfile:**

```bash
python -m pip install --upgrade pip
pip install -r requirements.lock
```

4. **Install development dependencies:**

```bash
pip install -r requirements-dev.txt
```

5. **Verify installation by running tests:**

```bash
pytest -m "not slow" --ignore=tests/test_model_builders.py --cov=utils --cov-fail-under=70
```

All tests should pass with ≥70% coverage.

### Development in Docker

Alternatively, use the provided Docker container:

```bash
docker build -t river-morphology .
docker run --rm -it river-morphology /bin/bash
```

Or with VS Code Dev Containers:
1. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
2. Open the repository in VS Code
3. Press F1 → "Dev Containers: Reopen in Container"

---

## Development Workflow

### Project Structure

Key directories:
- **`utils/`** — Shared Python library (single source of truth for models, losses, metrics)
- **`scripts/`** — CLI tools, notebook generation, validation
- **`tests/`** — Pytest suite covering the shared library
- **`models/`** — Self-contained training notebooks (generated from `utils/`)
- **`analysis/`** — Cross-cutting analysis notebooks
- **`docs/`** — Documentation and thesis report

For detailed structure, see [`docs/repository_guide.md`](docs/repository_guide.md).

### Making Changes

1. **Create a feature branch:**

```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes:**
   - Edit code in `utils/` or `scripts/`
   - Add corresponding tests in `tests/`
   - Update documentation if needed

3. **If modifying model code:**

After changing `utils/model_architectures.py`, `utils/model_losses.py`, or `utils/pipeline_utils.py`:

```bash
# Regenerate self-contained notebooks
python scripts/generate_notebooks.py

# Validate notebooks are well-formed and self-contained
python scripts/validate_notebooks.py
```

---

## Testing

### Running Tests

**Fast test lane** (unit tests, no TensorFlow model builds):

```bash
pytest -m "not slow" --ignore=tests/test_model_builders.py --cov=utils --cov-fail-under=70
```

**Model test lane** (builds all 5 architectures):

```bash
pytest -m "requires_tensorflow"
```

**Full test suite with coverage report:**

```bash
pytest --cov=utils --cov=scripts --cov-report=term-missing --cov-report=html
# Open htmlcov/index.html to view detailed coverage
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names: `test_<function>_<scenario>`
- Use pytest fixtures from `tests/conftest.py`
- Mark slow tests with `@pytest.mark.slow`
- Mark TensorFlow tests with `@pytest.mark.requires_tensorflow`

**Example test:**

```python
import pytest
from utils.model_losses import dice_coefficient
import numpy as np

def test_dice_coefficient_perfect_overlap():
    """Dice coefficient should be 1.0 for identical masks."""
    y_true = np.ones((1, 256, 256, 1))
    y_pred = np.ones((1, 256, 256, 1))
    result = dice_coefficient(y_true, y_pred).numpy()
    assert result > 0.99  # Account for float precision
```

### Test Coverage Requirements

- Minimum **70% coverage** for the fast test lane (enforced by CI)
- New code should include tests that cover:
  - Normal/expected behavior
  - Edge cases
  - Error conditions

---

## Code Quality

### Formatting and Linting

The project enforces code quality through CI. Run these checks locally before committing:

**1. Code formatting (black):**

```bash
black --check --line-length 100 utils tests scripts
```

To auto-fix formatting:

```bash
black --line-length 100 utils tests scripts
```

**2. Linting (flake8):**

```bash
flake8 utils tests scripts
```

Configuration is in `setup.cfg`.

**3. Type checking (mypy):**

```bash
mypy utils tests scripts
```

Configuration is in `setup.cfg`.

**4. Run all quality gates at once:**

```bash
python scripts/ci_local.py
```

### Code Style Guidelines

- **Line length:** Maximum 100 characters (black enforced)
- **Type hints:** Use type annotations for function signatures
- **Docstrings:** Use Google-style docstrings for public functions
- **Imports:** Group by standard library, third-party, local (isort compatible)
- **Naming:**
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

**Example function:**

```python
def prepare_split(
    X: np.ndarray,
    y: np.ndarray,
    target_years: np.ndarray,
    input_years: np.ndarray,
    cutoff_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split sequences into train/validation/test by target year.
    
    Args:
        X: Input sequences of shape (n_samples, seq_len, height, width, channels)
        y: Target masks of shape (n_samples, height, width, channels)
        target_years: Year of each target mask
        input_years: Years used in each input sequence
        cutoff_year: Train on years ≤ cutoff, test on years > cutoff
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, test_years)
        
    Raises:
        ValueError: if any test year appears in training sequences
    """
    # Implementation...
```

---

## Commit Guidelines

This project follows [Conventional Commits](https://www.conventionalcommits.org/) for clear commit history.

### Commit Message Format

```
<type>(<scope>): <short summary>

<optional body>

<optional footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring (no feature or bug fix)
- `test`: Adding or updating tests
- `ci`: CI/CD changes
- `deps`: Dependency updates
- `perf`: Performance improvements

**Scopes (optional):**
- `utils`: Changes to shared utilities
- `scripts`: Changes to CLI scripts
- `tests`: Changes to test suite
- `ci`: CI/CD configuration
- `models`: Model architecture changes
- `analysis`: Analysis notebooks

**Examples:**

```bash
feat(utils): add signed distance field interpolation for gap filling

Implements SDF-based temporal interpolation that preserves geometric
topology better than linear or spline methods. Achieves IoU 0.7290
on the gap-filling benchmark.

Closes #42
```

```bash
fix(tests): correct temporal split validation for edge case

The prepare_split function was incorrectly flagging valid splits when
target_year == cutoff_year. Updated validation logic and added test case.
```

```bash
docs(README): add troubleshooting section for common install issues

Addresses numpy binary incompatibility and TensorFlow GPU configuration
based on user feedback.
```

### Commit Best Practices

1. **One logical change per commit** — if you can't summarize it in one sentence, split it
2. **Include tests with code changes** — feature and test in the same commit
3. **Keep commits atomic** — each commit should leave the repo in a working state
4. **Write clear summaries** — explain *what* and *why*, not *how*

---

## Pull Request Process

### Before Submitting

1. **Ensure all tests pass:**

```bash
pytest --cov=utils --cov-fail-under=70
```

2. **Run all quality gates:**

```bash
python scripts/ci_local.py --all
```

3. **Update documentation if needed:**
   - Update README.md for user-facing changes
   - Update docstrings for API changes
   - Add entry to CHANGELOG.md under "Unreleased"

4. **Regenerate notebooks if you modified `utils/`:**

```bash
python scripts/generate_notebooks.py
python scripts/validate_notebooks.py
```

### Submitting a Pull Request

1. **Push your branch:**

```bash
git push origin feature/your-feature-name
```

2. **Create a pull request** on GitHub

3. **Fill out the PR template:**
   - Describe what the PR does
   - Link related issues
   - Mention any breaking changes
   - Include test coverage information

4. **Wait for CI to pass:**
   - All quality gates must pass (lint, tests, build)
   - Address any CI failures

5. **Respond to review feedback:**
   - Make requested changes
   - Push updates to the same branch
   - Mark conversations as resolved

### PR Review Process

Maintainers will review for:
- ✅ Code quality and adherence to style guide
- ✅ Test coverage (≥70% for new code)
- ✅ Clear commit messages
- ✅ Documentation updates
- ✅ CI passing
- ✅ No breaking changes without justification

---

## Additional Resources

- **Documentation:** See [`docs/`](docs/) directory
  - [Repository Guide](docs/repository_guide.md) — codebase navigation
  - [Methodology](docs/methodology.md) — scientific approach
  - [Deployment & Reproducibility](docs/deployment_and_reproducibility.md) — operations guide
- **CI Configuration:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
- **Project README:** [`README.md`](README.md)

---

## Questions?

- **Issues:** Open an issue on GitHub for bugs or feature requests
- **Email:** Contact the maintainers via kaoserahamed@gmail.com
- **Documentation:** Check [`docs/repository_guide.md`](docs/repository_guide.md) first

---

Thank you for contributing! Your work helps advance research on river morphology prediction and climate adaptation.
