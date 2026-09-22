# Contributing

Thank you for helping improve the river-morphology forecasting repository. Keep
changes focused, reproducible, and consistent with the existing utilities and
self-contained notebook workflow.

## Development setup

Install the development dependencies from a clean checkout:

```bash
python -m pip install -r requirements-dev.txt
```

Run the fast local test lane before opening a pull request:

```bash
pytest -m "not slow"
```

## Quality checks

The CI lint job runs these commands from the repository root. Run them locally
when changing Python code:

```bash
black --check --line-length 100 utils tests scripts
flake8 utils tests scripts
mypy utils tests scripts
```

Notebook changes should also pass:

```bash
python scripts/validate_notebooks.py
```

## Adding an architecture

1. Add a builder function to `utils/model_architectures.py`. Match the existing
   builder contract: accept `seq_len` and an optional `(height, width, channels)`
   `input_shape`, return a compiled Keras model, and produce a single-channel
   prediction with the repository's shared loss and metrics.
2. Add the builder to `MODEL_REGISTRY` using a stable, lowercase registry key.
   The key is the public name used by `build_model()` and the experiment runner.
3. Add the key to the expected architecture list and add a focused build and
   output-shape test in `tests/test_model_builders.py`. The test should verify
   that the model builds, is compiled, and returns one output channel. Add a
   small forward-pass check when the architecture has custom shape logic.
4. Run the fast tests and quality checks above. For TensorFlow architecture
   changes, also run the model lane:

```bash
pytest -m "requires_tensorflow"
```

5. Update the relevant model documentation and regenerate affected notebooks
   with `python scripts/generate_notebooks.py` when the architecture is part of
   the notebook experiment catalogue.

## Pull requests

Describe the purpose of the change, the commands used for validation, and any
data or environment variables required to reproduce the result. Do not commit
raw datasets, credentials, model checkpoints, or local MLflow output.