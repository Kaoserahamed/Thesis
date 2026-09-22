# Security Notes

This project processes caller-supplied filesystem paths, filename patterns,
configuration files, and numeric sequence controls. Validation is enforced at
the reusable utility boundary before any file or array operation.

## Input Validation

- `utils.data_utils.validate_path` rejects empty or non-path values.
- `utils.data_utils.validate_safe_pattern` rejects absolute paths and `..`
  traversal components from glob patterns.
- YAML configuration files must use `.yaml`/`.yml` extensions and contain a
  top-level mapping loaded with `yaml.safe_load`.
- GeoTIFF readers and writers require `.tif`/`.tiff` paths; writers require a
  two-dimensional NumPy array and required geospatial metadata.
- Temporal selectors must be sequences of integers.
- Sequence length, horizon, and stride controls must be positive integers, and
  image/year sequences must have matching lengths.

Invalid input raises a clear `TypeError`, `ValueError`, or `FileNotFoundError`
before the underlying I/O or sequence operation begins. Regression tests for
these cases live in `tests/test_data_utils.py` and
`tests/test_pipeline_utils.py`.

## Secrets and Dependencies

No credentials are committed. Local secrets belong outside Git; `.env` and
credential files are ignored. CI installs the committed dependency lockfile
and runs `pip-audit` as part of the dependency checks.

Error records can be delivered to an external monitoring service by setting
`THESIS_ERROR_WEBHOOK_URL`. The webhook is opt-in, uses a short timeout, and
delivery failures are logged without interrupting the pipeline.