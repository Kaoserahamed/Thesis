# Dependency Management

The repository uses `requirements-dev.txt` as the human-maintained dependency
input and `requirements.lock` as the committed, exact-version installation
lockfile for Python 3.11 CI and local development.

## Install

```bash
python -m pip install -r requirements.lock
```

Do not install the range manifests in CI. `requirements.txt` remains the
runtime dependency contract, while `requirements-dev.txt` adds test, lint,
type-checking, and packaging tools.

## Refresh

After changing `requirements.txt` or `requirements-dev.txt`, regenerate the
lockfile with the pinned resolver tooling:

```bash
python -m pip install "pip-tools==7.6.1"
pip-compile --output-file=requirements.lock requirements-dev.txt
```

Review the complete transitive diff, run `pip check`, and run the fast test
lane before committing. Dependabot may update the input manifests; the CI
lock-consistency check ensures the committed lockfile is refreshed before a
change can merge.

## CI Contract

CI performs these dependency steps:

1. Install exactly from `requirements.lock`.
2. Run `pip check`.
3. Re-resolve the manifest without upgrading pinned packages and fail if the
	committed lockfile changes.
4. Run tests, coverage, packaging, and dependency audit checks.

The lockfile is intentionally committed because the project includes large
scientific and geospatial dependency graphs where a fresh range resolution can
otherwise change the environment without a source-code change.
