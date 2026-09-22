"""Notebook regression gate wired into the pytest coverage report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_notebooks import validate_all  # noqa: E402


def test_generated_notebooks_validate() -> None:
    """Every generated notebook passes scripts/validate_notebooks.py."""
    count, errors = validate_all(ROOT)
    assert count > 0, "no generated notebooks found under repo root"
    assert not errors, "notebook validation failed:\n" + "\n".join(errors)


def test_forecast_notebooks_inline_forecast_helpers() -> None:
    """Prediction notebooks carry the extracted forecast helpers."""
    import json

    stems = {
        "02_long_term_prediction.ipynb",
        "03_short_term_prediction.ipynb",
        "04_quarterly_prediction.ipynb",
    }
    checked = 0
    for path in sorted((ROOT / "analysis").glob("*.ipynb")):
        if path.name not in stems:
            continue
        code = "\n".join(
            "".join(c["source"])
            for c in json.loads(path.read_text(encoding="utf-8"))["cells"]
            if c["cell_type"] == "code"
        )
        assert "def frames_to_cube" in code, f"{path.name} missing forecast helpers"
        assert "def make_forecast_sequences" in code
        assert "%%" not in code, f"{path.name} has leftover template tokens"
        checked += 1
    assert checked == 3, f"expected 3 prediction notebooks, found {checked}"
