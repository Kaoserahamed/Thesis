"""Tests for the notebook generator (scripts/notebookgen/templates.py)."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from notebookgen.extract import code, md  # noqa: E402
from notebookgen.metadata import ARCHITECTURES  # noqa: E402
from notebookgen.templates import (  # noqa: E402
    LOAD_DATA,
    PRELUDE,
    architecture_code,
    callbacks_code,
    shared_pipeline_code,
)
from notebookgen.writers import forecast_helpers_code  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "golden_notebook.ipynb"

RES_KEY = "yearly"
ARCH_KEY = "convlstm"


def _fill_prelude() -> str:
    """Fill PRELUDE exactly as writers.model_notebook does."""
    text = PRELUDE
    for token, value in (
        ("%%RES%%", RES_KEY),
        ("%%LOWER%%", "yearly"),
        ("%%PERIOD%%", "year"),
        ("%%ENVVAR%%", "YEARLY_DIR"),
        ("%%SEQLENS%%", repr([4, 5, 6])),
        ("%%LABEL%%", ARCHITECTURES[ARCH_KEY]["label"]),
        ("%%KEY%%", ARCH_KEY),
        ("%%BUILDER%%", ARCHITECTURES[ARCH_KEY]["builder"]),
        ("%%SECTION%%", ARCHITECTURES[ARCH_KEY]["section"]),
        ("%%FIGURE%%", ARCHITECTURES[ARCH_KEY]["figure"]),
    ):
        text = text.replace(token, value)
    text = text.replace("%%CODE%%", architecture_code(ARCH_KEY))
    text = text.replace("%%CALLBACKS%%", callbacks_code())
    return text


def build_golden_cells() -> list:
    """Assemble golden notebook cells from current template output."""
    load = (
        LOAD_DATA.replace("%%LOWER%%", "yearly")
        .replace("%%PERIOD%%", "year")
        .replace("%%RES%%", RES_KEY)
        .replace("%%ENVVAR%%", "YEARLY_DIR")
    )
    return [
        md("# Golden fixture - notebookgen templates"),
        md("## 1. Configuration and imports"),
        code(_fill_prelude()),
        md("## 2. Shared pipeline"),
        code(shared_pipeline_code()),
        md("## 3. Architecture (inlined)"),
        code(architecture_code(ARCH_KEY)),
        md("## 4. Callbacks + forecast helpers"),
        code(callbacks_code() + "\n\n" + forecast_helpers_code()),
        md("## 5. Data loading snippet"),
        code(load),
    ]


def build_golden_notebook() -> dict:
    return {
        "cells": build_golden_cells(),
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


class TestTemplateContents:
    def test_shared_pipeline_inlines_expected_helpers(self) -> None:
        body = shared_pipeline_code()
        for name in [
            "dice_coefficient",
            "combined_loss",
            "iou_np",
            "calculate_area_difference",
            "keep_largest_n_components_cv2",
            "load_image_stack",
            "create_sequences",
            "prepare_split",
            "evaluate_model",
            "summarise_results",
        ]:
            assert f"def {name}" in body

    def test_architecture_code_inlines_builder_and_helpers(self) -> None:
        body = architecture_code(ARCH_KEY)
        assert "def build_convlstm" in body
        attn = architecture_code("attention_unet_convlstm")
        assert "def build_attention_unet_convlstm" in attn
        assert "class AttentionGate" in attn
        assert "def _conv_bn" in attn

    def test_callbacks_code_inlines_logger(self) -> None:
        body = callbacks_code()
        assert "class StructuredTrainingLogger" in body
        assert "def create_callbacks" in body

    def test_forecast_helpers_inline_utils(self) -> None:
        body = forecast_helpers_code()
        assert "def frames_to_cube" in body
        assert "def make_forecast_sequences" in body

    def test_no_template_tokens_survive_fill(self) -> None:
        assert "%%" not in _fill_prelude()

    def test_filled_prelude_parses(self) -> None:
        cleaned = "\n".join(
            line for line in _fill_prelude().splitlines() if not line.strip().startswith(("%", "!"))
        )
        ast.parse(cleaned)


class TestGoldenFixture:
    def test_fixture_exists(self) -> None:
        assert FIXTURE.is_file(), "golden fixture missing"

    def test_golden_notebook_matches_fixture(self) -> None:
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = build_golden_notebook()
        assert expected["nbformat"] == actual["nbformat"] == 4
        assert len(expected["cells"]) == len(actual["cells"])
        for i, (exp_cell, act_cell) in enumerate(zip(expected["cells"], actual["cells"])):
            assert exp_cell["cell_type"] == act_cell["cell_type"]
            assert "".join(exp_cell["source"]) == "".join(
                act_cell["source"]
            ), f"cell {i} drifted; regenerate fixture if intentional"

    def test_fixture_cells_are_parseable(self) -> None:
        nb = json.loads(FIXTURE.read_text(encoding="utf-8"))
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            cleaned = "\n".join(
                line for line in src.splitlines() if not line.strip().startswith(("%", "!"))
            )
            if cleaned.strip():
                compile(cleaned, f"<golden cell {i}>", "exec")
