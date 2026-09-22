"""Validate every generated notebook: valid nbformat JSON, parseable code
cells, self-contained (no model_utils imports, no leftover %% tokens), and
every called builder/helper is defined in the same notebook.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


def collect_notebooks(root: str | Path = ".") -> tuple[list[str], list[str]]:
    """Return ``(all_notebooks, generated_notebooks)`` under *root*."""
    root = Path(root)
    patterns = [
        "models/**/*.ipynb",
        "analysis/*.ipynb",
        "data_collection/*.ipynb",
        "preprocessing/*.ipynb",
        "results/*.ipynb",
    ]
    notebooks = sorted(str(p) for pat in patterns for p in root.glob(pat) if p.is_file())
    # Notebooks we generate (self-contained) -- exclude hand-written legacy ones.
    generated = [nb for nb in notebooks if "01_gap_filling_comparison" not in nb]
    return notebooks, generated


def validate_notebook(nb_path: str | Path) -> list[str]:
    """Validate one notebook file; return a list of error strings."""
    errors: list[str] = []
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    if "cells" not in nb or "nbformat" not in nb:
        return [f"{nb_path}: not valid nbformat"]

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    all_code = "\n".join("".join(c["source"]) for c in code_cells)

    # 1. no template placeholders left
    if "%%" in all_code:
        errors.append(f"{nb_path}: leftover %% template token")

    # 2. fully self-contained
    if re.search(r"^\s*(from|import)\s+(utils\.model_utils|model_utils)", all_code, re.M):
        errors.append(f"{nb_path}: still imports model_utils")

    # 3. every code cell parses
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        # strip IPython magics / shell escapes so ast can parse
        cleaned = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith(("%", "!"))
        )
        if not cleaned.strip():
            continue
        try:
            ast.parse(cleaned)
        except SyntaxError as exc:
            errors.append(f"{nb_path} cell {i}: SyntaxError {exc}")

    # 4. self-containment: called names must be defined in the same notebook
    defined = set()
    called = set()
    for cell in code_cells:
        src = "".join(cell["source"])
        cleaned = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith(("%", "!"))
        )
        try:
            tree = ast.parse(cleaned)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                called.add(node.id)

    # Only check names that look like our own helpers/builders
    interesting = {
        n
        for n in called - defined
        if n.startswith(("build_", "create_", "prepare_", "evaluate_", "seed_", "plot_"))
    }
    if interesting:
        errors.append(f"{nb_path}: undefined referenced helpers {sorted(interesting)}")
    return errors


def validate_all(root: str | Path = ".") -> tuple[int, list[str]]:
    """Validate every generated notebook; return ``(count, errors)``."""
    _, generated = collect_notebooks(root)
    errors: list[str] = []
    for nb_path in generated:
        errors.extend(validate_notebook(nb_path))
    return len(generated), errors


def main() -> int:
    count, errors = validate_all(".")
    print(f"Validated {count} generated notebooks.")
    if errors:
        print("\nISSUES:")
        for e in errors:
            print("  -", e)
        return 1
    print("All notebooks OK (valid JSON, parseable, self-contained).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
