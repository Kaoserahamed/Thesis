"""Validate every generated notebook: valid nbformat JSON, parseable code
cells, self-contained (no model_utils imports, no leftover %% tokens), and
every called builder/helper is defined in the same notebook.
"""

import ast
import json
import glob
import sys
import re

NOTEBOOKS = sorted(
    glob.glob("models/**/*.ipynb", recursive=True)
    + glob.glob("analysis/*.ipynb")
    + glob.glob("data_collection/*.ipynb")
    + glob.glob("preprocessing/*.ipynb")
    + glob.glob("results/*.ipynb")
)

# Notebooks we generate (self-contained) -- exclude hand-written legacy ones.
GENERATED = [nb for nb in NOTEBOOKS if "01_gap_filling_comparison" not in nb]

errors = []
for nb_path in GENERATED:
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    assert "cells" in nb and "nbformat" in nb, f"{nb_path}: not valid nbformat"

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

print(f"Validated {len(GENERATED)} generated notebooks.")
if errors:
    print("\nISSUES:")
    for e in errors:
        print("  -", e)
    sys.exit(1)
print("All notebooks OK (valid JSON, parseable, self-contained).")
