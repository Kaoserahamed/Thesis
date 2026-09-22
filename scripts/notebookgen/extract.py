"""Source extraction and notebook-cell helpers for the generator.

``grab(name)`` returns the verbatim source of a top-level function or class
from the reviewed modules in ``utils/``, so the notebooks under ``models/`` and
``analysis/`` are always regenerated from a single source of truth.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
UTILS = ROOT / "utils"

# Modules the inlined notebook helpers are copied from, in search order.
SOURCE_MODULES = (
    UTILS / "metrics_numpy.py",
    UTILS / "pipeline_utils.py",
    UTILS / "forecast_utils.py",
    UTILS / "model_losses.py",
    UTILS / "model_architectures.py",
    UTILS / "model_utils.py",
)


def _index(path: Path):
    """Return ``(source, ast_tree, lines)`` for a module on disk."""
    src = path.read_text(encoding="utf-8")
    return src, ast.parse(src), src.splitlines(keepends=True)


def grab(name: str) -> str:
    """Return the verbatim source of a top-level function/class in ``utils/``.

    Searches :data:`SOURCE_MODULES` in order so a helper can be inlined from
    whichever module now owns it, e.g. ``iou_np`` (metrics_numpy),
    ``create_sequences`` (pipeline_utils) or ``build_convlstm``
    (model_architectures).
    """
    for path in SOURCE_MODULES:
        _src, tree, lines = _SOURCES[path.name]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == name:
                    start = node.lineno - 1
                    for dec in node.decorator_list:
                        start = min(start, dec.lineno - 1)
                    end = node.end_lineno
                    return "".join(lines[start:end]).rstrip() + "\n"
    searched = ", ".join(p.name for p in SOURCE_MODULES)
    raise KeyError(f"'{name}' not found in any of: {searched}")


def md(text: str) -> dict:
    body = text.strip("\n") + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": body.splitlines(keepends=True)}


def code(text: str) -> dict:
    body = text.strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": body.splitlines(keepends=True),
    }


def write_notebook(path: Path, cells: list) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")


def _banner(title: str) -> str:
    bar = "# " + "=" * 76
    return bar + "\n# " + title + "\n" + bar + "\n"


_SOURCES = {path.name: _index(path) for path in SOURCE_MODULES}
