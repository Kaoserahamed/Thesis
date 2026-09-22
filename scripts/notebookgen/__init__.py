"""Generator package for the self-contained thesis notebooks.

Run ``python scripts/generate_notebooks.py`` (or ``notebookgen.main()``) from
the repository root to rewrite every notebook under ``models/`` and
``analysis/``.
"""

from .writers import main

__all__ = ["main"]
