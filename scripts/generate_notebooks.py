"""Regenerate every self-contained thesis notebook.

The implementation lives in the :mod:`notebookgen` package next to this file;
run it from the repository root::

    python scripts/generate_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebookgen import main  # noqa: E402

if __name__ == "__main__":
    main()
