"""Verify the built distribution artifacts.

Run after ``python -m build`` to assert that the wheel contains every shared
library module and that each module is syntactically valid Python.  This needs
no project dependencies installed (it byte-compiles instead of importing), so
it is cheap enough to run in CI:

    python scripts/verify_dist.py [dist_dir]
"""

from __future__ import annotations

import glob
import os
import py_compile
import sys
import tempfile
import zipfile

EXPECTED_MODULES = {
    "utils/__init__.py",
    "utils/model_utils.py",
    "utils/data_utils.py",
    "utils/visualization_utils.py",
}


def find_wheel(dist_dir: str) -> str:
    wheels = sorted(glob.glob(os.path.join(dist_dir, "*.whl")))
    if not wheels:
        raise SystemExit(f"No wheel found in '{dist_dir}' -- run `python -m build` first.")
    return wheels[0]


def verify(dist_dir: str = "dist") -> None:
    wheel = find_wheel(dist_dir)
    with zipfile.ZipFile(wheel) as zf:
        modules = [n for n in zf.namelist() if n.startswith("utils/") and n.endswith(".py")]

    missing = EXPECTED_MODULES - set(modules)
    if missing:
        raise SystemExit(f"Wheel {wheel} is missing modules: {sorted(missing)}")

    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(tmp)
        for module in modules:
            py_compile.compile(os.path.join(tmp, module), doraise=True)

    print(f"OK: {os.path.basename(wheel)}")
    print(f"    modules : {sorted(modules)}")
    print("    all modules byte-compile cleanly")


if __name__ == "__main__":
    verify(sys.argv[1] if len(sys.argv) > 1 else "dist")
