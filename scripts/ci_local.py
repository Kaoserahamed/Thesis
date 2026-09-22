"""Run the same quality gates as the GitHub Actions pipeline, locally.

Cross-platform wrapper so contributors can reproduce the CI result before
pushing (the workflow itself uses raw ``black``/``flake8``/``mypy``/``pytest``
commands; this script just sequences them and reports one summary).

Usage
-----
    python scripts/ci_local.py            # lint + notebooks + tests (fast lane)
    python scripts/ci_local.py --all      # include the TensorFlow model lane
    python scripts/ci_local.py --build    # also build & verify the wheel
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

PY = sys.executable


def _run(label: str, cmd: list[str]) -> bool:
    print(f"\n=== {label} ===")
    print("$ " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f">>> FAILED ({result.returncode}): {label}")
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CI quality gates locally.")
    parser.add_argument("--all", action="store_true", help="also run the TensorFlow model lane")
    parser.add_argument("--build", action="store_true", help="also build and verify the wheel")
    args = parser.parse_args()

    # Run pytest via the current interpreter; the other tools are console scripts.
    pytest_cmd = [PY, "-m", "pytest"]
    black_cmd = [sys.executable, "-m", "black"] if shutil.which("black") is None else ["black"]
    flake8_cmd = [sys.executable, "-m", "flake8"] if shutil.which("flake8") is None else ["flake8"]
    mypy_cmd = [sys.executable, "-m", "mypy"] if shutil.which("mypy") is None else ["mypy"]

    checks: list[tuple[str, list[str]]] = [
        (
            "black --check",
            [*black_cmd, "--check", "--line-length", "100", "utils", "tests", "scripts"],
        ),
        ("flake8", [*flake8_cmd, "utils", "tests", "scripts"]),
        ("mypy", [*mypy_cmd, "utils", "tests", "scripts"]),
        ("validate notebooks", [PY, "scripts/validate_notebooks.py"]),
        ("health self-probe", [PY, "scripts/serve_health.py", "--once"]),
    ]

    if args.build:
        checks += [
            ("build sdist + wheel", [PY, "-m", "build", "--outdir", "dist"]),
            ("verify wheel", [PY, "scripts/verify_dist.py", "dist"]),
        ]

    # `slow` marks model construction; the workflow's fast lane deselects it.
    fast_args = [*pytest_cmd, "-m", "not slow"]
    if not args.all:
        fast_args.extend(
            [
                "--ignore",
                "tests/test_model_builders.py",
                "--cov=utils",
                "--cov-report=term-missing",
                "--cov-fail-under=50",
            ]
        )
    checks.append(("pytest (fast lane)", fast_args))
    if args.all:
        checks.append(("pytest (model lane)", [*pytest_cmd, "-m", "requires_tensorflow"]))

    failed = [label for label, cmd in checks if not _run(label, cmd)]

    print("\n" + "=" * 60)
    if failed:
        print("SOME GATES FAILED:")
        for label in failed:
            print(f"  - {label}")
        return 1
    print(f"ALL {len(checks)} GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
