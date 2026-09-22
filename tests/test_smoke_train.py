"""Executable reproducibility check for the documented smoke training path."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = pytest.mark.requires_tensorflow


def test_smoke_train_script_reproduces_loss():
    result = subprocess.run(
        [sys.executable, "scripts/smoke_train.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "smoke_train_loss=1.5668581724" in result.stdout
