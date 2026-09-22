"""Tests for ablation/error-analysis summary artifacts.

Covers ``analysis/05_ablation_summary.py`` and the committed
``results/ablation_summary.csv``: README anchors, sequence-length and
loss-component ablations, error-breakdown identities, and CSV regeneration.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "analysis" / "05_ablation_summary.py"
CSV_PATH = ROOT / "results" / "ablation_summary.csv"

sys.path.insert(0, str(ROOT / "analysis"))
loader = pytest.importorskip("importlib.util", reason="importlib unavailable")
spec = loader.spec_from_file_location("ablation_summary_mod", str(SCRIPT))
assert spec is not None and spec.loader is not None
ablation = loader.module_from_spec(spec)
spec.loader.exec_module(ablation)


REQUIRED_COLUMNS = [
    "resolution",
    "setup",
    "ablation",
    "variant",
    "iou",
    "dice",
    "false_positive_rate",
    "false_negative_rate",
    "mean_instability",
    "source",
]


def test_readme_baselines_match_key_results():
    """README anchors must equal the committed Key Results IoU/Dice values."""
    parsed = ablation.parse_readme_baselines()
    assert parsed["yearly"]["iou"] == pytest.approx(0.7005)
    assert parsed["yearly"]["dice"] == pytest.approx(0.8236)
    assert parsed["quarterly"]["iou"] == pytest.approx(0.6956)
    assert parsed["bimonthly"]["dice"] == pytest.approx(0.8701)


def test_statistical_notebook_outputs_present():
    """Statistical notebook must retain areal/erosion/migration/WOF outputs."""
    required = ablation.check_statistical_notebook()
    assert all(required.values())


def test_error_breakdown_partitions_union():
    """TP/FP/FN rates must partition the water-pixel union."""
    truth = np.zeros((8, 8), dtype=np.float32)
    truth[:4, :] = 1.0
    pred = np.zeros((8, 8), dtype=np.float32)
    pred[2:6, :] = 1.0
    breakdown = ablation.error_breakdown(truth, pred)
    total = breakdown["true_positive"] + breakdown["false_positive"]
    total += breakdown["false_negative"]
    assert total == pytest.approx(1.0)
    assert breakdown["false_positive"] > 0.0
    assert breakdown["false_negative"] > 0.0


def test_ablation_table_covers_seq_len_and_loss():
    """Every resolution needs sequence-length rows plus 3 loss variants."""
    table = ablation.build_ablation_table()
    assert list(table.columns) == REQUIRED_COLUMNS
    for resolution, seq_lens in (("yearly", 3), ("quarterly", 3), ("bimonthly", 3)):
        sub = table[table["resolution"] == resolution]
        seq_rows = sub[sub["ablation"] == "sequence_length"]
        loss_rows = sub[sub["ablation"] == "loss_component"]
        assert len(seq_rows) == seq_lens
        assert sorted(loss_rows["variant"]) == ["BCE only", "BCE+Dice", "Dice only"]
        assert ((seq_rows["iou"] > 0.0) & (seq_rows["iou"] <= 1.0)).all()
        assert ((seq_rows["dice"] > 0.0) & (seq_rows["dice"] <= 1.0)).all()


def _seq_len_value(variant: str) -> int:
    """Extract integer sequence length from ``seq_len=<n>`` variants."""
    return int(str(variant).split("=")[-1])


def test_longer_sequences_do_not_degrade_iou():
    """Longer temporal context must not reduce calibrated IoU."""
    table = ablation.build_ablation_table()
    seq_rows = table[table["ablation"] == "sequence_length"]
    for _, sub in seq_rows.groupby("resolution"):
        ordered = sub.sort_values("variant", key=lambda col: col.map(_seq_len_value))
        ious = ordered["iou"].to_numpy()
        assert (np.diff(ious) >= -1e-6).all()


def test_combined_loss_is_best_variant():
    """BCE+Dice must score >= single-term losses per resolution."""
    table = ablation.build_ablation_table()
    loss_rows = table[table["ablation"] == "loss_component"]
    for _, sub in loss_rows.groupby("resolution"):
        best = sub[sub["variant"] == "BCE+Dice"].iloc[0]
        assert (sub["iou"] <= best["iou"] + 1e-9).all()
        assert (sub["dice"] <= best["dice"] + 1e-9).all()


def test_committed_csv_matches_generator(tmp_path):
    """Committed CSV must equal fresh generator output byte-for-byte."""
    assert CSV_PATH.is_file(), "results/ablation_summary.csv is not committed"
    committed = CSV_PATH.read_text(encoding="utf-8")
    fresh = tmp_path / "fresh.csv"
    ablation.main(["--output", str(fresh)])
    assert fresh.read_text(encoding="utf-8") == committed


def test_script_regenerates_csv_subprocess(tmp_path):
    """Script entry point regenerates the CSV with expected columns."""
    out = tmp_path / "ablation_summary.csv"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    regenerated = pd.read_csv(out)
    assert list(regenerated.columns) == REQUIRED_COLUMNS
    assert len(regenerated) == 18
