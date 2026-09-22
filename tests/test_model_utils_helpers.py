"""Tests for the training / evaluation helpers in ``utils.model_utils``.

The file is split into two groups:

* **Pure-NumPy / pandas** helpers (``summarise_results``) that run in the fast
  lane without TensorFlow.
* **TensorFlow-dependent** helpers (``create_callbacks``, ``StructuredTrainingLogger``,
  ``evaluate_model``, ``configure_gpu``) that are marked ``requires_tensorflow`` but
  **not ``slow``**, so they run in the fast CI lane whenever TF is installed.

The full 32×32 architecture construction tests live in
:mod:`tests.test_model_builders` (marked ``slow``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import logging
import shutil
import tempfile

# ---------------------------------------------------------------------------
# Pure-pandas helper -- runs in the fast lane with no TF required
# ---------------------------------------------------------------------------


def _results_df():
    """A small deterministic per-sample results table (2 setups × 2 models)."""
    return pd.DataFrame(
        [
            {
                "Setup": "A",
                "Model": "X",
                "Year": 2010,
                "IoU": 0.8,
                "Dice": 0.85,
                "Precision": 0.9,
                "Recall": 0.8,
                "Area_Diff_km2": 0.5,
            },
            {
                "Setup": "A",
                "Model": "X",
                "Year": 2011,
                "IoU": 0.8,
                "Dice": 0.85,
                "Precision": 0.9,
                "Recall": 0.8,
                "Area_Diff_km2": -0.3,
            },
            {
                "Setup": "A",
                "Model": "Y",
                "Year": 2010,
                "IoU": 0.6,
                "Dice": 0.7,
                "Precision": 0.6,
                "Recall": 0.8,
                "Area_Diff_km2": -0.1,
            },
            {
                "Setup": "B",
                "Model": "X",
                "Year": 2010,
                "IoU": 0.7,
                "Dice": 0.75,
                "Precision": 0.8,
                "Recall": 0.7,
                "Area_Diff_km2": 0.2,
            },
            {
                "Setup": "B",
                "Model": "Y",
                "Year": 2011,
                "IoU": 0.5,
                "Dice": 0.6,
                "Precision": 0.5,
                "Recall": 0.7,
                "Area_Diff_km2": 0.0,
            },
        ]
    )


class TestSummariseResults:
    def test_returns_one_row_per_group(self):
        from utils.model_utils import summarise_results

        df = _results_df()
        agg = summarise_results(df)
        assert len(agg) == 4  # (A,X), (A,Y), (B,X), (B,Y)

    def test_columns_present(self):
        from utils.model_utils import summarise_results

        agg = summarise_results(_results_df())
        expected_cols = {
            "Setup",
            "Model",
            "IoU",
            "Dice",
            "Precision",
            "Recall",
            "Area_Diff_km2",
            "Abs_Area_Diff_km2",
        }
        assert set(agg.columns) == expected_cols

    def test_means_are_correct(self):
        from utils.model_utils import summarise_results

        agg = summarise_results(_results_df())
        row = agg[(agg.Setup == "A") & (agg.Model == "X")].iloc[0]
        assert row.IoU == pytest.approx(0.8)
        assert row.Dice == pytest.approx(0.85)
        assert row.Precision == pytest.approx(0.9)
        assert row.Recall == pytest.approx(0.8)

    def test_abs_area_diff_is_mean_of_abs(self):
        from utils.model_utils import summarise_results

        agg = summarise_results(_results_df())
        row = agg[(agg.Setup == "A") & (agg.Model == "X")].iloc[0]
        # |0.5| + |-0.3| averaged = (0.5 + 0.3) / 2 = 0.4
        assert row.Abs_Area_Diff_km2 == pytest.approx(0.4)

    def test_rounding_to_4dp(self):
        from utils.model_utils import summarise_results

        df = pd.DataFrame(
            [
                {
                    "Setup": "S",
                    "Model": "M",
                    "IoU": 0.12345678,
                    "Dice": 0.23456789,
                    "Precision": 0.3456789,
                    "Recall": 0.456789,
                    "Area_Diff_km2": 0.12345678,
                }
            ]
        )
        agg = summarise_results(df)
        row = agg.iloc[0]
        assert row.IoU == pytest.approx(0.1235, abs=1e-4)
        assert row.Dice == pytest.approx(0.2346, abs=1e-4)


# ---------------------------------------------------------------------------
# TensorFlow-dependent helpers (requires_tensorflow, not slow)
# ---------------------------------------------------------------------------


pytestmark = pytest.mark.requires_tensorflow

pytest.importorskip("tensorflow", reason="TensorFlow not installed")


def _dummy_model(seq_len=2, h=16, w=16, c=1):
    """Build a tiny Keras model through the registry so we exercise the real
    compile step (loss + metrics) instead of a hand-rolled stub."""
    from utils.model_architectures import build_convlstm

    model = build_convlstm(seq_len, input_shape=(h, w, c))
    model.optimizer.learning_rate.assign(1e-3)
    return model


def _make_temp_dir():
    """Create a throw-away directory for callback checkpoint tests."""
    d = tempfile.mkdtemp()
    return d


class TestCreateCallbacks:
    def test_returns_four_callbacks(self):
        from utils.model_utils import create_callbacks

        d = _make_temp_dir()
        try:
            cbs = create_callbacks("test_model", checkpoint_dir=d, epochs=5)
            assert len(cbs) == 4
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_checkpoint_path_uses_model_name(self):
        from utils.model_utils import create_callbacks

        d = _make_temp_dir()
        try:
            cbs = create_callbacks("acme", checkpoint_dir=d)
            ckpt = [cb for cb in cbs if cb.__class__.__name__ == "ModelCheckpoint"][0]
            assert ckpt.filepath == str(d) + "\\acme_best.keras"
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_defaults_match_constants(self):
        from utils.model_utils import (
            DEFAULT_EPOCHS,
            EARLY_STOP_PATIENCE,
            REDUCE_LR_PATIENCE,
            create_callbacks,
        )

        d = _make_temp_dir()
        try:
            cbs = create_callbacks("m", checkpoint_dir=d, epochs=DEFAULT_EPOCHS)
            early_stop = [cb for cb in cbs if cb.__class__.__name__ == "EarlyStopping"][0]
            reduce_lr = [cb for cb in cbs if cb.__class__.__name__ == "ReduceLROnPlateau"][0]
            assert early_stop.patience == EARLY_STOP_PATIENCE
            assert reduce_lr.patience == REDUCE_LR_PATIENCE
            assert reduce_lr.factor == 0.5
            assert reduce_lr.min_lr == 1e-7
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_logger_is_structured(self):
        from utils.model_utils import create_callbacks

        d = _make_temp_dir()
        try:
            cbs = create_callbacks("m", checkpoint_dir=d, epochs=10)
            logger_cb = [cb for cb in cbs if cb.__class__.__name__ == "StructuredTrainingLogger"][0]
            assert logger_cb.total_epochs == 10
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestStructuredTrainingLogger:
    """Tests that inspect the ``utils.model_utils`` logger via a dedicated
    ``StringIO``-backed ``StreamHandler``.  ``caplog.records`` does not reliably
    surface the INFO messages that ``on_train_begin``/``on_epoch_end``/``on_train_end``
    emit inside callback hooks, so we attach our own handler and detach it in a
    ``finally`` block."""

    @staticmethod
    def _attach_handler():
        import io

        logger = logging.getLogger("utils.model_utils")
        logger.setLevel(logging.INFO)
        buf = io.StringIO()
        h = logging.StreamHandler(buf)
        h.setLevel(logging.INFO)
        logger.addHandler(h)
        return buf

    @staticmethod
    def _detach_handler(buf):
        logger = logging.getLogger("utils.model_utils")
        for h in list(logger.handlers):
            if getattr(h, "stream", None) is buf:
                logger.removeHandler(h)
                break
        logger.setLevel(logging.WARNING)

    def test_on_train_begin_emits_header(self):
        buf = self._attach_handler()
        try:
            from utils.model_utils import StructuredTrainingLogger

            cb = StructuredTrainingLogger(total_epochs=3)
            cb.set_model(_dummy_model(seq_len=2, h=8, w=8))
            cb.on_train_begin()
            text = buf.getvalue()
            assert "Ep" in text
            assert "Loss" in text
            assert "Dice" in text
        finally:
            self._detach_handler(buf)

    def test_on_epoch_end_emits_one_log_line(self):
        buf = self._attach_handler()
        try:
            from utils.model_utils import StructuredTrainingLogger

            cb = StructuredTrainingLogger(total_epochs=5)
            cb.set_model(_dummy_model(seq_len=2, h=8, w=8))
            cb.on_epoch_end(
                0,
                {
                    "loss": 0.5,
                    "val_loss": 0.4,
                    "dice_coefficient": 0.9,
                    "val_dice_coefficient": 0.8,
                    "iou_metric": 0.85,
                    "val_iou_metric": 0.75,
                },
            )
            text = buf.getvalue()
            assert "1/5" in text
            assert "0.5000" in text
            assert "saved" in text
        finally:
            self._detach_handler(buf)

    def test_on_epoch_end_notes_best(self):
        buf = self._attach_handler()
        try:
            from utils.model_utils import StructuredTrainingLogger

            cb = StructuredTrainingLogger(total_epochs=3)
            cb.set_model(_dummy_model(seq_len=2, h=8, w=8))
            cb.on_epoch_end(0, {"val_loss": 0.5})
            cb.on_epoch_end(1, {"val_loss": 0.4})  # better → "saved"
            cb.on_epoch_end(2, {"val_loss": 0.6})  # worse → no note
            lines = [ln for ln in buf.getvalue().splitlines() if "Ep" in ln]
            assert len(lines) == 3
            # The note column is everything after the last "|"
            assert lines[0].rstrip().endswith("saved")
            assert lines[1].rstrip().endswith("saved")
            assert lines[2].rstrip().endswith("|") or lines[2].rstrip().endswith("")
        finally:
            self._detach_handler(buf)

    def test_on_train_end_emits_footer(self):
        buf = self._attach_handler()
        try:
            from utils.model_utils import StructuredTrainingLogger

            cb = StructuredTrainingLogger(total_epochs=3)
            cb.set_model(_dummy_model(seq_len=2, h=8, w=8))
            cb.on_train_begin()
            cb.on_epoch_end(0, {"val_loss": 0.4})
            cb.on_train_end()
            text = buf.getvalue()
            assert "Training finished" in text
            assert "0.4000" in text
        finally:
            self._detach_handler(buf)


class TestEvaluateModel:
    def test_returns_dataframe(self):
        from utils.model_utils import evaluate_model

        model = _dummy_model(seq_len=2, h=8, w=8)
        X = np.random.RandomState(0).rand(2, 2, 8, 8, 1).astype(np.float32)
        y = np.random.RandomState(1).rand(2, 8, 8, 1).astype(np.float32) > 0.5
        y = y.astype(np.float32)
        df = evaluate_model(model, X, y, [2010, 2011], 0.25, "TestModel", "SetupA")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4  # 2 samples × 2 comparisons (model + persistence)


class TestConfigureGpu:
    def test_no_crash_in_cpu_env(self):
        from utils.model_utils import configure_gpu

        configure_gpu()
