"""Tests for the optional, local-first experiment tracking helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_tracking_module():
    spec = importlib.util.spec_from_file_location(
        "experiment_tracking_under_test", "utils/experiment_tracking.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeMlflow:
    def __init__(self):
        self.active = None
        self.statuses = []
        self.params = []
        self.metrics = []
        self.artifacts = []
        self.tensorflow = type("Tensorflow", (), {"autolog": lambda *_args, **_kwargs: None})()

    def set_tracking_uri(self, uri):
        self.uri = uri

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, **kwargs):
        self.active = object()
        self.run_kwargs = kwargs
        return self.active

    def active_run(self):
        return self.active

    def log_params(self, params):
        self.params.append(params)

    def end_run(self, status):
        self.statuses.append(status)
        self.active = None

    def log_metrics(self, metrics, step=None):
        self.metrics.append((metrics, step))

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((path, artifact_path))


def test_flatten_preserves_nested_experiment_parameters():
    tracking = _load_tracking_module()

    assert tracking._flatten({"model": {"name": "convlstm", "seed": 42}}) == {
        "model.name": "convlstm",
        "model.seed": 42,
    }


def test_tracked_run_finalizes_successfully(monkeypatch):
    tracking = _load_tracking_module()
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    with tracking.tracked_run(run_name="smoke", params={"seed": 42}):
        pass

    assert fake.run_kwargs == {"run_name": "smoke", "tags": {}}
    assert fake.params == [{"seed": 42}]
    assert fake.statuses == ["FINISHED"]


def test_tracked_run_marks_failures(monkeypatch):
    tracking = _load_tracking_module()
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    with pytest.raises(RuntimeError):
        with tracking.tracked_run(run_name="failed"):
            raise RuntimeError("expected")

    assert fake.statuses == ["FAILED"]


def test_write_error_analysis_creates_per_sample_diagnostics(tmp_path):
    tracking = _load_tracking_module()
    output = tracking.write_error_analysis(
        y_true=[[[1, 0], [0, 1]]],
        y_pred=[[[1, 1], [0, 0]]],
        sample_ids=["year-2020"],
        output_path=tmp_path / "errors.csv",
    )

    assert output == Path(tmp_path / "errors.csv")
    frame = tracking.pd.read_csv(output)
    assert frame.loc[0, "sample_id"] == "year-2020"
    assert frame.loc[0, "false_positive_pixels"] == 1
    assert frame.loc[0, "false_negative_pixels"] == 1


def test_fit_and_track_logs_history_checkpoint_and_analysis(monkeypatch, tmp_path):
    tracking = _load_tracking_module()
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    class FakeHistory:
        history = {"loss": [0.5, 0.25]}

    class FakeModel:
        def fit(self, *_args, **kwargs):
            assert kwargs["validation_data"] is not None
            return FakeHistory()

    checkpoint = tmp_path / "model.keras"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    history = tracking.fit_and_track(
        FakeModel(),
        [[0]],
        [[1]],
        config={"seed": 42},
        run_name="tracked-smoke",
        validation_data=([[0]], [[1]]),
        checkpoint_path=checkpoint,
        error_analysis=(
            [[[1, 0]]],
            [[[1, 1]]],
            ["sample-1"],
            tmp_path / "error-analysis.csv",
        ),
        enable_autolog=False,
    )

    assert history.history["loss"][-1] == 0.25
    assert fake.statuses == ["FINISHED"]
    assert any(path == str(checkpoint) for path, _ in fake.artifacts)
    assert any(path.endswith("error-analysis.csv") for path, _ in fake.artifacts)
