"""Tests for the optional, local-first experiment tracking helpers."""

from __future__ import annotations

import importlib.util
import sys

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
