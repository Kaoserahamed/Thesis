from __future__ import annotations

from argparse import Namespace

import numpy as np

from scripts import run_experiment


class _FakeModel:
    def predict(self, values, verbose=0):
        return np.zeros_like(values[:, 0:1])


def test_run_uses_synthetic_stack_and_writes_outputs(tmp_path, monkeypatch):
    captured = {}

    def fake_fit(model, train_data, train_labels, **kwargs):
        captured["train_shape"] = train_data.shape
        captured["config"] = kwargs["config"]

    monkeypatch.setattr(run_experiment, "build_model", lambda *args, **kwargs: _FakeModel())
    monkeypatch.setattr(run_experiment, "fit_and_track", fake_fit)
    monkeypatch.setattr(run_experiment, "create_callbacks", lambda *args, **kwargs: [])

    output_dir = run_experiment.run(
        Namespace(
            resolution="yearly",
            architecture="convlstm",
            epochs=1,
            seed=42,
            data_dir=tmp_path / "missing-data",
            output_dir=tmp_path / "outputs",
        )
    )

    assert output_dir == tmp_path / "outputs" / "yearly"
    assert (output_dir / "checkpoints").is_dir()
    assert (output_dir / "predictions.npy").is_file()
    assert captured["train_shape"][1:] == (4, 256, 256, 1)
    assert captured["config"]["seed"] == 42