"""Optional MLflow tracking helpers for reproducible model experiments.

Tracking is opt-in. By default, MLflow stores runs in ``./mlruns`` using a
local file backend, so experiments can be recorded without an account or a
remote tracking server. Set ``MLFLOW_TRACKING_URI`` to use a shared server.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, cast

import numpy as np
import pandas as pd

DEFAULT_EXPERIMENT = "river-morphology"
DEFAULT_TRACKING_URI = "file:./mlruns"


def _mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking requires the optional 'mlflow' dependency. "
            "Install requirements.txt to enable experiment tracking."
        ) from exc
    return mlflow


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if is_dataclass(value):
            value = asdict(cast(Any, value))
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, name))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flattened[name] = value
        else:
            flattened[name] = str(value)
    return flattened


def start_run(
    experiment_name: str | None = None,
    run_name: str | None = None,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
):
    """Start an MLflow run using local storage unless configured otherwise."""
    mlflow = _mlflow()
    mlflow.set_tracking_uri(_tracking_uri())
    mlflow.set_experiment(
        experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    )
    run = mlflow.start_run(run_name=run_name, tags=dict(tags or {}))
    if params:
        mlflow.log_params(_flatten(params))
    return run


@contextmanager
def tracked_run(
    experiment_name: str | None = None,
    run_name: str | None = None,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
) -> Iterator[Any]:
    """Run a block with automatic MLflow success/failure finalisation."""
    mlflow = _mlflow()
    start_run(experiment_name, run_name, params, tags)
    try:
        yield mlflow.active_run()
    except Exception:
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.end_run(status="FINISHED")


def log_config(config: Any, prefix: str = "config") -> None:
    """Log an ExperimentConfig or mapping as flattened MLflow parameters."""
    values = asdict(cast(Any, config)) if is_dataclass(config) else config
    if not isinstance(values, Mapping):
        raise TypeError("config must be a dataclass or mapping")
    _mlflow().log_params(_flatten(values, prefix))


def log_metrics(metrics: Mapping[str, float], step: int | None = None) -> None:
    """Log numeric evaluation metrics for the active run."""
    numeric = {str(name): float(value) for name, value in metrics.items()}
    _mlflow().log_metrics(numeric, step=step)


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    """Attach an existing result file to the active run."""
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {artifact}")
    _mlflow().log_artifact(str(artifact), artifact_path=artifact_path)


def enable_keras_autolog(log_models: bool = False) -> None:
    """Enable MLflow's Keras callback logging for the next ``model.fit``."""
    mlflow = _mlflow()
    mlflow.tensorflow.autolog(log_models=log_models, silent=True)


def write_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_ids: Iterable[Any],
    output_path: str | Path,
    threshold: float = 0.5,
) -> Path:
    """Write per-sample IoU, Dice, and pixel-error diagnostics to CSV."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    identifiers = list(sample_ids)
    if true.shape != pred.shape or true.ndim < 2:
        raise ValueError("y_true and y_pred must have matching batch-shaped arrays")
    if len(identifiers) != len(true):
        raise ValueError("sample_ids must match the prediction batch length")

    rows = []
    for identifier, target, prediction in zip(identifiers, true, pred):
        target_mask = np.asarray(target) > threshold
        prediction_mask = np.asarray(prediction) > threshold
        intersection = np.logical_and(target_mask, prediction_mask).sum()
        union = np.logical_or(target_mask, prediction_mask).sum()
        target_count = target_mask.sum()
        prediction_count = prediction_mask.sum()
        rows.append(
            {
                "sample_id": identifier,
                "iou": float(intersection / union) if union else 1.0,
                "dice": (
                    float(2 * intersection / (target_count + prediction_count))
                    if target_count + prediction_count
                    else 1.0
                ),
                "false_positive_pixels": int(np.logical_and(~target_mask, prediction_mask).sum()),
                "false_negative_pixels": int(np.logical_and(target_mask, ~prediction_mask).sum()),
                "absolute_pixel_error": int(np.not_equal(target_mask, prediction_mask).sum()),
            }
        )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def fit_and_track(
    model: Any,
    train_data: Any,
    train_labels: Any,
    *,
    config: Any,
    run_name: str,
    validation_data: Any = None,
    callbacks: Iterable[Any] = (),
    checkpoint_path: str | Path | None = None,
    error_analysis: tuple[np.ndarray, np.ndarray, Iterable[Any], str | Path] | None = None,
    enable_autolog: bool = True,
    **fit_kwargs: Any,
) -> Any:
    """Train one model and log config, history, checkpoint, and error analysis."""
    with tracked_run(run_name=run_name, params={"tracking.schema_version": 1}):
        log_config(config)
        if enable_autolog:
            enable_keras_autolog()
        fit_args = dict(fit_kwargs)
        if validation_data is not None:
            fit_args["validation_data"] = validation_data
        history = model.fit(train_data, train_labels, callbacks=list(callbacks), **fit_args)

        for metric, values in getattr(history, "history", {}).items():
            if values:
                log_metrics({f"final.{metric}": values[-1]})
        if checkpoint_path is not None and Path(checkpoint_path).is_file():
            log_artifact(checkpoint_path, artifact_path="checkpoints")
        if error_analysis is not None:
            analysis_path = write_error_analysis(*error_analysis)
            log_artifact(analysis_path, artifact_path="error_analysis")
        return history


__all__ = [
    "DEFAULT_EXPERIMENT",
    "DEFAULT_TRACKING_URI",
    "enable_keras_autolog",
    "log_artifact",
    "log_config",
    "log_metrics",
    "fit_and_track",
    "start_run",
    "tracked_run",
    "write_error_analysis",
]
