"""Run one reproducible river-morphology experiment from the command line.

Example::

    python scripts/run_experiment.py --preset yearly_setup1 \
        --model attention_unet_convlstm --seq-len 4

The command reads the preset and data directory from the environment, seeds
NumPy/TensorFlow, trains through the MLflow-backed tracking pipeline, and
writes evaluation and per-sample error-analysis artifacts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from utils.experiment_tracking import (
    fit_and_track,
    log_artifact,
    log_config,
    log_metrics,
    tracked_run,
    write_error_analysis,
)
from utils.model_utils import (
    EXPERIMENT_PRESETS,
    build_model,
    create_callbacks,
    evaluate_model,
    get_pixel_area_km2,
    load_image_stack,
    prepare_split,
    seed_everything,
)
from utils.pipeline_utils import build_catalog, create_sequences

_DATA_ENV = {"yearly": "YEARLY_DIR", "quarterly": "QUARTERLY_DIR", "bimonthly": "BIMONTHLY_DIR"}
_MODEL_NAMES = ("convlstm", "unet_lstm", "attention_unet_convlstm", "swin_st", "vit_st")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(EXPERIMENT_PRESETS), required=True)
    parser.add_argument("--model", choices=_MODEL_NAMES, required=True)
    parser.add_argument(
        "--seq-len", type=int, help="Sequence length; defaults to the preset first value."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/runs"))
    return parser


def run(args: argparse.Namespace) -> Path:
    config = EXPERIMENT_PRESETS[args.preset]
    if args.seq_len is None:
        seq_len = config.sequence_lengths[0]
    elif args.seq_len in config.sequence_lengths:
        seq_len = args.seq_len
    else:
        raise ValueError(f"seq_len must be one of {config.sequence_lengths} for {args.preset}")

    seed_everything(args.seed)
    data_dir = Path(os.environ.get(_DATA_ENV[config.resolution], f"data/raw/{config.resolution}"))
    images, years = load_image_stack(str(data_dir))
    X_all, y_all, input_years, target_years = create_sequences(images, years, seq_len)
    X_tr, y_tr, X_val, y_val, X_test, y_test, test_years = prepare_split(
        X_all, y_all, target_years, input_years, config.cutoff_year
    )

    reference = build_catalog(str(data_dir)).iloc[0]["filepath"]
    pixel_area = get_pixel_area_km2(reference)
    run_name = f"{config.resolution}-{args.model}-L{seq_len}-{args.preset}-seed{args.seed}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / f"{run_name}_best.keras"

    model = build_model(args.model, seq_len=seq_len)
    fit_and_track(
        model,
        X_tr,
        y_tr,
        config={"preset": args.preset, "model": args.model, "seed": args.seed, **config.__dict__},
        run_name=run_name,
        validation_data=(X_val, y_val),
        callbacks=create_callbacks(run_name, checkpoint_dir=str(run_dir), epochs=config.epochs),
        checkpoint_path=checkpoint,
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=0,
    )

    predictions = model.predict(X_test, verbose=0)
    with tracked_run(run_name=f"{run_name}-evaluation", params={"phase": "evaluation"}):
        log_config(config)
        results = evaluate_model(
            model, X_test, y_test, test_years, pixel_area, args.model, args.preset
        )
        summary = results.groupby("Model")[["IoU", "Dice", "Precision", "Recall"]].mean()
        summary_path = run_dir / "metrics.csv"
        results.to_csv(summary_path, index=False)
        error_path = write_error_analysis(
            y_test, predictions, test_years, run_dir / "error_analysis.csv"
        )
        log_metrics({f"test.{key.lower()}": value for key, value in summary.iloc[0].items()})
        log_artifact(summary_path, artifact_path="evaluation")
        log_artifact(error_path, artifact_path="evaluation")

    print(f"Completed {run_name}")
    print(f"Artifacts: {run_dir}")
    return run_dir


def main() -> int:
    args = _parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
