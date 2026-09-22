"""Run one reproducible tracked experiment from the command line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.experiment_tracking import fit_and_track
from utils.model_utils import build_model, create_callbacks, seed_everything
from utils.pipeline_utils import (
    EXPERIMENT_PRESETS,
    create_sequences,
    load_image_stack,
    prepare_split,
)

RESOLUTIONS = tuple(sorted({c.resolution for c in EXPERIMENT_PRESETS.values()}))
ARCHITECTURES = ("convlstm", "unet_lstm", "attention_unet_convlstm", "swin_st", "vit_st")
DATA_ENV = {resolution: f"{resolution.upper()}_DIR" for resolution in RESOLUTIONS}
SEQUENCE_LENGTHS = {
    resolution: next(
        config.sequence_lengths[0]
        for config in EXPERIMENT_PRESETS.values()
        if config.resolution == resolution
    )
    for resolution in RESOLUTIONS
}
CUTOFF_YEARS = {
    resolution: next(
        config.cutoff_year
        for config in EXPERIMENT_PRESETS.values()
        if config.resolution == resolution
    )
    for resolution in RESOLUTIONS
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", choices=RESOLUTIONS, required=True)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def _synthetic_stack(seed: int = 42):
    rng = np.random.RandomState(seed)
    years = list(range(2000, 2020))
    images = [(rng.rand(256, 256) > 0.5).astype(np.float32) for _ in years]
    return images, years


def _load_stack(resolution: str, data_dir: Path | None, seed: int):
    env_default = f"data/raw/{resolution}"
    directory = data_dir or Path(os.environ.get(DATA_ENV[resolution], env_default))
    if directory.is_dir() and any(directory.glob("*.tif")):
        return load_image_stack(str(directory))
    return _synthetic_stack(seed)


def run(args: argparse.Namespace) -> Path:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")

    seed_everything(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    seq_len = SEQUENCE_LENGTHS[args.resolution]
    images, years = _load_stack(args.resolution, args.data_dir, args.seed)
    X_all, y_all, input_years, target_years = create_sequences(images, years, seq_len)
    cutoff = CUTOFF_YEARS[args.resolution]
    X_tr, y_tr, X_val, y_val, X_test, y_test, test_years = prepare_split(
        X_all, y_all, target_years, input_years, cutoff
    )

    output_dir = args.output_dir / args.resolution
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{args.resolution}-{args.architecture}-seed{args.seed}"
    checkpoint = checkpoint_dir / f"{run_name}_best.keras"
    model = build_model(args.architecture, seq_len=seq_len)
    fit_and_track(
        model,
        X_tr,
        y_tr,
        config={
            "resolution": args.resolution,
            "architecture": args.architecture,
            "seed": args.seed,
        },
        run_name=run_name,
        validation_data=(X_val, y_val),
        callbacks=create_callbacks(
            run_name, checkpoint_dir=str(checkpoint_dir), epochs=args.epochs
        ),
        checkpoint_path=checkpoint,
        epochs=args.epochs,
        batch_size=min(4, len(X_tr)),
        verbose=0,
    )
    predictions = model.predict(X_test, verbose=0)
    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "target_years.npy", test_years)
    print(f"Completed {run_name}")
    print(f"Artifacts: {output_dir}")
    return output_dir


def main() -> int:
    run(_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
