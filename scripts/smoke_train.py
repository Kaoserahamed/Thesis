"""Run a deterministic one-epoch ConvLSTM smoke training check."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.model_architectures import build_convlstm  # noqa: E402
from utils.model_utils import seed_everything  # noqa: E402

SEED = 42
EXPECTED_LOSS = 1.5668581724


def run_smoke() -> float:
    """Train ConvLSTM once on fixed synthetic data and return the loss."""
    seed_everything(SEED)
    rng = np.random.RandomState(SEED)
    features = rng.rand(2, 2, 16, 16, 1).astype(np.float32)
    labels = (features[:, -1] > 0.5).astype(np.float32)
    model = build_convlstm(seq_len=2, input_shape=(16, 16, 1))
    history = model.fit(features, labels, epochs=1, batch_size=2, shuffle=False, verbose=0)
    return float(history.history["loss"][0])


def main() -> int:
    loss = run_smoke()
    if EXPECTED_LOSS is not None and not np.isclose(loss, EXPECTED_LOSS, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"deterministic loss changed: expected {EXPECTED_LOSS}, got {loss}")
    print(f"smoke_train_loss={loss:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
