"""Fast shape / forward-pass tests for the utils model-architecture library.

Complements :mod:`tests.test_model_architectures_fast` (per-builder smoke tests
at 16×16) and :mod:`tests.test_model_builders` (the ``slow`` 32×32 lane) with
the assertions the notebook-driven M-dimension needs from tests alone:

* every registry entry builds at the historical 32×32 / seq_len=4 shape and
  exposes a 5-D ``(batch, L, H, W, C)`` input with a single-channel output;
* every compiled model runs a single forward pass at a tiny 16×16 / seq_len=2
  shape and emits values in the sigmoid ``[0, 1]`` range with the expected
  output shape.

Marked ``requires_tensorflow`` but **not** ``slow``: each case builds and runs
one ``predict`` on CPU in well under a second, so the whole module stays in
the fast CI lane (``pytest -m 'not slow'``).
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_tensorflow

pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from utils.model_architectures import MODEL_REGISTRY, build_model  # noqa: E402

ALL_NAMES = [
    "convlstm",
    "unet_lstm",
    "attention_unet_convlstm",
    "swin_st",
    "vit_st",
]

HISTORICAL_H = 32
HISTORICAL_W = 32
HISTORICAL_SEQ_LEN = 4
TINY_H = 16
TINY_W = 16
TINY_SEQ_LEN = 2


@pytest.mark.parametrize("name", ALL_NAMES)
def test_registry_entry_builds_historical_shape(name: str) -> None:
    """Each builder accepts the historical 32x32 / seq_len=4 notebook shape."""
    model = build_model(name, seq_len=HISTORICAL_SEQ_LEN)
    assert model is not None
    assert len(model.layers) > 0


@pytest.mark.parametrize("name", ALL_NAMES)
def test_input_is_five_dimensional(name: str) -> None:
    """All models accept (batch, seq_len, H, W, C) input tensors."""
    from utils.pipeline_utils import IMG_HEIGHT, IMG_WIDTH

    model = build_model(name, seq_len=HISTORICAL_SEQ_LEN)
    in_shape = tuple(model.inputs[0].shape[1:])
    assert len(in_shape) == 4
    assert in_shape[0] == HISTORICAL_SEQ_LEN
    assert in_shape[1] == IMG_HEIGHT
    assert in_shape[2] == IMG_WIDTH
    assert in_shape[3] == 1


@pytest.mark.parametrize("name", ALL_NAMES)
def test_output_is_single_channel_mask(name: str) -> None:
    """All models emit single-channel water-mask predictions."""
    model = build_model(name, seq_len=HISTORICAL_SEQ_LEN)
    assert model.output_shape[-1] == 1


@pytest.mark.parametrize("name", ALL_NAMES)
def test_forward_pass_shape_and_range(name: str) -> None:
    """One tiny forward pass yields the expected shape with sigmoid outputs."""
    from typing import Any

    from utils import model_architectures as arch

    builder: Any = getattr(arch, arch.MODEL_REGISTRY[name].__name__)
    tiny = builder(TINY_SEQ_LEN, input_shape=(TINY_H, TINY_W, 1))
    assert tiny is not None
    rng = np.random.RandomState(0)
    x = rng.rand(1, TINY_SEQ_LEN, TINY_H, TINY_W, 1).astype(np.float32)
    y = tiny.predict(x, verbose=0)
    assert y.shape == (1, TINY_H, TINY_W, 1)
    assert float(np.min(y)) >= 0.0
    assert float(np.max(y)) <= 1.0


def test_registry_matches_notebook_architectures() -> None:
    """The registry still covers exactly the five thesis architectures."""
    assert set(MODEL_REGISTRY.keys()) == set(ALL_NAMES)
