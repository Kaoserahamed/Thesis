"""Fast smoke tests for the five model architectures using tiny spatial shapes.

Unlike :mod:`tests.test_model_builders` (which is marked ``slow`` and runs at the
historical 32×32 shape), these tests build every architecture at **16×16, seq_len=2**
and perform a single forward pass.  The goal is to give the CI fast lane genuine
visibility into the model-building code path that is the largest source of missing
coverage today.

They are marked ``requires_tensorflow`` (TF must be present) but **not** ``slow``:
at 16×16 / seq_len=2 each model builds and runs a single ``predict`` in well under a
second on CPU, so the whole module adds only a few seconds to the fast lane.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_tensorflow

pytest.importorskip("tensorflow", reason="TensorFlow not installed")

# ---------------------------------------------------------------------------
# Builders under test
# ---------------------------------------------------------------------------

from utils.model_architectures import (  # noqa: E402
    AttentionGate,
    MODEL_REGISTRY,
    build_attention_unet_convlstm,
    build_convlstm,
    build_swin_st,
    build_unet_lstm,
    build_vit_st,
)

SEQ_LEN = 2
SMALL_H = 16
SMALL_W = 16
SMALL_C = 1
BATCH = 1

ALL_NAMES = [
    "convlstm",
    "unet_lstm",
    "attention_unet_convlstm",
    "swin_st",
    "vit_st",
]


# ---------------------------------------------------------------------------
# Registry / build_model (dispatcher)
# ---------------------------------------------------------------------------


class TestRegistry:
    EXPECTED = set(ALL_NAMES)

    def test_registry_has_five_entries(self):
        assert set(MODEL_REGISTRY.keys()) == self.EXPECTED

    def test_each_entry_is_callable(self):
        for name, fn in MODEL_REGISTRY.items():
            assert callable(fn), f"{name} is not callable"

    def test_build(self):
        from utils.model_architectures import build_model

        for name in ALL_NAMES:
            model = build_model(name, seq_len=SEQ_LEN)
            assert model is not None
            # Every model should be compiled
            assert model.optimizer is not None

    def test_build_model_unknown_raises(self):
        from utils.model_architectures import build_model

        with pytest.raises(ValueError, match="Unknown model"):
            build_model("not_a_real_model", seq_len=2)


# ---------------------------------------------------------------------------
# Per-architecture build + single forward pass at tiny shape
# ---------------------------------------------------------------------------


class TestConvLSTM:
    def test_build(self):
        model = build_convlstm(SEQ_LEN, input_shape=(SMALL_H, SMALL_W, SMALL_C))
        assert model is not None
        assert model.name == f"ConvLSTM_seq{SEQ_LEN}"

    def test_forward_pass(self):
        model = build_convlstm(SEQ_LEN, input_shape=(SMALL_H, SMALL_W, SMALL_C))
        x = (
            np.random.RandomState(0)
            .rand(BATCH, SEQ_LEN, SMALL_H, SMALL_W, SMALL_C)
            .astype(np.float32)
        )
        y = model.predict(x, verbose=0)
        assert y.shape == (BATCH, SMALL_H, SMALL_W, SMALL_C)


class TestUNetLSTM:
    def test_build(self):
        model = build_unet_lstm(SEQ_LEN, input_shape=(SMALL_H, SMALL_W, SMALL_C))
        assert model is not None

    def test_forward_pass(self):
        model = build_unet_lstm(SEQ_LEN, input_shape=(SMALL_H, SMALL_W, SMALL_C))
        x = (
            np.random.RandomState(0)
            .rand(BATCH, SEQ_LEN, SMALL_H, SMALL_W, SMALL_C)
            .astype(np.float32)
        )
        y = model.predict(x, verbose=0)
        assert y.shape == (BATCH, SMALL_H, SMALL_W, SMALL_C)


class TestAttentionUNetConvLSTM:
    def test_build(self):
        model = build_attention_unet_convlstm(SEQ_LEN, input_shape=(SMALL_H, SMALL_W, SMALL_C))
        assert model is not None
        assert AttentionGate is not None

    def test_forward_pass(self):
        model = build_attention_unet_convlstm(SEQ_LEN, input_shape=(SMALL_H, SMALL_W, SMALL_C))
        x = (
            np.random.RandomState(0)
            .rand(BATCH, SEQ_LEN, SMALL_H, SMALL_W, SMALL_C)
            .astype(np.float32)
        )
        y = model.predict(x, verbose=0)
        assert y.shape == (BATCH, SMALL_H, SMALL_W, SMALL_C)


class TestSwinST:
    def test_build(self):
        model = build_swin_st(SEQ_LEN, input_shape=(SMALL_H, SMALL_W, SMALL_C))
        assert model is not None

    def test_forward_pass(self):
        model = build_swin_st(SEQ_LEN, input_shape=(SMALL_H, SMALL_W, SMALL_C))
        x = (
            np.random.RandomState(0)
            .rand(BATCH, SEQ_LEN, SMALL_H, SMALL_W, SMALL_C)
            .astype(np.float32)
        )
        y = model.predict(x, verbose=0)
        assert y.shape == (BATCH, SMALL_H, SMALL_W, SMALL_C)


class TestViTST:
    def test_build(self):
        model = build_vit_st(SEQ_LEN, input_shape=(SMALL_H, SMALL_W, SMALL_C))
        assert model is not None

    def test_forward_pass(self):
        model = build_vit_st(SEQ_LEN, input_shape=(SMALL_H, SMALL_W, SMALL_C))
        x = (
            np.random.RandomState(0)
            .rand(BATCH, SEQ_LEN, SMALL_H, SMALL_W, SMALL_C)
            .astype(np.float32)
        )
        y = model.predict(x, verbose=0)
        assert y.shape == (BATCH, SMALL_H, SMALL_W, SMALL_C)
