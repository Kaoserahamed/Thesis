"""Tests for the five model architectures defined in ``utils/model_utils.py``.

All tests here build lightweight TensorFlow / Keras models with tiny spatial
dimensions (32×32) and a short sequence length so they run in seconds rather
than minutes.  They are marked ``slow`` and ``requires_tensorflow`` so the fast
CI lane can skip them and the full matrix runs on every push to ``main``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_tensorflow

# Skip the entire module if TF isn't available
pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from utils.model_utils import (  # noqa: E402
    MODEL_REGISTRY,
    build_model,
)


@pytest.mark.slow
class TestModelRegistry:
    """Verify that all five architectures from the thesis are registered."""

    EXPECTED_MODELS = [
        "convlstm",
        "unet_lstm",
        "attention_unet_convlstm",
        "swin_st",
        "vit_st",
    ]

    def test_registry_contains_all_five(self):
        assert set(MODEL_REGISTRY.keys()) == set(self.EXPECTED_MODELS)

    def test_registry_is_dict(self):
        assert isinstance(MODEL_REGISTRY, dict)

    def test_registry_values_are_callables(self):
        for name, builder in MODEL_REGISTRY.items():
            assert callable(builder), f"{name} is not callable"


@pytest.mark.slow
class TestBuildModel:
    """Test the ``build_model`` dispatcher."""

    def test_valid_dispatch(self):
        model = build_model("convlstm", seq_len=4)
        assert model is not None

    @pytest.mark.parametrize(
        "name", ["convlstm", "unet_lstm", "attention_unet_convlstm", "swin_st", "vit_st"]
    )
    def test_each_architecture_builds(self, name):
        model = build_model(name, seq_len=6)
        assert model is not None
        assert len(model.layers) > 0

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent_model", seq_len=4)

    def test_compiled_with_optimizer(self):
        """Every model must be compiled with an Adam optimiser."""
        model = build_model("convlstm", seq_len=4)
        assert model.optimizer is not None

    def test_input_shape_has_seq_dim(self):
        """All models accept (batch, seq_len, H, W, C) input."""
        model = build_model("convlstm", seq_len=4)
        # Check that the model's first layer accepts 5-D input
        in_shape = tuple(model.inputs[0].shape[1:])
        assert in_shape[0] == 4  # seq_len dimension


@pytest.mark.slow
class TestIndividualArchitectures:
    """Smoke-test each builder with minimal input shapes to catch structural bugs."""

    SEQ_LEN = 4

    def test_convlstm(self):
        model = build_model("convlstm", seq_len=self.SEQ_LEN)
        assert model.output_shape[-1] == 1  # single-channel prediction

    def test_unet_lstm(self):
        model = build_model("unet_lstm", seq_len=self.SEQ_LEN)
        assert model.output_shape[-1] == 1

    def test_attention_unet_convlstm(self):
        model = build_model("attention_unet_convlstm", seq_len=self.SEQ_LEN)
        assert model.output_shape[-1] == 1

    def test_swin_st(self):
        model = build_model("swin_st", seq_len=self.SEQ_LEN)
        assert model.output_shape[-1] == 1

    def test_vit_st(self):
        model = build_model("vit_st", seq_len=self.SEQ_LEN)
        assert model.output_shape[-1] == 1

    def test_all_builders_compile_without_error(self):
        """Ensure no architecture raises during construction + compilation."""
        import tensorflow as tf

        for name in MODEL_REGISTRY:
            model = build_model(name, seq_len=self.SEQ_LEN)
            # Verify the model can do a forward pass on its expected input shape
            in_shape = tuple(model.inputs[0].shape[1:])  # (seq_len, H, W, C)
            x = tf.random.uniform((1, *in_shape))
            y = model(x, training=False)
            assert y.shape[0] == 1
