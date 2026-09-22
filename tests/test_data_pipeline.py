"""Tests for the shared data-pipeline helpers.

* ``create_sequences`` -- overlapping (X, y) window generation
* ``prepare_split``    -- leakage-proof temporal split (the core reproducibility
  guarantee of the thesis, section 5.3)
* ``ExperimentConfig`` / ``EXPERIMENT_PRESETS`` -- experiment configuration parity
* ``seed_numpy``       -- NumPy half of the reproducibility guarantee

All of the above live in ``utils.pipeline_utils`` and are pure NumPy, so this
module runs in the **fast** CI lane with no TensorFlow installed.  Only the
TensorFlow half of ``seed_everything`` is marked ``requires_tensorflow``.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.pipeline_utils import (
    DEFAULT_EPOCHS,
    EXPERIMENT_PRESETS,
    ExperimentConfig,
    create_sequences,
    prepare_split,
    seed_numpy,
)


class TestCreateSequences:
    def test_shapes(self, synthetic_sequence):
        images, years = synthetic_sequence
        X, y, in_yr, tgt_yr = create_sequences(images, years, seq_len=4, horizon=1)
        assert X.shape == (6, 4, 16, 16)
        assert y.shape == (6, 16, 16)
        assert in_yr.shape == (6, 4)
        assert tgt_yr.shape == (6,)

    def test_target_year_is_next(self, synthetic_sequence):
        images, years = synthetic_sequence
        X, y, in_yr, tgt_yr = create_sequences(images, years, seq_len=4, horizon=1)
        for i in range(len(tgt_yr)):
            assert tgt_yr[i] == in_yr[i][-1] + 1

    def test_stride_reduces_count(self, synthetic_sequence):
        images, years = synthetic_sequence
        X, _, _, _ = create_sequences(images, years, seq_len=4, horizon=1, stride=2)
        assert X.shape[0] == 3

    def test_horizon_skips_frames(self, synthetic_sequence):
        images, years = synthetic_sequence
        X, _, in_yr, tgt_yr = create_sequences(images, years, seq_len=4, horizon=2)
        assert X.shape[0] == 5
        for i in range(len(tgt_yr)):
            assert tgt_yr[i] == in_yr[i][-1] + 2

    def test_horizon_default_is_1(self, synthetic_sequence):
        images, years = synthetic_sequence
        X1, _, _, tgt1 = create_sequences(images, years, seq_len=4)
        X2, _, _, tgt2 = create_sequences(images, years, seq_len=4, horizon=1)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(tgt1, tgt2)

    def test_too_few_frames_returns_empty(self):
        images = np.zeros((3, 8, 8), dtype=np.float32)
        years = [2000, 2001, 2002]
        X, y, in_yr, tgt_yr = create_sequences(images, years, seq_len=4, horizon=1)
        assert len(X) == 0


def _make_split_data(n_samples=20, start_year=2000):
    """Build synthetic X, y and year arrays for split testing."""
    rng = np.random.RandomState(0)
    X = rng.rand(n_samples, 4, 8, 8).astype(np.float32)
    y = rng.randint(0, 2, size=(n_samples, 8, 8)).astype(np.float32)
    target_years = np.array(range(start_year, start_year + n_samples))
    input_years = np.column_stack(
        [target_years - 4, target_years - 3, target_years - 2, target_years - 1]
    )
    return X, y, target_years, input_years


class TestPrepareSplit:
    def test_no_leakage(self):
        """train/test must be strictly separated by cutoff_year."""
        X, y, ty, iy = _make_split_data(n_samples=20, start_year=2000)
        result = prepare_split(X, y, ty, iy, cutoff_year=2010)
        X_tr, y_tr, X_val, y_val, X_test, y_test, ty_test = result
        assert (ty_test > 2010).all()
        train_mask = (ty <= 2010) & (np.array([iy.max() for iy in iy]) <= 2010)
        assert (ty[train_mask] <= 2010).all()
        train_years = set(ty[train_mask].tolist())
        test_years = set(ty_test.tolist())
        assert not (train_years & test_years)

    def test_split_sizes_add_up(self):
        X, y, ty, iy = _make_split_data(n_samples=20, start_year=2000)
        result = prepare_split(X, y, ty, iy, cutoff_year=2010)
        X_tr, y_tr, X_val, y_val, X_test, y_test, ty_test = result
        assert len(X_tr) == 9
        assert len(X_val) == 2
        assert len(X_test) == 9

    def test_channel_axis_added(self):
        """Output arrays must have a trailing channel dimension."""
        X, y, ty, iy = _make_split_data(n_samples=10, start_year=2000)
        X_tr, y_tr, X_val, y_val, X_test, y_test = prepare_split(X, y, ty, iy, cutoff_year=2005)[:6]
        for arr in (X_tr, X_val, X_test):
            assert arr.ndim == X.ndim + 1
            assert arr.shape[-1] == 1
        for arr in (y_tr, y_val, y_test):
            assert arr.ndim == y.ndim + 1
            assert arr.shape[-1] == 1

    def test_empty_split_raises(self):
        """Cutoff before all targets -> ValueError."""
        X, y, ty, iy = _make_split_data(n_samples=20, start_year=2020)
        with pytest.raises(ValueError, match="Empty split"):
            prepare_split(X, y, ty, iy, cutoff_year=2010)

    def test_no_train_samples_raises(self):
        """Cutoff after all targets -> empty train -> ValueError."""
        X, y, ty, iy = _make_split_data(n_samples=20, start_year=2000)
        with pytest.raises(ValueError, match="Empty split"):
            prepare_split(X, y, ty, iy, cutoff_year=2050)

    def test_target_years_returned(self):
        X, y, ty, iy = _make_split_data(n_samples=20, start_year=2000)
        result = prepare_split(X, y, ty, iy, cutoff_year=2010)
        ty_test = result[6]
        assert len(ty_test) == len(result[4])


class TestExperimentPresets:
    def test_all_six_presets_exist(self):
        expected = {
            "yearly_setup1",
            "yearly_setup2",
            "quarterly_setup1",
            "quarterly_setup2",
            "bimonthly_setup1",
            "bimonthly_setup2",
        }
        assert set(EXPERIMENT_PRESETS.keys()) == expected

    def test_yearly_sequence_lengths(self):
        for key in ("yearly_setup1", "yearly_setup2"):
            assert EXPERIMENT_PRESETS[key].sequence_lengths == [4, 5, 6]

    def test_quarterly_sequence_lengths(self):
        for key in ("quarterly_setup1", "quarterly_setup2"):
            assert EXPERIMENT_PRESETS[key].sequence_lengths == [6, 8, 10]

    def test_bimonthly_sequence_lengths(self):
        for key in ("bimonthly_setup1", "bimonthly_setup2"):
            assert EXPERIMENT_PRESETS[key].sequence_lengths == [6, 9, 12]

    def test_setup1_cutoff_is_2015(self):
        for key in ("yearly_setup1", "quarterly_setup1", "bimonthly_setup1"):
            assert EXPERIMENT_PRESETS[key].cutoff_year == 2015

    def test_setup2_cutoff_is_2020(self):
        for key in ("yearly_setup2", "quarterly_setup2", "bimonthly_setup2"):
            assert EXPERIMENT_PRESETS[key].cutoff_year == 2020

    def test_experiment_config_defaults(self):
        cfg = ExperimentConfig(resolution="yearly", sequence_lengths=[4, 5, 6], cutoff_year=2015)
        assert cfg.batch_size == 4
        assert cfg.epochs == DEFAULT_EPOCHS
        assert cfg.test_label == "Test: 2016-2025"


class TestSeedNumpy:
    def test_reproducible_numpy(self):
        """Same seed -> same numpy random output."""
        seed_numpy(seed=123)
        a = np.random.rand(10)
        seed_numpy(seed=123)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        """Different seeds -> different random output."""
        seed_numpy(seed=1)
        a = np.random.rand(100)
        seed_numpy(seed=2)
        b = np.random.rand(100)
        assert not np.array_equal(a, b)


class TestSeedEverything:
    @pytest.mark.requires_tensorflow
    def test_seeds_tensorflow_too(self):
        """``seed_everything`` must delegate to ``seed_numpy`` and TF."""
        tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")
        from utils.model_utils import seed_everything

        seed_everything(seed=7)
        a = np.random.rand(5)
        tf_a = tf.random.uniform((5,)).numpy()

        seed_everything(seed=7)
        np.testing.assert_array_equal(a, np.random.rand(5))
        np.testing.assert_allclose(tf_a, tf.random.uniform((5,)).numpy())
