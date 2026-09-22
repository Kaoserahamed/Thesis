"""Tests for data-utility functions (data I/O, normalisation, water masks).
These tests rely only on numpy and rasterio -- no TensorFlow needed.
"""

from __future__ import annotations
import numpy as np
import pytest

from utils.data_utils import compute_statistics, split_temporal_data, normalize_array
from utils.data_utils import create_water_mask, extract_year_from_filename
from utils.data_utils import calculate_water_area, load_config, load_geotiff, load_temporal_sequence, save_geotiff


class TestComputeStatistics:
    def test_basic_stats(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        stats = compute_statistics(arr)
        assert stats["mean"] == pytest.approx(2.5)
        assert stats["std"] == pytest.approx(np.std(arr))
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0
        assert stats["median"] == pytest.approx(2.5)
        assert stats["count"] == 4
        assert stats["nan_count"] == 0

    def test_with_nan(self):
        arr = np.array([[1.0, np.nan], [3.0, 4.0]])
        stats = compute_statistics(arr)
        assert stats["nan_count"] == 1
        assert stats["count"] == 3
        assert stats["min"] == 1.0
        assert stats["max"] == 4.0

    def test_empty_array_raises(self):
        arr = np.array([], dtype=np.float64).reshape(0, 0)
        with pytest.raises(ValueError):
            compute_statistics(arr)


class TestSplitTemporalData:
    def test_split_sizes(self):
        seq = np.arange(100, dtype=np.float32)
        years = list(range(2000, 2100))
        splits = split_temporal_data(seq, years, 0.7, 0.15, 0.15)
        assert len(splits["train"]["data"]) == 70
        assert len(splits["val"]["data"]) == 15
        assert len(splits["test"]["data"]) == 15

    def test_years_split_correctly(self):
        seq = np.arange(10, dtype=np.float32)
        years = list(range(2000, 2010))
        splits = split_temporal_data(seq, years)
        assert splits["train"]["years"] == [2000, 2001, 2002, 2003, 2004, 2005, 2006]
        assert splits["test"]["years"] == [2008, 2009]

    def test_invalid_ratios_raise(self):
        seq = np.arange(10, dtype=np.float32)
        years = list(range(2000, 2010))
        with pytest.raises(AssertionError, match="Ratios must sum to 1.0"):
            split_temporal_data(seq, years, 0.5, 0.2, 0.5)

    def test_chronological_order_preserved(self):
        seq = np.arange(20, dtype=np.float32)
        years = list(range(2000, 2020))
        splits = split_temporal_data(seq, years)
        assert splits["train"]["years"][-1] < splits["val"]["years"][0]
        assert splits["val"]["years"][-1] < splits["test"]["years"][0]


class TestNormalizeArray:
    def test_minmax_range(self):
        arr = np.array([0.0, 5.0, 10.0])
        norm = normalize_array(arr, method="minmax")
        assert norm.min() == pytest.approx(0.0)
        assert norm.max() == pytest.approx(1.0)

    def test_minmax_custom_range(self):
        arr = np.array([0.0, 5.0, 10.0])
        norm = normalize_array(arr, method="minmax", feature_range=(-1, 1))
        assert norm.min() == pytest.approx(-1.0)
        assert norm.max() == pytest.approx(1.0)

    def test_zscore(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norm = normalize_array(arr, method="zscore")
        assert norm.mean() == pytest.approx(0.0, abs=1e-6)
        assert norm.std() == pytest.approx(1.0, abs=1e-6)

    def test_none_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        norm = normalize_array(arr, method="none")
        np.testing.assert_array_equal(norm, arr)

    def test_constant_array_minmax(self):
        arr = np.full((5,), 7.0)
        norm = normalize_array(arr, method="minmax")
        assert not np.any(np.isnan(norm))
        assert np.allclose(norm, 0.0)

    def test_constant_array_zscore(self):
        arr = np.full((5,), 3.0)
        norm = normalize_array(arr, method="zscore")
        np.testing.assert_array_equal(norm, arr - 3.0)

    def test_nan_handling(self):
        arr = np.array([1.0, np.nan, 3.0])
        norm = normalize_array(arr, method="minmax")
        assert norm[0] == pytest.approx(0.0)
        assert norm[2] == pytest.approx(1.0)


class TestCreateWaterMask:
    def test_mndwi_threshold(self):
        arr = np.array([0.3, 0.6, 0.5, 0.7])
        mask = create_water_mask(arr, threshold=0.5, method="mndwi")
        assert mask.tolist() == [0, 1, 0, 1]

    def test_sar_inversion(self):
        arr = np.array([0.1, 0.5, 0.2, 0.8])
        mask = create_water_mask(arr, threshold=0.3, method="sar")
        assert mask.tolist() == [1, 0, 1, 0]

    def test_dtype_uint8(self):
        arr = np.array([0.6, 0.2])
        mask = create_water_mask(arr, threshold=0.5, method="mndwi")
        assert mask.dtype == np.uint8

    def test_unknown_method_raises(self):
        arr = np.array([0.5])
        with pytest.raises(ValueError, match="Unknown water detection method"):
            create_water_mask(arr, threshold=0.5, method="unknown")


class TestExtractYear:
    def test_extract_valid_year(self):
        assert extract_year_from_filename("Landsat_2015_composite.tif") == 2015

    def test_extract_two_thousands(self):
        assert extract_year_from_filename("data_2000_v1.nc") == 2000

    def test_extract_19xx_year(self):
        assert extract_year_from_filename("image_1998_mask.tif") == 1998

    def test_no_year_raises(self):
        with pytest.raises(ValueError, match="Could not extract year"):
            extract_year_from_filename("no_year_here.txt")


class TestCalculateWaterArea:
    def test_basic_area(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[:10, :10] = 1
        area = calculate_water_area(mask, pixel_size_m=60.0)
        assert area == pytest.approx(0.36)

    def test_all_water(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        area = calculate_water_area(mask, pixel_size_m=30.0)
        assert area == pytest.approx(0.09)

    def test_all_land(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        area = calculate_water_area(mask, pixel_size_m=60.0)
        assert area == 0.0


class TestGeoTIFFIO:
    def test_save_load_roundtrip(self, tmp_path):
        from rasterio.transform import from_origin

        arr = np.random.randint(0, 2, size=(64, 64)).astype(np.uint8)
        out_path = str(tmp_path / "test_mask.tif")
        metadata = {
            "transform": from_origin(0, 64, 1, 1),
            "crs": "EPSG:4326",
            "width": 64,
            "height": 64,
            "nodata": 255,
        }
        save_geotiff(arr, out_path, metadata, dtype="uint8")
        assert tmp_path.joinpath("test_mask.tif").exists()
        loaded, meta = load_geotiff(out_path)
        np.testing.assert_array_equal(loaded, arr)
        assert meta["width"] == 64
        assert meta["height"] == 64

    def test_save_creates_parent_dirs(self, tmp_path):
        arr = np.zeros((10, 10), dtype=np.uint8)
        out_path = str(tmp_path / "nested" / "deep" / "mask.tif")
        metadata = {"transform": None, "crs": "EPSG:4326", "width": 10, "height": 10, "nodata": 255}
        save_geotiff(arr, out_path, metadata, dtype="uint8")
        assert tmp_path.joinpath("nested", "deep", "mask.tif").exists()

    def test_rejects_non_geotiff_paths(self, tmp_path):
        with pytest.raises(ValueError, match="GeoTIFF"):
            load_geotiff(tmp_path / "mask.txt")

    def test_save_rejects_non_2d_arrays(self, tmp_path):
        with pytest.raises(ValueError, match="two-dimensional"):
            save_geotiff(np.zeros((2, 2, 1)), tmp_path / "mask.tif", {})


class TestInputValidation:
    def test_config_must_be_yaml_mapping(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        config_path.write_text("- not-a-mapping\n", encoding="utf-8")

        with pytest.raises(ValueError, match="top-level mapping"):
            load_config(config_path)

    def test_temporal_pattern_rejects_parent_traversal(self, tmp_path):
        with pytest.raises(ValueError, match="relative local"):
            load_temporal_sequence(tmp_path, pattern="../*.tif")

    def test_temporal_years_must_be_integers(self, tmp_path):
        with pytest.raises(TypeError, match="sequence of integers"):
            load_temporal_sequence(tmp_path, years=["2020"])
