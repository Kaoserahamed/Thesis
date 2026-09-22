"""Tests for ``utils.pipeline_utils`` -- the TensorFlow-free half of the data
pipeline (GeoTIFF preprocessing, sequence generation, temporal split, presets)
plus the package-level logging configuration.

Everything here runs with **no TensorFlow, no Google Earth Engine account and
no external data**: synthetic GeoTIFFs are written into ``tmp_path`` with
rasterio and read back, so the fast CI lane exercises the real I/O code paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from utils.pipeline_utils import (
    BINARY_THRESHOLD,
    IMG_HEIGHT,
    IMG_WIDTH,
    N_COMPONENTS,
    build_catalog,
    get_pixel_area_km2,
    keep_largest_n_components_cv2,
    load_and_preprocess_image,
    load_image_stack,
)

# ---------------------------------------------------------------------------
# GeoTIFF helpers
# ---------------------------------------------------------------------------

GEOGRAPHIC_CRS = "EPSG:4326"  # lat/lon
PROJECTED_CRS = "EPSG:32645"  # UTM zone 45N (Bangladesh -> metric units)


def _write_tif(path, array, crs=GEOGRAPHIC_CRS, pixel_size=0.001, origin=(89.0, 24.0)):
    """Write *array* to a single-band GeoTIFF so the pipeline can read it back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = array.shape
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=from_origin(origin[0], origin[1], pixel_size, pixel_size),
        nodata=0,
        compress="lzw",
    ) as dst:
        dst.write(array.astype("uint8"), 1)
    return str(path)


def _water_mask(height=512, width=512, block=128):
    """A deterministic binary water mask with a single large blob."""
    mask = np.zeros((height, width), dtype="uint8")
    mask[block : block * 2, block : block * 2] = 1
    return mask


# ---------------------------------------------------------------------------
# keep_largest_n_components_cv2
# ---------------------------------------------------------------------------


class TestKeepLargestComponents:
    def test_keeps_only_n_largest(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[0:40, 0:40] = 1.0  # 1600 px  (largest)
        mask[50:80, 50:80] = 1.0  # 900 px
        mask[85:95, 85:95] = 1.0  # 100 px
        mask[20:25, 60:70] = 1.0  # 50 px   (smallest -> dropped)

        cleaned = keep_largest_n_components_cv2(mask, n=3)

        assert cleaned.shape == mask.shape
        assert cleaned[0:40, 0:40].min() == 1.0  # largest kept
        assert cleaned[50:80, 50:80].min() == 1.0  # second kept
        assert cleaned[85:95, 85:95].min() == 1.0  # third kept
        assert cleaned[20:25, 60:70].max() == 0.0  # smallest removed

    def test_empty_mask_is_safe(self):
        mask = np.zeros((16, 16), dtype=np.float32)
        cleaned = keep_largest_n_components_cv2(mask, n=N_COMPONENTS)
        assert cleaned.shape == (16, 16)
        assert cleaned.max() == 0.0

    def test_n_larger_than_component_count(self):
        mask = np.zeros((16, 16), dtype=np.float32)
        mask[2:6, 2:6] = 1.0
        cleaned = keep_largest_n_components_cv2(mask, n=99)
        assert cleaned.sum() == mask.sum()


# ---------------------------------------------------------------------------
# load_and_preprocess_image
# ---------------------------------------------------------------------------


class TestLoadAndPreprocessImage:
    def test_output_shape_and_dtype(self, tmp_path):
        path = _write_tif(tmp_path / "frame_2010.tif", _water_mask())
        out = load_and_preprocess_image(path)

        assert out.shape == (IMG_HEIGHT, IMG_WIDTH)
        assert out.dtype == np.float32
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_cleaning_can_be_disabled(self, tmp_path):
        mask = np.zeros((512, 512), dtype="uint8")
        mask[0:200, 0:200] = 1
        mask[400:410, 400:410] = 1  # isolated speckle
        path = _write_tif(tmp_path / "speckled.tif", mask)

        cleaned = load_and_preprocess_image(path, apply_cleaning=True, n_components=1)
        raw = load_and_preprocess_image(path, apply_cleaning=False)

        # Cleaning can only remove pixels, never add them.
        assert cleaned.sum() <= raw.sum()

    def test_all_land_frame_is_all_zero(self, tmp_path):
        path = _write_tif(tmp_path / "empty.tif", np.zeros((512, 512), dtype="uint8"))
        out = load_and_preprocess_image(path)
        assert out.max() == 0.0
        assert BINARY_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# get_pixel_area_km2
# ---------------------------------------------------------------------------


class TestGetPixelAreaKm2:
    def test_projected_crs_matches_30m_landsat(self, tmp_path):
        """30 m x 30 m pixels -> 900 m2 = 0.0009 km2 per 256x256 frame pixel."""
        path = _write_tif(
            tmp_path / "utm.tif",
            np.zeros((256, 256), dtype="uint8"),
            crs=PROJECTED_CRS,
            pixel_size=30.0,
            origin=(500_000.0, 2_700_000.0),
        )
        assert get_pixel_area_km2(path) == pytest.approx(0.0009, rel=1e-6)

    def test_geographic_crs_is_positive_and_plausible(self, tmp_path):
        path = _write_tif(tmp_path / "geo.tif", np.zeros((256, 256), dtype="uint8"))
        area = get_pixel_area_km2(path)
        # A 256x256 Landsat scene on a degree grid is well under 1 km2/pixel.
        assert 0.0 < area < 1.0

    def test_geographic_crs_uses_latitude_scaling(self, tmp_path):
        """Pixels further north must cover less ground (cos(lat) shrinking)."""
        equator = _write_tif(
            tmp_path / "eq.tif", np.zeros((256, 256), dtype="uint8"), origin=(0.0, 0.0)
        )
        northern = _write_tif(
            tmp_path / "north.tif", np.zeros((256, 256), dtype="uint8"), origin=(0.0, 60.0)
        )
        assert get_pixel_area_km2(northern) < get_pixel_area_km2(equator)


# ---------------------------------------------------------------------------
# build_catalog
# ---------------------------------------------------------------------------


class TestBuildCatalog:
    def test_sorted_by_year_and_columns(self, tmp_path):
        _write_tif(tmp_path / "Padma_2012.tif", _water_mask())
        _write_tif(tmp_path / "Padma_2010.tif", _water_mask())
        _write_tif(tmp_path / "Padma_2011.tif", _water_mask())

        df = build_catalog(str(tmp_path))

        assert list(df["year"]) == [2010, 2011, 2012]
        assert {"filepath", "filename", "year"}.issubset(set(df.columns))
        assert df.iloc[0]["filename"] == "Padma_2010.tif"

    def test_files_without_year_are_skipped(self, tmp_path):
        _write_tif(tmp_path / "Padma_2015.tif", _water_mask())
        _write_tif(tmp_path / "mask_no_year.tif", _water_mask())

        df = build_catalog(str(tmp_path))

        assert len(df) == 1
        assert df.iloc[0]["year"] == 2015

    def test_empty_directory_returns_empty_frame_with_schema(self, tmp_path):
        """The column schema must survive an empty scan so callers can index it."""
        df = build_catalog(str(tmp_path))
        assert df.empty
        assert list(df.columns) == ["filepath", "filename", "year"]
        assert df["year"].tolist() == []  # must not raise KeyError


# ---------------------------------------------------------------------------
# load_image_stack  (end-to-end, no network / no GEE)
# ---------------------------------------------------------------------------


class TestLoadImageStack:
    def test_loads_every_frame_in_year_order(self, tmp_path):
        for year in (2019, 2017, 2018):
            _write_tif(tmp_path / f"river_{year}.tif", _water_mask())

        images, years = load_image_stack(str(tmp_path))

        assert years == [2017, 2018, 2019]
        assert len(images) == 3
        for frame in images:
            assert frame.shape == (IMG_HEIGHT, IMG_WIDTH)
        np.testing.assert_array_equal(images[0], images[1])

    def test_missing_directory_raises(self, tmp_path):
        """A wrong path must fail loudly, not return an empty stack."""
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_image_stack(str(missing))

    def test_directory_without_year_filenames_raises(self, tmp_path):
        _write_tif(tmp_path / "no_year_here.tif", _water_mask())
        with pytest.raises(FileNotFoundError, match="No year-tagged GeoTIFFs"):
            load_image_stack(str(tmp_path))

    def test_empty_directory_raises_rather_than_returning_empty(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image_stack(str(tmp_path))


# ---------------------------------------------------------------------------
# Package-level logging configuration
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_installs_single_stream_handler(self):
        from utils import configure_logging

        package_logger = logging.getLogger("utils")
        configure_logging(logging.INFO)
        configure_logging(logging.INFO)  # idempotent

        streams = [
            h
            for h in package_logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        ]
        assert len(streams) == 1
        assert package_logger.level == logging.INFO

    def test_save_figure_logs_instead_of_printing(self, tmp_path, caplog):
        """The structured-logging migration: one INFO record, no stdout."""
        import matplotlib.pyplot as plt

        from utils.visualization_utils import save_figure

        fig, _ = plt.subplots()
        with caplog.at_level(logging.INFO, logger="utils.visualization_utils"):
            save_figure(fig, tmp_path / "nested" / "figure.png")

        assert (tmp_path / "nested" / "figure.png").exists()
        assert any(
            "figure.png" in record.message and record.levelno == logging.INFO
            for record in caplog.records
        )
