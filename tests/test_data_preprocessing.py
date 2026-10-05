"""Tests for data preprocessing and validation functions."""

import pytest
import numpy as np

from utils.data_utils import validate_image_data, normalize_image_data


class TestValidateImageData:
    """Test image data validation."""
    
    def test_valid_2d_image(self):
        """Valid 2D image should pass."""
        data = np.random.rand(256, 256).astype(np.float32)
        validate_image_data(data)  # Should not raise
    
    def test_valid_3d_image_with_channels(self):
        """Valid 3D image with channels should pass."""
        data = np.random.rand(256, 256, 3).astype(np.float32)
        validate_image_data(data)  # Should not raise
    
    def test_1d_array_rejected(self):
        """1D array should be rejected."""
        data = np.random.rand(256)
        with pytest.raises(ValueError, match="2D.*or 3D"):
            validate_image_data(data)
    
    def test_4d_array_rejected(self):
        """4D array should be rejected."""
        data = np.random.rand(10, 256, 256, 3)
        with pytest.raises(ValueError, match="2D.*or 3D"):
            validate_image_data(data)
    
    def test_shape_validation(self):
        """Expected shape should be enforced."""
        data = np.random.rand(256, 256)
        
        # Matching shape should pass
        validate_image_data(data, expected_shape=(256, 256))
        
        # Mismatching shape should fail
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_image_data(data, expected_shape=(128, 128))
    
    def test_dtype_validation(self):
        """Expected dtype should be enforced."""
        data = np.random.rand(256, 256).astype(np.float32)
        
        # Matching dtype should pass
        validate_image_data(data, expected_dtype=np.float32)
        
        # Mismatching dtype should fail
        with pytest.raises(ValueError, match="Data type mismatch"):
            validate_image_data(data, expected_dtype=np.uint8)
    
    def test_nan_detection(self):
        """NaN values should be detected."""
        data = np.array([[1.0, 2.0], [np.nan, 4.0]])
        
        with pytest.raises(ValueError, match="NaN values"):
            validate_image_data(data, allow_nan=False)
    
    def test_nan_allowed(self):
        """NaN values should be allowed when specified."""
        data = np.array([[1.0, 2.0], [np.nan, 4.0]])
        validate_image_data(data, allow_nan=True)  # Should not raise
    
    def test_inf_detection(self):
        """Infinite values should be detected."""
        data = np.array([[1.0, 2.0], [np.inf, 4.0]])
        
        with pytest.raises(ValueError, match="infinite values"):
            validate_image_data(data, check_finite=True)
    
    def test_negative_inf_detection(self):
        """Negative infinite values should be detected."""
        data = np.array([[1.0, 2.0], [-np.inf, 4.0]])
        
        with pytest.raises(ValueError, match="infinite values"):
            validate_image_data(data)
    
    def test_value_range_validation(self):
        """Value range should be enforced."""
        data = np.array([[0.5, 0.7], [0.3, 0.9]])
        
        # Within range should pass
        validate_image_data(data, value_range=(0.0, 1.0))
        
        # Out of range should fail
        data_out = np.array([[0.5, 1.5], [0.3, 0.9]])  # 1.5 > 1.0
        with pytest.raises(ValueError, match="Values out of range"):
            validate_image_data(data_out, value_range=(0.0, 1.0))
    
    def test_non_numpy_array_rejected(self):
        """Non-numpy array should be rejected."""
        with pytest.raises(TypeError, match="Expected numpy array"):
            validate_image_data([[1, 2], [3, 4]])
    
    def test_typical_geotiff_data(self):
        """Should validate typical GeoTIFF data."""
        # Simulated binary water mask (uint8)
        data = np.random.choice([0, 1], size=(512, 512), p=[0.7, 0.3]).astype(np.uint8)
        
        validate_image_data(
            data,
            expected_shape=(512, 512),
            expected_dtype=np.uint8,
            value_range=(0, 1)
        )
    
    def test_typical_normalized_data(self):
        """Should validate typical normalized data."""
        # Simulated normalized continuous mask (float32)
        data = np.random.rand(256, 256).astype(np.float32)
        
        validate_image_data(
            data,
            expected_shape=(256, 256),
            expected_dtype=np.float32,
            value_range=(0.0, 1.0)
        )


class TestNormalizeImageData:
    """Test image data normalization."""
    
    def test_uint8_to_01_range(self):
        """Should normalize uint8 to [0, 1]."""
        data = np.array([[0, 127], [255, 64]], dtype=np.uint8)
        normalized = normalize_image_data(data, target_range=(0, 1))
        
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)
        assert normalized.dtype == np.float32
    
    def test_preserves_relative_values(self):
        """Normalization should preserve relative ordering."""
        data = np.array([[10, 20], [30, 40]], dtype=np.float32)
        normalized = normalize_image_data(data)
        
        # Check ordering is preserved
        assert normalized[0, 0] < normalized[0, 1] < normalized[1, 0] < normalized[1, 1]
    
    def test_custom_target_range(self):
        """Should normalize to custom target range."""
        data = np.array([[0, 50], [100, 150]], dtype=np.float32)
        normalized = normalize_image_data(data, target_range=(-1, 1))
        
        assert normalized.min() == pytest.approx(-1.0)
        assert normalized.max() == pytest.approx(1.0)
    
    def test_constant_image(self):
        """Constant image should map to middle of target range."""
        data = np.full((64, 64), 42.0)
        normalized = normalize_image_data(data, target_range=(0, 1))
        
        # Should all be 0.5 (middle of range)
        assert np.all(normalized == pytest.approx(0.5))
    
    def test_clip_outliers(self):
        """Should clip outliers before normalization."""
        data = np.array([[1, 2, 3], [4, 5, 100]], dtype=np.float32)  # 100 is outlier
        
        normalized = normalize_image_data(
            data,
            clip_outliers=True,
            percentile_range=(10, 90)
        )
        
        # After clipping, should not have extreme values
        assert normalized.max() <= 1.0
        assert normalized.min() >= 0.0
    
    def test_large_image(self):
        """Should handle large images efficiently."""
        data = np.random.rand(1024, 1024).astype(np.float32)
        normalized = normalize_image_data(data)
        
        assert normalized.shape == (1024, 1024)
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0
    
    def test_non_numpy_array_rejected(self):
        """Non-numpy array should be rejected."""
        with pytest.raises(TypeError, match="Expected numpy array"):
            normalize_image_data([[1, 2], [3, 4]])
    
    def test_does_not_modify_original(self):
        """Should not modify original array."""
        data = np.array([[10, 20], [30, 40]], dtype=np.float32)
        data_copy = data.copy()
        
        normalized = normalize_image_data(data)
        
        np.testing.assert_array_equal(data, data_copy)
        assert normalized is not data
    
    def test_negative_values(self):
        """Should handle negative values."""
        data = np.array([[-10, -5], [0, 5]], dtype=np.float32)
        normalized = normalize_image_data(data, target_range=(0, 1))
        
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)
    
    def test_very_small_range(self):
        """Should handle data with very small range."""
        data = np.array([[1.0, 1.0000001], [1.0000002, 1.0000003]])
        normalized = normalize_image_data(data)
        
        # Should produce valid normalized data
        assert 0 <= normalized.min() <= 1
        assert 0 <= normalized.max() <= 1


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_pixel_image(self):
        """Single pixel should be handled."""
        data = np.array([[42]])
        
        validate_image_data(data)
        normalized = normalize_image_data(data)
        assert normalized.shape == (1, 1)
    
    def test_zero_image(self):
        """All-zero image should be handled."""
        data = np.zeros((64, 64), dtype=np.float32)
        
        validate_image_data(data, value_range=(0, 1))
        normalized = normalize_image_data(data)
        # All zeros should stay at min of target range
        assert np.all(normalized == pytest.approx(0.5))  # Middle of [0,1]
    
    def test_ones_image(self):
        """All-ones image should be handled."""
        data = np.ones((64, 64), dtype=np.float32)
        
        validate_image_data(data, value_range=(0, 1))
        normalized = normalize_image_data(data)
        # All ones should map to middle (constant image)
        assert np.all(normalized == pytest.approx(0.5))
    
    def test_binary_mask(self):
        """Binary mask should validate and normalize correctly."""
        data = np.random.choice([0, 1], size=(100, 100))
        
        validate_image_data(data, value_range=(0, 1))
        normalized = normalize_image_data(data)
        
        # Should only have 0 and 1
        unique_values = np.unique(normalized)
        assert len(unique_values) == 2
        assert 0.0 in unique_values
        assert 1.0 in unique_values


class TestIntegrationScenarios:
    """Integration tests for realistic preprocessing workflows."""
    
    def test_geotiff_loading_pipeline(self):
        """Simulate GeoTIFF loading and validation."""
        # Simulate loaded uint8 data from GeoTIFF
        raw_data = np.random.randint(0, 2, size=(512, 512), dtype=np.uint8)
        
        # Validate raw data
        validate_image_data(
            raw_data,
            expected_shape=(512, 512),
            expected_dtype=np.uint8,
            value_range=(0, 1)
        )
        
        # Convert to float32 for model input
        float_data = raw_data.astype(np.float32)
        
        # Normalize to [0, 1]
        normalized = normalize_image_data(float_data, target_range=(0, 1))
        
        # Validate normalized data
        validate_image_data(
            normalized,
            expected_shape=(512, 512),
            expected_dtype=np.float32,
            value_range=(0, 1)
        )
        
        assert normalized.shape == (512, 512)
        assert 0 <= normalized.min() <= 1
        assert 0 <= normalized.max() <= 1
    
    def test_batch_preprocessing(self):
        """Simulate preprocessing a batch of images."""
        batch_size = 10
        images = []
        
        for i in range(batch_size):
            # Simulate varied input data
            data = np.random.randint(0, 256, size=(128, 128), dtype=np.uint8)
            
            # Validate
            validate_image_data(
                data,
                expected_shape=(128, 128),
                expected_dtype=np.uint8
            )
            
            # Normalize
            normalized = normalize_image_data(data, target_range=(0, 1))
            images.append(normalized)
        
        # Stack into batch
        batch = np.stack(images, axis=0)
        assert batch.shape == (batch_size, 128, 128)
        assert 0 <= batch.min() <= 1
        assert 0 <= batch.max() <= 1
    
    def test_error_recovery_workflow(self):
        """Test validation error detection and handling."""
        # Simulate corrupted data with NaN
        corrupted_data = np.random.rand(256, 256)
        corrupted_data[50:60, 50:60] = np.nan
        
        # Should detect corruption
        with pytest.raises(ValueError, match="NaN"):
            validate_image_data(corrupted_data, allow_nan=False)
        
        # Can proceed if NaN is allowed (for debugging)
        validate_image_data(corrupted_data, allow_nan=True)
    
    def test_outlier_handling_workflow(self):
        """Test outlier detection and normalization."""
        # Create data with outliers
        data = np.random.rand(200, 200) * 0.8  # Most values in [0, 0.8]
        data[10:15, 10:15] = 10.0  # Outliers
        
        # Detect outliers via range check
        with pytest.raises(ValueError, match="Values out of range"):
            validate_image_data(data, value_range=(0, 1))
        
        # Normalize with outlier clipping
        normalized = normalize_image_data(
            data,
            clip_outliers=True,
            percentile_range=(1, 99)
        )
        
        # After clipping and normalizing, should be in valid range
        validate_image_data(normalized, value_range=(0, 1))
    
    def test_multi_channel_preprocessing(self):
        """Test preprocessing multi-channel images."""
        # Simulate RGB-like data
        rgb_data = np.random.rand(256, 256, 3).astype(np.float32)
        
        # Validate 3-channel image
        validate_image_data(
            rgb_data,
            expected_shape=(256, 256, 3),
            expected_dtype=np.float32,
            value_range=(0, 1)
        )
        
        # Normalize (works on multi-channel)
        normalized = normalize_image_data(rgb_data)
        
        assert normalized.shape == (256, 256, 3)
        validate_image_data(normalized, value_range=(0, 1))
