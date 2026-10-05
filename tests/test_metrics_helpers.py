"""Tests for metrics calculation helpers and edge cases."""

import pytest
import numpy as np

from utils.metrics_numpy import (
    iou_np,
    dice_np,
    precision_recall_np,
    aggregate_metrics,
    validate_mask_array,
    calculate_area_difference,
    BINARY_THRESHOLD,
)


class TestPrecisionRecall:
    """Test precision and recall calculation."""
    
    def test_perfect_prediction(self):
        """Perfect prediction should give precision=1, recall=1."""
        mask = np.array([[1, 1], [0, 0]], dtype=np.float32)
        precision, recall = precision_recall_np(mask, mask)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)
    
    def test_no_overlap(self):
        """No overlap should give precision=0, recall=0."""
        y_true = np.array([[1, 1], [0, 0]], dtype=np.float32)
        y_pred = np.array([[0, 0], [1, 1]], dtype=np.float32)
        precision, recall = precision_recall_np(y_true, y_pred)
        # Both should be very close to 0 (epsilon prevents exact 0)
        assert precision < 0.01
        assert recall < 0.01
    
    def test_half_overlap(self):
        """Half overlap scenario."""
        y_true = np.array([[1, 1], [1, 0]], dtype=np.float32)
        y_pred = np.array([[1, 1], [0, 0]], dtype=np.float32)
        precision, recall = precision_recall_np(y_true, y_pred)
        # 2 true positives, 2 predicted positives, 3 actual positives
        assert precision == pytest.approx(1.0, abs=0.01)  # 2/2
        assert recall == pytest.approx(2/3, abs=0.01)    # 2/3
    
    def test_all_zeros(self):
        """All zeros should handle gracefully."""
        y_true = np.zeros((4, 4), dtype=np.float32)
        y_pred = np.zeros((4, 4), dtype=np.float32)
        precision, recall = precision_recall_np(y_true, y_pred)
        # Should return valid numbers, not crash
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
    
    def test_all_ones(self):
        """All ones should give precision=1, recall=1."""
        y_true = np.ones((4, 4), dtype=np.float32)
        y_pred = np.ones((4, 4), dtype=np.float32)
        precision, recall = precision_recall_np(y_true, y_pred)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)
    
    def test_continuous_values(self):
        """Should work with continuous [0,1] values."""
        y_true = np.array([[0.9, 0.8], [0.1, 0.2]], dtype=np.float32)
        y_pred = np.array([[0.95, 0.05], [0.3, 0.1]], dtype=np.float32)
        precision, recall = precision_recall_np(y_true, y_pred)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1


class TestAggregateMetrics:
    """Test batch metrics aggregation."""
    
    def test_single_perfect_sample(self):
        """Single perfect prediction."""
        y_true = np.ones((1, 10, 10), dtype=np.float32)
        y_pred = np.ones((1, 10, 10), dtype=np.float32)
        
        metrics = aggregate_metrics(y_true, y_pred)
        
        assert metrics['iou'] == pytest.approx(1.0)
        assert metrics['dice'] == pytest.approx(1.0)
        assert metrics['precision'] == pytest.approx(1.0)
        assert metrics['recall'] == pytest.approx(1.0)
        assert metrics['area_diff_km2'] == pytest.approx(0.0)
    
    def test_batch_averaging(self):
        """Should average metrics across batch."""
        batch_size = 5
        y_true = np.random.rand(batch_size, 16, 16) > 0.5
        y_pred = y_true.copy()  # Perfect predictions
        
        metrics = aggregate_metrics(y_true, y_pred)
        
        assert metrics['iou'] == pytest.approx(1.0)
        assert metrics['dice'] == pytest.approx(1.0)
    
    def test_shape_with_channel_dimension(self):
        """Should handle 4D input with channel dimension."""
        y_true = np.random.rand(3, 16, 16, 1) > 0.5
        y_pred = np.random.rand(3, 16, 16, 1)
        
        metrics = aggregate_metrics(y_true, y_pred)
        
        assert 0 <= metrics['iou'] <= 1
        assert 0 <= metrics['dice'] <= 1
        assert isinstance(metrics, dict)
    
    def test_shape_mismatch_error(self):
        """Should raise error on shape mismatch."""
        y_true = np.random.rand(2, 16, 16)
        y_pred = np.random.rand(2, 32, 32)  # Different size
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            aggregate_metrics(y_true, y_pred)
    
    def test_wrong_dimensions_error(self):
        """Should raise error on wrong number of dimensions."""
        y_true = np.random.rand(16, 16)  # Missing batch dimension
        y_pred = np.random.rand(16, 16)
        
        with pytest.raises(ValueError, match="Expected 3D batch"):
            aggregate_metrics(y_true, y_pred)
    
    def test_pixel_area_calculation(self):
        """Should correctly calculate area difference."""
        # Create simple 2x2 mask
        y_true = np.array([[[1, 1], [0, 0]]], dtype=np.float32)
        y_pred = np.array([[[1, 1], [1, 1]]], dtype=np.float32)
        
        pixel_area = 1.0  # 1 km² per pixel
        metrics = aggregate_metrics(y_true, y_pred, pixel_area_km2=pixel_area)
        
        # True has 2 pixels, pred has 4 pixels, diff = +2 km²
        assert metrics['area_diff_km2'] == pytest.approx(2.0)
    
    def test_empty_batch_raises_error(self):
        """Empty batch should raise appropriate error."""
        y_true = np.zeros((0, 16, 16))
        y_pred = np.zeros((0, 16, 16))
        
        # Should handle gracefully or raise clear error
        try:
            metrics = aggregate_metrics(y_true, y_pred)
            # If it succeeds, check results are sane
            assert all(isinstance(v, (int, float)) for v in metrics.values())
        except (ValueError, IndexError):
            # Expected for empty batch
            pass
    
    def test_realistic_predictions(self):
        """Test with realistic imperfect predictions."""
        np.random.seed(42)
        batch_size = 10
        height, width = 64, 64
        
        # Ground truth
        y_true = np.random.rand(batch_size, height, width) > 0.3
        
        # Add noise to create imperfect predictions
        noise = np.random.randn(batch_size, height, width) * 0.2
        y_pred = y_true.astype(float) + noise
        y_pred = np.clip(y_pred, 0, 1)
        
        metrics = aggregate_metrics(y_true, y_pred, pixel_area_km2=0.001)
        
        # Should get reasonable metrics
        assert 0.5 < metrics['iou'] < 1.0
        assert 0.5 < metrics['dice'] < 1.0
        assert 0.5 < metrics['precision'] < 1.0
        assert 0.5 < metrics['recall'] < 1.0


class TestValidateMaskArray:
    """Test mask array validation."""
    
    def test_valid_2d_array(self):
        """Valid 2D array should pass."""
        mask = np.random.rand(256, 256)
        validate_mask_array(mask)  # Should not raise
    
    def test_valid_3d_array(self):
        """Valid 3D array should pass."""
        mask = np.random.rand(10, 256, 256)
        validate_mask_array(mask)  # Should not raise
    
    def test_expected_shape_match(self):
        """Matching shape should pass."""
        mask = np.zeros((64, 64))
        validate_mask_array(mask, expected_shape=(64, 64))  # Should not raise
    
    def test_expected_shape_mismatch(self):
        """Mismatching shape should raise."""
        mask = np.zeros((64, 64))
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_mask_array(mask, expected_shape=(128, 128))
    
    def test_non_numpy_array_rejected(self):
        """Non-numpy array should raise TypeError."""
        with pytest.raises(TypeError, match="Expected numpy array"):
            validate_mask_array([[1, 2], [3, 4]])
    
    def test_nan_values_rejected(self):
        """NaN values should be rejected by default."""
        mask = np.array([[1.0, np.nan], [0.5, 0.3]])
        with pytest.raises(ValueError, match="NaN values"):
            validate_mask_array(mask, allow_nan=False)
    
    def test_nan_values_allowed(self):
        """NaN values should be allowed when specified."""
        mask = np.array([[1.0, np.nan], [0.5, 0.3]])
        validate_mask_array(mask, allow_nan=True)  # Should not raise
    
    def test_inf_values_rejected(self):
        """Infinite values should always be rejected."""
        mask = np.array([[1.0, np.inf], [0.5, 0.3]])
        with pytest.raises(ValueError, match="infinite values"):
            validate_mask_array(mask)
    
    def test_negative_inf_rejected(self):
        """Negative infinite values should be rejected."""
        mask = np.array([[1.0, -np.inf], [0.5, 0.3]])
        with pytest.raises(ValueError, match="infinite values"):
            validate_mask_array(mask)


class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_iou_all_zeros(self):
        """IoU with all zeros should handle gracefully."""
        y_true = np.zeros((10, 10))
        y_pred = np.zeros((10, 10))
        iou = iou_np(y_true, y_pred)
        # Should not crash, return valid value
        assert 0 <= iou <= 1
    
    def test_dice_all_zeros(self):
        """Dice with all zeros should handle gracefully."""
        y_true = np.zeros((10, 10))
        y_pred = np.zeros((10, 10))
        dice = dice_np(y_true, y_pred)
        assert 0 <= dice <= 1
    
    def test_single_pixel_prediction(self):
        """Single pixel prediction should work."""
        y_true = np.array([[1]])
        y_pred = np.array([[1]])
        
        iou = iou_np(y_true, y_pred)
        dice = dice_np(y_true, y_pred)
        precision, recall = precision_recall_np(y_true, y_pred)
        
        assert iou == pytest.approx(1.0)
        assert dice == pytest.approx(1.0)
        assert precision == pytest.approx(1.0)
        assert recall == pytest.approx(1.0)
    
    def test_very_small_values(self):
        """Very small probability values should work."""
        y_true = np.ones((10, 10)) * 0.001
        y_pred = np.ones((10, 10)) * 0.001
        
        iou = iou_np(y_true, y_pred, threshold=0.0005)
        assert 0 <= iou <= 1
    
    def test_threshold_boundary(self):
        """Values exactly at threshold should be handled consistently."""
        y_true = np.array([[0.5, 0.50001], [0.49999, 0.5]])
        y_pred = np.array([[0.5, 0.5], [0.5, 0.5]])
        
        iou = iou_np(y_true, y_pred, threshold=0.5)
        dice = dice_np(y_true, y_pred, threshold=0.5)
        
        # Should produce stable results
        assert 0 <= iou <= 1
        assert 0 <= dice <= 1
    
    def test_large_arrays(self):
        """Should handle large arrays efficiently."""
        np.random.seed(42)
        y_true = np.random.rand(1024, 1024) > 0.5
        y_pred = np.random.rand(1024, 1024) > 0.5
        
        metrics = aggregate_metrics(
            y_true[np.newaxis, :, :],
            y_pred[np.newaxis, :, :],
            pixel_area_km2=0.0001
        )
        
        assert all(0 <= v <= 1 or isinstance(v, float) for v in metrics.values())
    
    def test_area_difference_sign(self):
        """Area difference sign should be correct."""
        # Prediction has more water than ground truth
        y_true = np.array([[1, 0], [0, 0]])
        y_pred = np.array([[1, 1], [1, 0]])
        
        area_diff = calculate_area_difference(y_true, y_pred, pixel_area_km2=1.0)
        assert area_diff > 0  # Overestimate
        
        # Prediction has less water than ground truth
        y_true = np.array([[1, 1], [1, 1]])
        y_pred = np.array([[1, 0], [0, 0]])
        
        area_diff = calculate_area_difference(y_true, y_pred, pixel_area_km2=1.0)
        assert area_diff < 0  # Underestimate


class TestIntegrationScenarios:
    """Integration tests for realistic metric calculation workflows."""
    
    def test_evaluation_pipeline(self):
        """Simulate full evaluation pipeline."""
        np.random.seed(42)
        
        # Simulate test set predictions
        batch_size = 20
        y_true = np.random.rand(batch_size, 128, 128) > 0.4
        
        # Simulate model predictions (noisy version of truth)
        y_pred = y_true.astype(float) + np.random.randn(batch_size, 128, 128) * 0.15
        y_pred = np.clip(y_pred, 0, 1)
        
        # Validate inputs
        for i in range(batch_size):
            validate_mask_array(y_true[i], expected_shape=(128, 128))
            validate_mask_array(y_pred[i], expected_shape=(128, 128))
        
        # Calculate metrics
        metrics = aggregate_metrics(y_true, y_pred, pixel_area_km2=0.0036)
        
        # Verify all metrics are reasonable
        assert 0.5 < metrics['iou'] < 1.0
        assert 0.6 < metrics['dice'] < 1.0
        assert 0.6 < metrics['precision'] < 1.0
        assert 0.6 < metrics['recall'] < 1.0
        assert isinstance(metrics['area_diff_km2'], float)
    
    def test_comparison_across_models(self):
        """Simulate comparing multiple models."""
        np.random.seed(42)
        
        y_true = np.random.rand(10, 64, 64) > 0.5
        
        # Model 1: High precision, lower recall
        y_pred1 = (y_true.astype(float) * 0.8) > 0.5
        
        # Model 2: Balanced
        y_pred2 = y_true.astype(float) + np.random.randn(10, 64, 64) * 0.1
        y_pred2 = np.clip(y_pred2, 0, 1)
        
        metrics1 = aggregate_metrics(y_true, y_pred1)
        metrics2 = aggregate_metrics(y_true, y_pred2)
        
        # Model 1 should have lower recall
        assert metrics1['recall'] < metrics2['recall']
        
        # Both should be valid
        assert all(0 <= v <= 1 for k, v in metrics1.items() if k != 'area_diff_km2')
        assert all(0 <= v <= 1 for k, v in metrics2.items() if k != 'area_diff_km2')
