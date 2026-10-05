"""Tests for structured exception classes."""

import pytest
from utils.error_tracking import (
    ThesisError,
    DataLoadError,
    ConfigurationError,
    ModelBuildError,
    TrainingError,
    PredictionError,
    ValidationError,
    ResourceError,
)


class TestThesisError:
    """Test base exception class."""
    
    def test_basic_message(self):
        """Should store message."""
        error = ThesisError("Something went wrong")
        assert error.message == "Something went wrong"
        assert str(error) == "Something went wrong"
    
    def test_with_context(self):
        """Should store context dict."""
        context = {"user": "test", "action": "load_data"}
        error = ThesisError("Error occurred", context=context)
        assert error.context == context
        assert error.context["user"] == "test"
    
    def test_default_empty_context(self):
        """Default context should be empty dict."""
        error = ThesisError("Error")
        assert error.context == {}
        assert isinstance(error.context, dict)
    
    def test_is_exception(self):
        """Should be a proper Exception."""
        error = ThesisError("Test")
        assert isinstance(error, Exception)
    
    def test_can_be_raised_and_caught(self):
        """Should work in try/except."""
        with pytest.raises(ThesisError) as exc_info:
            raise ThesisError("Test error")
        
        assert "Test error" in str(exc_info.value)


class TestDataLoadError:
    """Test data loading exception."""
    
    def test_with_filepath(self):
        """Should store filepath."""
        error = DataLoadError("File not found", filepath="/data/test.tif")
        assert error.filepath == "/data/test.tif"
        assert error.message == "File not found"
    
    def test_without_filepath(self):
        """Filepath should be optional."""
        error = DataLoadError("Invalid data format")
        assert error.filepath is None
    
    def test_inherits_from_thesis_error(self):
        """Should inherit from base class."""
        error = DataLoadError("Error")
        assert isinstance(error, ThesisError)
        assert isinstance(error, Exception)
    
    def test_with_context(self):
        """Should support context."""
        context = {"expected_shape": (256, 256), "actual_shape": (128, 128)}
        error = DataLoadError(
            "Shape mismatch",
            filepath="/data/image.tif",
            context=context
        )
        assert error.context["expected_shape"] == (256, 256)
        assert error.filepath == "/data/image.tif"
    
    def test_catchable_as_thesis_error(self):
        """Should be catchable as base type."""
        with pytest.raises(ThesisError):
            raise DataLoadError("Test")


class TestConfigurationError:
    """Test configuration exception."""
    
    def test_with_field(self):
        """Should store field name."""
        error = ConfigurationError("Invalid value", field="mlflow_tracking_uri")
        assert error.field == "mlflow_tracking_uri"
    
    def test_without_field(self):
        """Field should be optional."""
        error = ConfigurationError("Missing configuration")
        assert error.field is None
    
    def test_with_context(self):
        """Should support context."""
        context = {"provided": "invalid-uri", "expected": "file:// or http://"}
        error = ConfigurationError(
            "Invalid URI scheme",
            field="mlflow_tracking_uri",
            context=context
        )
        assert error.context["provided"] == "invalid-uri"
        assert error.field == "mlflow_tracking_uri"


class TestModelBuildError:
    """Test model building exception."""
    
    def test_with_model_name(self):
        """Should store model name."""
        error = ModelBuildError("Failed to compile", model_name="attention_unet_convlstm")
        assert error.model_name == "attention_unet_convlstm"
    
    def test_without_model_name(self):
        """Model name should be optional."""
        error = ModelBuildError("Unknown model architecture")
        assert error.model_name is None
    
    def test_with_context(self):
        """Should support context."""
        context = {"input_shape": (5, 256, 256, 1), "error_type": "shape_mismatch"}
        error = ModelBuildError(
            "Invalid input shape",
            model_name="convlstm",
            context=context
        )
        assert error.context["input_shape"] == (5, 256, 256, 1)


class TestTrainingError:
    """Test training exception."""
    
    def test_with_epoch(self):
        """Should store epoch number."""
        error = TrainingError("Training diverged", epoch=42)
        assert error.epoch == 42
    
    def test_without_epoch(self):
        """Epoch should be optional."""
        error = TrainingError("Out of memory during training")
        assert error.epoch is None
    
    def test_with_context(self):
        """Should support context."""
        context = {
            "loss": float('inf'),
            "learning_rate": 1e-4,
            "batch_size": 4
        }
        error = TrainingError(
            "Loss exploded to infinity",
            epoch=10,
            context=context
        )
        assert error.context["loss"] == float('inf')
        assert error.epoch == 10


class TestPredictionError:
    """Test prediction exception."""
    
    def test_with_sample_id(self):
        """Should store sample ID."""
        error = PredictionError("Prediction failed", sample_id="2020_Q1")
        assert error.sample_id == "2020_Q1"
    
    def test_without_sample_id(self):
        """Sample ID should be optional."""
        error = PredictionError("Model not loaded")
        assert error.sample_id is None
    
    def test_with_context(self):
        """Should support context."""
        context = {"input_shape": (5, 256, 256, 1), "model_output_shape": (1, 256, 256)}
        error = PredictionError(
            "Output shape mismatch",
            sample_id="test_001",
            context=context
        )
        assert error.context["input_shape"] == (5, 256, 256, 1)


class TestValidationError:
    """Test validation exception."""
    
    def test_with_validation_type(self):
        """Should store validation type."""
        error = ValidationError("Invalid range", validation_type="pixel_values")
        assert error.validation_type == "pixel_values"
    
    def test_without_validation_type(self):
        """Validation type should be optional."""
        error = ValidationError("Validation failed")
        assert error.validation_type is None
    
    def test_with_context(self):
        """Should support context."""
        context = {"min": -0.5, "max": 1.5, "expected_range": "[0, 1]"}
        error = ValidationError(
            "Values out of range",
            validation_type="pixel_values",
            context=context
        )
        assert error.context["min"] == -0.5
        assert error.validation_type == "pixel_values"


class TestResourceError:
    """Test resource exception."""
    
    def test_with_resource_type(self):
        """Should store resource type."""
        error = ResourceError("Insufficient memory", resource_type="RAM")
        assert error.resource_type == "RAM"
    
    def test_without_resource_type(self):
        """Resource type should be optional."""
        error = ResourceError("Resource limit exceeded")
        assert error.resource_type is None
    
    def test_with_context(self):
        """Should support context."""
        context = {"required_gb": 32, "available_gb": 16, "process": "model_training"}
        error = ResourceError(
            "Not enough RAM",
            resource_type="RAM",
            context=context
        )
        assert error.context["required_gb"] == 32
        assert error.resource_type == "RAM"


class TestExceptionHierarchy:
    """Test exception inheritance and catching behavior."""
    
    def test_all_inherit_from_thesis_error(self):
        """All custom exceptions should inherit from ThesisError."""
        exceptions = [
            DataLoadError("test"),
            ConfigurationError("test"),
            ModelBuildError("test"),
            TrainingError("test"),
            PredictionError("test"),
            ValidationError("test"),
            ResourceError("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, ThesisError)
            assert isinstance(exc, Exception)
    
    def test_catch_any_thesis_error(self):
        """Should be able to catch any thesis error with base class."""
        errors_raised = []
        
        for error_class in [DataLoadError, ConfigurationError, ModelBuildError]:
            try:
                raise error_class("test error")
            except ThesisError as e:
                errors_raised.append(type(e).__name__)
        
        assert len(errors_raised) == 3
        assert "DataLoadError" in errors_raised
    
    def test_specific_exception_catching(self):
        """Should be able to catch specific exception types."""
        with pytest.raises(DataLoadError):
            raise DataLoadError("specific error")
        
        # Should not catch as ConfigurationError
        with pytest.raises(DataLoadError):
            try:
                raise DataLoadError("test")
            except ConfigurationError:
                pytest.fail("Should not catch as ConfigurationError")


class TestExceptionUsagePatterns:
    """Test realistic usage patterns."""
    
    def test_data_pipeline_error_propagation(self):
        """Simulate error propagation in data pipeline."""
        def load_geotiff(path: str):
            if not path.endswith('.tif'):
                raise DataLoadError(
                    "Invalid file format",
                    filepath=path,
                    context={"expected": ".tif", "got": path.split('.')[-1]}
                )
        
        with pytest.raises(DataLoadError) as exc_info:
            load_geotiff("/data/image.png")
        
        error = exc_info.value
        assert error.filepath == "/data/image.png"
        assert error.context["got"] == "png"
    
    def test_configuration_validation_error(self):
        """Simulate configuration validation."""
        def validate_config(config: dict):
            if "mlflow_tracking_uri" not in config:
                raise ConfigurationError(
                    "Required field missing",
                    field="mlflow_tracking_uri",
                    context={"available_fields": list(config.keys())}
                )
        
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config({"yearly_dir": "/data"})
        
        error = exc_info.value
        assert error.field == "mlflow_tracking_uri"
        assert "yearly_dir" in error.context["available_fields"]
    
    def test_training_interruption(self):
        """Simulate training interruption."""
        def train_epoch(epoch: int, loss: float):
            if loss > 1e6:
                raise TrainingError(
                    "Loss exploded",
                    epoch=epoch,
                    context={"loss": loss, "threshold": 1e6}
                )
        
        with pytest.raises(TrainingError) as exc_info:
            train_epoch(epoch=5, loss=1e7)
        
        error = exc_info.value
        assert error.epoch == 5
        assert error.context["loss"] == 1e7
    
    def test_resource_exhaustion(self):
        """Simulate resource exhaustion."""
        def check_disk_space(required_gb: float, available_gb: float):
            if available_gb < required_gb:
                raise ResourceError(
                    f"Insufficient disk space: need {required_gb}GB, have {available_gb}GB",
                    resource_type="disk",
                    context={
                        "required_gb": required_gb,
                        "available_gb": available_gb,
                        "deficit_gb": required_gb - available_gb
                    }
                )
        
        with pytest.raises(ResourceError) as exc_info:
            check_disk_space(required_gb=100, available_gb=50)
        
        error = exc_info.value
        assert error.resource_type == "disk"
        assert error.context["deficit_gb"] == 50


class TestContextPreservation:
    """Test that context is properly preserved through exception flow."""
    
    def test_context_accessible_after_catch(self):
        """Context should be accessible after catching exception."""
        original_context = {"key": "value", "number": 42}
        
        try:
            raise ThesisError("test", context=original_context)
        except ThesisError as e:
            assert e.context == original_context
            assert e.context["key"] == "value"
            assert e.context["number"] == 42
    
    def test_context_mutation_after_creation(self):
        """Should be able to add to context after creation."""
        error = ThesisError("test")
        error.context["added_field"] = "new_value"
        
        assert error.context["added_field"] == "new_value"
    
    def test_nested_context_data(self):
        """Should handle nested context data."""
        context = {
            "model": {
                "name": "attention_unet",
                "parameters": {"seq_len": 5, "lr": 1e-4}
            },
            "data": {
                "shape": (256, 256),
                "source": "/data/yearly"
            }
        }
        
        error = ModelBuildError("Test", context=context)
        assert error.context["model"]["name"] == "attention_unet"
        assert error.context["data"]["shape"] == (256, 256)
