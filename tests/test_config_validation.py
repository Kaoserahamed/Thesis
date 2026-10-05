"""Tests for configuration loading and validation."""

import pytest
import tempfile
import yaml
from pathlib import Path
from pydantic import ValidationError

from utils.data_utils import load_config, ThesisConfig


class TestThesisConfigModel:
    """Test the Pydantic configuration model."""
    
    def test_minimal_valid_config(self):
        """Config with only defaults should be valid."""
        config = ThesisConfig()
        assert config.mlflow_tracking_uri == "file:./mlruns"
        assert config.mlflow_experiment_name == "river-morphology"
        assert config.disk_free_gb == 10.0
    
    def test_full_valid_config(self):
        """Config with all fields should be valid."""
        config = ThesisConfig(
            yearly_dir="/data/yearly",
            quarterly_dir="/data/quarterly",
            bimonthly_dir="/data/bimonthly",
            mlflow_tracking_uri="http://mlflow-server:5000",
            mlflow_experiment_name="custom-experiment",
            disk_free_gb=50.0,
            error_webhook_url="https://example.com/webhook"
        )
        assert config.yearly_dir == "/data/yearly"
        assert config.mlflow_tracking_uri == "http://mlflow-server:5000"
        assert config.disk_free_gb == 50.0
    
    def test_mlflow_uri_defaults_to_local(self):
        """MLflow URI should default to local file-based tracking."""
        config = ThesisConfig(mlflow_tracking_uri=None)
        assert config.mlflow_tracking_uri == "file:./mlruns"
        
        config = ThesisConfig(mlflow_tracking_uri="")
        assert config.mlflow_tracking_uri == "file:./mlruns"
    
    def test_mlflow_uri_accepts_valid_schemes(self):
        """MLflow URI should accept standard schemes."""
        valid_uris = [
            "file:./mlruns",
            "file:///absolute/path/mlruns",
            "http://localhost:5000",
            "https://mlflow.example.com",
            "sqlite:///mlflow.db",
            "postgresql://user:pass@host:5432/mlflow",
            "mysql://user:pass@host:3306/mlflow",
        ]
        for uri in valid_uris:
            config = ThesisConfig(mlflow_tracking_uri=uri)
            assert config.mlflow_tracking_uri == uri
    
    def test_mlflow_uri_rejects_invalid_schemes(self):
        """MLflow URI should reject invalid schemes."""
        invalid_uris = [
            "ftp://server/path",
            "invalid-scheme://path",
            "just-a-path",
            "/absolute/path/without/scheme",
        ]
        for uri in invalid_uris:
            with pytest.raises(ValidationError, match="MLflow tracking URI"):
                ThesisConfig(mlflow_tracking_uri=uri)
    
    def test_disk_free_gb_validation(self):
        """Disk free threshold should be validated."""
        # Valid values
        for value in [0.0, 1.0, 10.0, 100.0, 1000.0]:
            config = ThesisConfig(disk_free_gb=value)
            assert config.disk_free_gb == value
        
        # Negative value should fail
        with pytest.raises(ValidationError, match="non-negative"):
            ThesisConfig(disk_free_gb=-1.0)
        
        # Unreasonably large value should fail
        with pytest.raises(ValidationError, match="unreasonably large"):
            ThesisConfig(disk_free_gb=20000.0)
    
    def test_disk_free_gb_type_validation(self):
        """Disk free threshold should accept numeric types."""
        # Integer should be converted to float
        config = ThesisConfig(disk_free_gb=50)
        assert config.disk_free_gb == 50.0
        assert isinstance(config.disk_free_gb, float)
        
        # String number should fail
        with pytest.raises(ValidationError):
            ThesisConfig(disk_free_gb="not-a-number")
    
    def test_extra_fields_allowed(self):
        """Config should allow extra fields for forward compatibility."""
        config = ThesisConfig(
            yearly_dir="/data/yearly",
            custom_field="custom_value",
            another_setting=123
        )
        assert config.yearly_dir == "/data/yearly"
        # Extra fields are allowed but not stored in the model


class TestLoadConfig:
    """Test the load_config function with YAML files."""
    
    def test_load_valid_config_file(self):
        """Should load and validate a valid YAML config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'yearly_dir': '/data/yearly',
                'mlflow_tracking_uri': 'file:./mlruns',
                'disk_free_gb': 20.0,
            }, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert isinstance(config, ThesisConfig)
            assert config.yearly_dir == '/data/yearly'
            assert config.mlflow_tracking_uri == 'file:./mlruns'
            assert config.disk_free_gb == 20.0
        finally:
            Path(config_path).unlink()
    
    def test_load_minimal_config(self):
        """Should load config with only defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({}, f)  # Empty config, should use defaults
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config.mlflow_tracking_uri == 'file:./mlruns'
            assert config.disk_free_gb == 10.0
        finally:
            Path(config_path).unlink()
    
    def test_load_config_with_environment_vars(self):
        """Should load config that references environment variables."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump({
                'yearly_dir': '${YEARLY_DIR}',  # Placeholder for env var
                'mlflow_experiment_name': 'test-experiment',
            }, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config.yearly_dir == '${YEARLY_DIR}'  # Raw value preserved
            assert config.mlflow_experiment_name == 'test-experiment'
        finally:
            Path(config_path).unlink()
    
    def test_load_config_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config('/nonexistent/path/config.yaml')
    
    def test_load_config_invalid_extension(self):
        """Should reject non-YAML file extensions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("some content")
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="must point to a YAML file"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_load_config_invalid_yaml_structure(self):
        """Should reject non-dict top-level YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(['list', 'not', 'dict'], f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="top-level mapping"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_load_config_invalid_field_value(self):
        """Should reject config with invalid field values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'disk_free_gb': -5.0,  # Negative value
            }, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid configuration"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_load_config_invalid_mlflow_uri(self):
        """Should reject config with invalid MLflow URI."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'mlflow_tracking_uri': 'ftp://invalid-scheme',
            }, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid configuration"):
                load_config(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_load_config_empty_string_path(self):
        """Should reject empty string as config path."""
        with pytest.raises(TypeError, match="non-empty path"):
            load_config("")
    
    def test_load_config_malformed_yaml(self):
        """Should handle malformed YAML gracefully."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [[[")
            config_path = f.name
        
        try:
            with pytest.raises(Exception):  # yaml.YAMLError or similar
                load_config(config_path)
        finally:
            Path(config_path).unlink()


class TestConfigValidationIntegration:
    """Integration tests for config validation in realistic scenarios."""
    
    def test_config_with_all_data_directories(self):
        """Should validate config with all temporal resolutions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'yearly_dir': '/data/raw/yearly',
                'quarterly_dir': '/data/raw/quarterly',
                'bimonthly_dir': '/data/raw/bimonthly',
                'mlflow_tracking_uri': 'file:./mlruns',
                'mlflow_experiment_name': 'river-morphology',
                'disk_free_gb': 50.0,
            }, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config.yearly_dir == '/data/raw/yearly'
            assert config.quarterly_dir == '/data/raw/quarterly'
            assert config.bimonthly_dir == '/data/raw/bimonthly'
        finally:
            Path(config_path).unlink()
    
    def test_config_with_remote_mlflow(self):
        """Should validate config with remote MLflow server."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'mlflow_tracking_uri': 'https://mlflow.example.com',
                'mlflow_experiment_name': 'production-experiment',
            }, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config.mlflow_tracking_uri == 'https://mlflow.example.com'
            assert config.mlflow_experiment_name == 'production-experiment'
        finally:
            Path(config_path).unlink()
    
    def test_config_preserves_forward_compatibility(self):
        """Should allow unknown fields for future versions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                'yearly_dir': '/data/yearly',
                'future_feature_v2': 'some_value',
                'another_new_field': 123,
            }, f)
            config_path = f.name
        
        try:
            config = load_config(config_path)
            assert config.yearly_dir == '/data/yearly'
            # Should not fail on unknown fields
        finally:
            Path(config_path).unlink()
