"""Tests for safe path handling utilities."""

import pytest
import tempfile
from pathlib import Path

from utils.data_utils import (
    validate_path,
    validate_safe_pattern,
    resolve_safe_path,
    validate_directory_path,
)


class TestValidatePath:
    """Test basic path validation."""
    
    def test_valid_string_path(self):
        """Valid string path should return Path object."""
        result = validate_path("/data/yearly", "test_path")
        assert isinstance(result, Path)
        assert str(result) == "/data/yearly"
    
    def test_valid_path_object(self):
        """Path object should be accepted and returned."""
        input_path = Path("/data/yearly")
        result = validate_path(input_path, "test_path")
        assert isinstance(result, Path)
        assert result == input_path
    
    def test_empty_string_rejected(self):
        """Empty string should raise TypeError."""
        with pytest.raises(TypeError, match="non-empty path"):
            validate_path("", "test_path")
    
    def test_whitespace_only_rejected(self):
        """Whitespace-only string should raise TypeError."""
        with pytest.raises(TypeError, match="non-empty path"):
            validate_path("   ", "test_path")
    
    def test_none_rejected(self):
        """None should raise TypeError."""
        with pytest.raises(TypeError, match="non-empty path"):
            validate_path(None, "test_path")


class TestValidateSafePattern:
    """Test filename pattern validation."""
    
    def test_simple_pattern(self):
        """Simple relative pattern should be accepted."""
        result = validate_safe_pattern("*.tif")
        assert result == "*.tif"
    
    def test_relative_path_pattern(self):
        """Relative path with subdirectory should be accepted."""
        result = validate_safe_pattern("data/yearly/*.tif")
        assert result == "data/yearly/*.tif"
    
    def test_absolute_path_rejected(self):
        """Absolute path should be rejected."""
        with pytest.raises(ValueError, match="relative local filename"):
            validate_safe_pattern("/etc/passwd")
    
    def test_parent_traversal_rejected(self):
        """Parent directory traversal should be rejected."""
        with pytest.raises(ValueError, match="relative local filename"):
            validate_safe_pattern("../etc/passwd")
    
    def test_nested_traversal_rejected(self):
        """Nested traversal should be rejected."""
        with pytest.raises(ValueError, match="relative local filename"):
            validate_safe_pattern("data/../../etc/passwd")
    
    def test_empty_pattern_rejected(self):
        """Empty pattern should be rejected."""
        with pytest.raises(ValueError, match="non-empty string"):
            validate_safe_pattern("")


class TestResolveSafePath:
    """Test safe path resolution."""
    
    def test_simple_relative_path(self):
        """Simple relative path should resolve correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            result = resolve_safe_path(base, "subdir/file.txt")
            assert result.is_absolute()
            assert result.parent.parent == base.resolve()
    
    def test_stays_within_base(self):
        """Resolved path should stay within base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            subdir = base / "data"
            subdir.mkdir()
            
            result = resolve_safe_path(base, "data/file.txt")
            assert result.is_relative_to(base.resolve())
    
    def test_traversal_attack_prevented(self):
        """Directory traversal should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            with pytest.raises(ValueError, match="escapes base directory"):
                resolve_safe_path(base, "../etc/passwd")
    
    def test_nested_traversal_blocked(self):
        """Nested traversal attack should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            (base / "data").mkdir()
            
            with pytest.raises(ValueError, match="escapes base directory"):
                resolve_safe_path(base, "data/../../etc/passwd")
    
    def test_symlink_traversal_blocked(self):
        """Symlink escape should be blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            # Create symlink pointing outside base
            link = base / "escape"
            try:
                link.symlink_to("/tmp")
                
                with pytest.raises(ValueError, match="escapes base directory"):
                    resolve_safe_path(base, "escape/../sensitive")
            except (OSError, NotImplementedError):
                # Symlinks may not be supported on Windows
                pytest.skip("Symlinks not supported")
    
    def test_must_exist_enforced(self):
        """must_exist parameter should be enforced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            with pytest.raises(FileNotFoundError):
                resolve_safe_path(base, "nonexistent.txt", must_exist=True)
    
    def test_must_exist_with_existing_file(self):
        """must_exist should pass for existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            test_file = base / "test.txt"
            test_file.touch()
            
            result = resolve_safe_path(base, "test.txt", must_exist=True)
            assert result.exists()
    
    def test_allowed_extensions_enforced(self):
        """Extension whitelist should be enforced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            # .txt not in allowed list
            with pytest.raises(ValueError, match="not in allowed list"):
                resolve_safe_path(
                    base,
                    "file.txt",
                    allowed_extensions=[".tif", ".tiff"]
                )
    
    def test_allowed_extensions_case_insensitive(self):
        """Extension check should be case-insensitive."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            # Should accept .TIF when .tif is in list
            result = resolve_safe_path(
                base,
                "file.TIF",
                allowed_extensions=[".tif"]
            )
            assert result.suffix == ".TIF"
    
    def test_allowed_extensions_pass(self):
        """Valid extension should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            result = resolve_safe_path(
                base,
                "data.tif",
                allowed_extensions=[".tif", ".tiff", ".geotiff"]
            )
            assert result.suffix == ".tif"
    
    def test_empty_relative_path_rejected(self):
        """Empty relative path should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            with pytest.raises(ValueError, match="non-empty string"):
                resolve_safe_path(base, "")


class TestValidateDirectoryPath:
    """Test directory path validation."""
    
    def test_existing_directory(self):
        """Existing directory should validate."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_directory_path(tmpdir)
            assert result.is_dir()
            assert result.exists()
    
    def test_nonexistent_without_create(self):
        """Nonexistent directory should raise error by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "does_not_exist"
            
            with pytest.raises(FileNotFoundError):
                validate_directory_path(str(nonexistent))
    
    def test_create_if_missing(self):
        """create_if_missing should create directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_directory"
            
            result = validate_directory_path(str(new_dir), create_if_missing=True)
            assert result.exists()
            assert result.is_dir()
    
    def test_create_nested_directories(self):
        """Should create nested directories with parents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "level1" / "level2" / "level3"
            
            result = validate_directory_path(str(nested), create_if_missing=True)
            assert result.exists()
            assert result.is_dir()
    
    def test_file_not_directory(self):
        """File path should raise NotADirectoryError."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath = f.name
        
        try:
            with pytest.raises(NotADirectoryError, match="not a directory"):
                validate_directory_path(filepath)
        finally:
            Path(filepath).unlink()
    
    def test_require_writable_pass(self):
        """Writable directory should pass writability check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validate_directory_path(tmpdir, require_writable=True)
            assert result.is_dir()
    
    def test_require_writable_fail(self):
        """Read-only directory should fail writability check."""
        # This test is platform-dependent and may not work on all systems
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Make directory read-only
            try:
                tmppath.chmod(0o555)
                
                with pytest.raises(PermissionError, match="not writable"):
                    validate_directory_path(tmpdir, require_writable=True)
            except (OSError, PermissionError):
                pytest.skip("Cannot set permissions on this system")
            finally:
                # Restore permissions for cleanup
                try:
                    tmppath.chmod(0o755)
                except (OSError, PermissionError):
                    pass
    
    def test_create_and_check_writable(self):
        """Should create directory and verify writability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "writable_dir"
            
            result = validate_directory_path(
                str(new_dir),
                create_if_missing=True,
                require_writable=True
            )
            assert result.exists()
            assert result.is_dir()
    
    def test_empty_path_rejected(self):
        """Empty path should be rejected."""
        with pytest.raises(TypeError, match="non-empty path"):
            validate_directory_path("")


class TestIntegrationScenarios:
    """Integration tests for realistic use cases."""
    
    def test_safe_data_loading_workflow(self):
        """Simulate safe data loading from user input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            data_dir = base / "data" / "yearly"
            data_dir.mkdir(parents=True)
            
            # Create test file
            test_file = data_dir / "2020.tif"
            test_file.touch()
            
            # User provides relative path
            user_input = "data/yearly/2020.tif"
            
            # Safely resolve and validate
            safe_path = resolve_safe_path(
                base,
                user_input,
                must_exist=True,
                allowed_extensions=[".tif", ".tiff"]
            )
            
            assert safe_path.exists()
            assert safe_path.is_file()
            assert safe_path.suffix == ".tif"
    
    def test_output_directory_preparation(self):
        """Simulate output directory setup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs" / "experiment_001"
            
            # Validate and create output directory
            validated = validate_directory_path(
                str(output_dir),
                create_if_missing=True,
                require_writable=True
            )
            
            # Should be able to write to it
            test_file = validated / "results.csv"
            test_file.write_text("test data")
            assert test_file.exists()
    
    def test_reject_malicious_input(self):
        """Should reject various attack patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            
            malicious_inputs = [
                "../../../etc/passwd",
                "data/../../../etc/passwd",
                "//etc/passwd",
                "./../sensitive",
            ]
            
            for malicious in malicious_inputs:
                with pytest.raises(ValueError):
                    resolve_safe_path(base, malicious)
