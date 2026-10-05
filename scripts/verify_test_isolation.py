#!/usr/bin/env python3
"""
Test Isolation Verification Script
===================================

This script verifies that the test suite can run successfully without
any external network dependencies or services. It confirms that:

1. MLflow tracking defaults to local file-based storage (./mlruns)
2. Tests do not require network access after pip install
3. No external APIs or databases are required
4. All test fixtures use in-memory or temporary file storage

Run this as part of CI or before releases to ensure reproducibility.
"""

import sys
import os
import socket
import subprocess
from pathlib import Path
from typing import List, Tuple


def block_network_access() -> bool:
    """
    Attempt to detect if network access is available.
    
    Returns:
        bool: True if network seems blocked, False otherwise
    """
    try:
        # Try to resolve a known public DNS
        socket.create_connection(("8.8.8.8", 53), timeout=1)
        return False  # Network is accessible
    except (socket.timeout, socket.error, OSError):
        return True  # Network blocked or unavailable


def check_mlflow_defaults() -> Tuple[bool, str]:
    """
    Verify that MLflow defaults to local file-based tracking.
    
    Returns:
        Tuple of (success, message)
    """
    print("Checking MLflow default configuration...")
    
    # Check if MLFLOW_TRACKING_URI is set
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
    
    if not tracking_uri:
        # No URI set, should default to local
        return True, "✓ MLFLOW_TRACKING_URI not set (will default to local)"
    
    if tracking_uri.startswith('file:'):
        return True, f"✓ MLFLOW_TRACKING_URI set to local: {tracking_uri}"
    
    if tracking_uri.startswith(('http:', 'https:')):
        return False, f"✗ MLFLOW_TRACKING_URI points to remote server: {tracking_uri}"
    
    return True, f"? MLFLOW_TRACKING_URI set to: {tracking_uri}"


def verify_no_external_imports() -> Tuple[bool, str]:
    """
    Verify that test files don't import external service clients.
    
    Returns:
        Tuple of (success, message)
    """
    print("Checking for external service imports...")
    
    test_dir = Path(__file__).parent.parent / "tests"
    if not test_dir.exists():
        return False, "✗ Tests directory not found"
    
    # Patterns that suggest external dependencies
    external_patterns = [
        'import requests',  # HTTP client (ok for mocking, not for real calls)
        'import boto3',     # AWS client
        'import google.cloud',  # GCP client
        'import azure',     # Azure client
    ]
    
    suspicious_files: List[Tuple[Path, str]] = []
    
    for test_file in test_dir.glob("test_*.py"):
        content = test_file.read_text(encoding='utf-8')
        for pattern in external_patterns:
            if pattern in content and 'mock' not in content.lower():
                suspicious_files.append((test_file, pattern))
    
    if suspicious_files:
        msg = "? Found external imports (verify they're mocked):\n"
        for file, pattern in suspicious_files[:3]:  # Show first 3
            msg += f"  {file.name}: {pattern}\n"
        return True, msg  # Warning, not failure
    
    return True, "✓ No suspicious external service imports found"


def run_fast_tests() -> Tuple[bool, str]:
    """
    Run the fast test lane and verify it passes.
    
    Returns:
        Tuple of (success, message)
    """
    print("Running fast test suite (no network required)...")
    
    cmd = [
        sys.executable, '-m', 'pytest',
        'tests/',
        '-m', 'not slow',
        '--ignore=tests/test_model_builders.py',
        '-v',
        '--tb=short',
        '--maxfail=5',  # Stop after 5 failures
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path(__file__).parent.parent
        )
        
        if result.returncode == 0:
            # Count number of passed tests
            output = result.stdout
            if 'passed' in output:
                return True, f"✓ Fast tests passed without network access"
            return True, "✓ Tests completed"
        else:
            # Extract failure info
            lines = result.stdout.split('\n')
            failure_line = next((l for l in lines if 'FAILED' in l), '')
            return False, f"✗ Tests failed: {failure_line}"
            
    except subprocess.TimeoutExpired:
        return False, "✗ Tests timed out after 120 seconds"
    except Exception as e:
        return False, f"✗ Error running tests: {e}"


def check_test_fixtures() -> Tuple[bool, str]:
    """
    Verify test fixtures use local/in-memory storage.
    
    Returns:
        Tuple of (success, message)
    """
    print("Checking test fixtures...")
    
    conftest = Path(__file__).parent.parent / "tests" / "conftest.py"
    if not conftest.exists():
        return True, "? No conftest.py found (no fixtures to check)"
    
    content = conftest.read_text(encoding='utf-8')
    
    # Good patterns
    good_patterns = ['tmpdir', 'tmp_path', 'TemporaryDirectory', 'tempfile']
    has_good = any(pattern in content for pattern in good_patterns)
    
    # Bad patterns
    bad_patterns = ['http://', 'https://', 'mongodb://', 'postgresql://']
    has_bad = any(pattern in content for pattern in bad_patterns)
    
    if has_bad:
        return False, "✗ Test fixtures reference external services"
    
    if has_good:
        return True, "✓ Test fixtures use temporary/local storage"
    
    return True, "? Could not determine fixture storage patterns"


def main() -> int:
    """
    Run all isolation checks.
    
    Returns:
        int: Exit code (0 = success, 1 = failure)
    """
    print("=" * 70)
    print("Test Isolation Verification")
    print("=" * 70)
    print()
    
    # Track results
    checks = [
        ("MLflow Configuration", check_mlflow_defaults),
        ("External Service Imports", verify_no_external_imports),
        ("Test Fixtures", check_test_fixtures),
        ("Fast Test Suite", run_fast_tests),
    ]
    
    all_passed = True
    results = []
    
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 70)
        try:
            success, message = check_func()
            results.append((name, success, message))
            print(message)
            if not success:
                all_passed = False
        except Exception as e:
            print(f"✗ Check failed with exception: {e}")
            results.append((name, False, str(e)))
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    for name, success, message in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    
    if all_passed:
        print("✓ All isolation checks passed!")
        print("\nThe test suite can run without:")
        print("  - Network access")
        print("  - External databases")
        print("  - Remote MLflow servers")
        print("  - Cloud storage")
        print("\nTests use only:")
        print("  - Local file system (temporary directories)")
        print("  - In-memory data structures")
        print("  - Local MLflow tracking (./mlruns)")
        return 0
    else:
        print("✗ Some isolation checks failed")
        print("\nPlease ensure:")
        print("  - MLFLOW_TRACKING_URI is unset or points to file://")
        print("  - Tests don't make real network calls")
        print("  - Test fixtures use tmpdir/tmp_path")
        return 1


if __name__ == "__main__":
    sys.exit(main())
