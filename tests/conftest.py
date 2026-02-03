"""
Pytest configuration and shared fixtures.

This file is automatically discovered by pytest and provides fixtures
that can be used across all test files.
"""

import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def device():
    """Provide test device (CPU for CI compatibility)."""
    return "cpu"
