"""Pytest configuration and fixtures."""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def temp_file():
    """Create a temporary file that's cleaned up after the test."""
    files = []

    def _temp_file(suffix=""):
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        files.append(Path(f.name))
        f.close()
        return Path(f.name)

    yield _temp_file

    # Cleanup
    for f in files:
        if f.exists():
            f.unlink()
