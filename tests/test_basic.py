"""Basic tests to verify project structure."""

import pytest


def test_basic_import():
    """Test that we can import the main package."""
    import open_lilli
    assert open_lilli.__version__ == "0.1.0"


def test_placeholder():
    """Placeholder test to ensure pytest works."""
    assert True