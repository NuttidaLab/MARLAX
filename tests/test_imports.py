# tests/test_imports.py
import pytest
import marlax

def test_can_import():
    # basic smoke‐test
    assert hasattr(marlax, "__version__")