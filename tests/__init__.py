"""
Test suite for pyduck-janitor package.
"""

import sys
from pathlib import Path

import pytest

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
