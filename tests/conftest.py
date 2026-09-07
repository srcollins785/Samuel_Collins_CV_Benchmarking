"""Shared pytest configuration.

pytest imports this before collecting any tests, which makes it the right
place for the two things every test file needs: the package on the import
path, and the location of the committed fixtures.

Putting ``src`` on ``sys.path`` lets the suite run in a bare checkout without
``pip install`` first. The published package is still imported by its real
name, so the tests exercise the same import path a user gets from PyPI.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture
def fixtures_dir() -> Path:
    """Absolute path to tests/fixtures.

    Tests resolve their data through this rather than a relative path, so the
    suite passes no matter which directory pytest was invoked from.
    """
    return Path(__file__).resolve().parent / "fixtures"
