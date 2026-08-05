"""Pytest configuration and shared fixtures for FDS Visualizer tests."""

import os
import pytest
from PyQt5 import QtWidgets

import isolation


# Must set before any Qt imports in the test modules themselves
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def pytest_configure(config):
    """Fail the session before a single test runs if code from another
    checkout is in play (see tests/isolation.py for why this exists).

    Deliberately fail-fast rather than a test: by the time an ordinary
    test executes, the contaminated module has already been imported and
    the suite may report green while running someone else's code.
    """
    root = isolation.project_root()
    report = isolation.fatal_report(root)
    if report:
        raise pytest.UsageError(report)
    warning = isolation.warning_report(root)
    if warning:
        config.stash[_WARNING_KEY] = warning


_WARNING_KEY = pytest.StashKey[str]()


def pytest_report_header(config):
    return config.stash.get(_WARNING_KEY, None)


def pytest_collection_modifyitems(session, config, items):
    """Second gate: every collected test file must live in this checkout.
    Catches the stale-bytecode case, where a copied __pycache__ made
    pytest execute the sibling's test module."""
    root = isolation.project_root()
    paths = {str(item.path) for item in items if getattr(item, "path", None)}
    report = isolation.fatal_report(root, test_paths=sorted(paths))
    if report:
        raise pytest.UsageError(report)


@pytest.fixture(scope="session")
def qapp():
    """Create a QApplication for all tests."""
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


@pytest.fixture
def fixtures_dir():
    """Path to the test fixtures directory."""
    return os.path.join(os.path.dirname(__file__), "fixtures", "c1_d0_vod0_voc0")
