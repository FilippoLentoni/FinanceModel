"""Offline suite (``tests/contract``): always hermetic, even when ``FINPLAN_TARGET_ENV`` is set
(mirrors the platform's ``tests/contract/conftest.py``; platform lesson L5).

``tests/conftest.py`` installs the offline harness (no network, no real AWS, no SageMaker, fake
credentials) only for offline runs, so the deployed suites keep the stage role's real credentials.
This directory re-installs it unconditionally: an offline test never runs with real credentials.
Run deployed suites on their own directory (as ``scripts/stage_runner.py`` does).
"""

from tests.harness import _offline_harness, install  # noqa: F401  (autouse fixture for this directory)

install()
