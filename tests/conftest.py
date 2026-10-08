"""Shared test configuration. Every offline test runs under the offline harness (``tests/harness.py``):
no network, no real AWS call, no SageMaker call (DEP-02), fake credentials only. The deployed suites
(``FINPLAN_TARGET_ENV`` set by ``scripts/stage_runner.py``) run without it.

Shared fixtures for all task groups:

* ``ctx`` - a :class:`~finplan_model.core.context.RunContext` with a frozen clock and seeded IDs;
* ``clock`` - the frozen clock of ``ctx``;
* ``artifact_store`` - an in-memory :class:`~finplan_model.core.artifacts.ArtifactStore`;
* ``platform`` - a :class:`~finplan_model.core.platform.FixturePlatformClient` (mock provider);
* ``market`` - small deterministic synthetic two-instrument market data (``synthetic=True``).
"""

from __future__ import annotations

import os

import pytest

if not os.environ.get("FINPLAN_TARGET_ENV"):
    # Offline suites only. The deployed suites (tests/integration, tests/smoke), run by
    # scripts/stage_runner.py with FINPLAN_TARGET_ENV set, need the stage role's real credentials
    # (the platform's first pipeline run failed its prod smoke because fake credentials were forced).
    from tests.harness import _offline_harness, pytest_configure, pytest_unconfigure  # noqa: F401  (hooks and autouse fixture)


@pytest.fixture
def ctx():
    from finplan_model.core.context import RunContext

    return RunContext.for_tests(seed=1)


@pytest.fixture
def clock(ctx):
    return ctx.clock


@pytest.fixture
def artifact_store():
    from finplan_model.core.artifacts import InMemoryArtifactStore

    return InMemoryArtifactStore()


@pytest.fixture
def platform(clock):
    from finplan_model.core.platform import FixturePlatformClient

    return FixturePlatformClient(clock=clock)


@pytest.fixture
def market():
    from finplan_model.sim.market import synthetic_market

    return synthetic_market(("AGG", "SPY"), n_sessions=60, seed=11)
