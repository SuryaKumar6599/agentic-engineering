"""
conftest.py
───────────
Pytest shared fixtures available across all test modules.
"""

from __future__ import annotations

import pytest
from agents.example_agent.agent import run as _example_agent_run


@pytest.fixture
def run_agent():
    """
    Returns the agent-under-test.

    Override at the CLI or in a conftest closer to your tests:
        --agent=my_package.agent.run
    """
    return _example_agent_run


@pytest.fixture(autouse=True)
def reset_cost_accumulator():
    """Reset the per-process cost accumulator before each test."""
    from harness.cost import run_cost
    run_cost.reset()
    yield
    # post-test: cost is captured by individual tests if needed
