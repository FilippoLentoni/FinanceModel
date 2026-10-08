"""Task 2.9: the documented example configuration validates, and every setting is documented."""

from __future__ import annotations

import json
import re
from pathlib import Path

from finplan_model.core.config import load_config
from finplan_model.sim.config import SimulationConfig

DOC = Path(__file__).resolve().parents[3] / "docs" / "simulator.md"


def _example() -> dict:
    text = DOC.read_text(encoding="utf-8")
    m = re.search(r"<!-- example: simulation-config -->\s*```json\n(.*?)```", text, re.DOTALL)
    assert m, "example block missing"
    return json.loads(m.group(1))


def test_documented_example_validates():
    cfg = SimulationConfig.from_dict(_example())
    assert cfg.constraints.max_weight == 0.4
    assert cfg.configuration_id.startswith("cfg_")


def test_every_setting_and_default_is_documented():
    text = DOC.read_text(encoding="utf-8")
    assert "configuration pending user review" in text and "FM-OQ-5" in text
    defaults = SimulationConfig().to_dict()
    for key in defaults:
        assert f"`{key}" in text, key
    for group in ("fees", "spread", "slippage", "liquidity", "constraints"):
        for sub in defaults[group]:
            assert sub in text, f"{group}.{sub}"
    # The documented defaults are the shipped defaults of every environment.
    for env in ("beta", "gamma", "prod"):
        assert SimulationConfig.from_dict(load_config(env).simulation_defaults) == SimulationConfig()
