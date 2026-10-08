"""The documented example preparation configuration validates (docs/datasets.md)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from finplan_model.datasets import PreparationConfig

DOC = Path(__file__).resolve().parents[3] / "docs" / "datasets.md"


def test_documented_example_validates():
    m = re.search(r"<!-- example: preparation-config -->\s*```json\n(.*?)```", DOC.read_text(encoding="utf-8"), re.DOTALL)
    assert m, "example block missing"
    cfg = PreparationConfig.from_dict(json.loads(m.group(1)))
    assert cfg.tickers == ("SPY",) and cfg.required_embargo_sessions == 20 and cfg.configuration_id.startswith("cfg_")


def test_documented_contract_gap_is_named():
    text = DOC.read_text(encoding="utf-8")
    assert "FM-A5" in text and "calendar.sessions" in text
