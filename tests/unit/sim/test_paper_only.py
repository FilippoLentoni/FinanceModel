"""SIM-07 (paper only, never live) and the live-permission scan; task 2.8."""

from __future__ import annotations

from pathlib import Path

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.sim.config import SimulationConfig
from scripts.build_gates import GateContext, gate_live_perms, live_trading_problems

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("venue", ["coinbase", "alpaca", "live", "interactive_brokers", "binance"])
def test_live_venue_is_operation_not_permitted(venue):
    with pytest.raises(FinplanError) as ei:
        SimulationConfig.from_dict({"venue": venue})
    assert ei.value.code == "OPERATION_NOT_PERMITTED"
    assert ei.value.to_envelope("corr-test-0001")["code"] == "OPERATION_NOT_PERMITTED"


@pytest.mark.parametrize("key", ["broker_api_key", "exchange_account", "wallet", "api_secret"])
def test_live_credentials_in_config_are_refused(key):
    with pytest.raises(FinplanError) as ei:
        SimulationConfig.from_dict({"venue": "paper", "liquidity": {key: "x"}})
    assert ei.value.code == "OPERATION_NOT_PERMITTED"


def test_paper_venue_is_accepted():
    assert SimulationConfig.from_dict({"venue": "paper"}).venue == "paper"


def test_live_permission_scan_passes_on_repository():
    assert gate_live_perms(GateContext(root=ROOT)) == []


def test_planted_live_trading_dependency_fails(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndependencies = ["ccxt>=4"]\n', encoding="utf-8")
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "trade.py").write_text("import alpaca\n", encoding="utf-8")
    (tmp_path / "Dockerfile").write_text("RUN pip install robin-stocks\n", encoding="utf-8")
    problems = live_trading_problems(tmp_path)
    assert any("ccxt" in p for p in problems)
    assert any("imports live-trading client alpaca" in p for p in problems)
    assert any("robin-stocks" in p for p in problems)


def test_planted_live_iam_permission_fails(tmp_path):
    (tmp_path / "policy.json").write_text('{"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": "payments:*", "Resource": "*"}]}', encoding="utf-8")
    assert gate_live_perms(GateContext(root=tmp_path))
