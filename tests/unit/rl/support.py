"""Tiny synthetic fixtures for the RL and model-selection tests (all data synthetic; nothing retrieved).

The tiny protocol keeps the real structure of decision 27 (train < validation < test, three control,
three traditional and two RL families, at least three seeds) on a few months of synthetic sessions and
a few dozen learner steps, so the whole RL test set stays well under 30 seconds and below the offline
harness's learner step limit."""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from typing import Any

from finplan_model.selection.protocol import DEFAULT_PROTOCOL
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.market import MarketData, synthetic_market

ROOT = Path(__file__).resolve().parents[3]
TICKERS = ("VOO", "GOOGL", "NFLX", "AAPL", "NVDA")
START = date(2025, 1, 1)


def sim_config() -> SimulationConfig:
    return SimulationConfig.from_dict(json.loads((ROOT / "config" / "beta.json").read_text())["simulation_defaults"])


def market(n_sessions: int = 170, seed: int = 3) -> MarketData:
    return synthetic_market(TICKERS, n_sessions=n_sessions, start=START, seed=seed)


def tiny_protocol(**over: Any) -> dict[str, Any]:
    p = copy.deepcopy(DEFAULT_PROTOCOL)
    p["splits"] = {"train": {"start": "2025-01-01", "end": "2025-04-30"}, "validation": {"start": "2025-05-01", "end": "2025-06-30"}, "test": {"start": "2025-07-01", "end": None}}
    p["traditional"]["min_variance"]["grid"] = {"lookback": [20, 60]}
    p["traditional"]["mean_variance"]["grid"] = {"lookback": [20], "risk_aversion": [1.0, 20.0]}
    p["traditional"]["scenario_cvar"]["grid"] = {"lookback": [40], "alpha": [0.9]}
    p["traditional"]["scenario_cvar"]["fixed"] = {"n_scenarios": 50, "scenario_method": "bootstrap", "seed": 0}
    p["rl"]["seeds"] = [0, 1, 2]
    p["rl"]["env"]["window"] = 5
    p["rl"]["env"]["step_sessions"] = 5
    p["rl"]["grid"] = {"risk_penalty": [0.5]}
    p["rl"]["ppo"] = {"total_timesteps": 32, "n_steps": 16, "batch_size": 8, "n_epochs": 1, "eval_every": 16, "patience": 2, "net_arch": [8, 8], "gamma": 0.9}
    p["rl"]["sac"] = {"total_timesteps": 28, "learning_starts": 16, "batch_size": 8, "buffer_size": 500, "eval_every": 14, "patience": 2, "net_arch": [8, 8], "gamma": 0.9}
    p["runtime"] = {"reserve_seconds": 0}
    p.update(over)
    return p
