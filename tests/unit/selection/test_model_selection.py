"""Decision 27 offline model selection: protocol rules (RL-05 validation-only selection), one common
evaluator for every family, per-split comparisons, chosen hyperparameters, RL seed spread, the separate
training-reward section (RL-06), the test lock (test evaluated once, after the frozen selection), the
promotion check of criteria v1 and the mandatory caveats. Tiny synthetic data and fixture step counts."""

from __future__ import annotations

import copy
import json
import math

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.selection.job import SplitEvaluator, restrict_market, select_models
from finplan_model.selection.protocol import (
    DEFAULT_PROTOCOL,
    validate_protocol,
    validate_selection_rule,
)
from tests.unit.rl.support import market, sim_config, tiny_protocol

FAMILIES = {"cash": "control", "buy_and_hold": "control", "equal_weight": "control", "min_variance": "traditional", "mean_variance": "traditional", "scenario_cvar": "traditional", "ppo": "rl", "sac": "rl"}


def daily_config():
    """The beta simulation defaults with daily rebalancing, the schedule model selection uses (decision 28)."""
    import dataclasses

    return dataclasses.replace(sim_config(), rebalance_frequency="daily")


@pytest.fixture(scope="module")
def outcome():
    return select_models(market(), daily_config(), protocol=tiny_protocol())


# ----------------------------------------------------------------- protocol
def test_default_protocol_is_decision_27():
    p = validate_protocol(DEFAULT_PROTOCOL)
    assert p["data_start"] == "2025-01-01"
    assert p["splits"] == {"train": {"start": "2025-01-01", "end": "2025-12-31"}, "validation": {"start": "2026-01-01", "end": "2026-06-30"}, "test": {"start": "2026-07-01", "end": None}}
    assert set(p["traditional"]) == {"min_variance", "mean_variance", "scenario_cvar"}
    assert p["rl"]["algorithms"] == ["ppo", "sac"] and p["rl"]["seeds"] == [0, 1, 2, 3, 4]
    assert p["selection"] == {"metric": "sharpe", "split": "validation"}


@pytest.mark.parametrize("rule", [{"metric": "test_sharpe"}, {"metric": "sharpe", "split": "test"}, {"metric": "holdout_return"}, {"split": "holdout"}])
def test_rl05_selection_rules_referencing_test_or_holdout_are_refused(rule):
    with pytest.raises(FinplanError) as exc:
        validate_selection_rule(rule)
    assert exc.value.code == "OPERATION_NOT_PERMITTED"
    with pytest.raises(FinplanError) as exc2:
        validate_protocol({**copy.deepcopy(DEFAULT_PROTOCOL), "selection": rule})
    assert exc2.value.code == "OPERATION_NOT_PERMITTED"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["splits"]["validation"].update(start="2025-12-01"),  # overlaps train
        lambda p: p.update(data_start="2025-06-01"),  # train starts before the data range
        lambda p: p["rl"].update(seeds=[0, 1]),  # fewer than 3 seeds
        lambda p: p["rl"].update(algorithms=["dqn"]),
        lambda p: p["rl"].update(grid={"learning_rate": [0.1]}),  # RL grid ranges over reward terms only
        lambda p: p["traditional"]["mean_variance"]["grid"].update(lookback=[1]),  # invalid optimizer parameter
        lambda p: p["traditional"].update(black_litterman={"grid": {"tau": [1]}}),
        lambda p: p["rl"]["ppo"].update(total_timesteps=0),
        lambda p: p.update(incumbent_fallback="ppo"),
    ],
)
def test_invalid_protocols_are_refused(mutate):
    p = copy.deepcopy(DEFAULT_PROTOCOL)
    mutate(p)
    with pytest.raises(FinplanError) as exc:
        validate_protocol(p)
    assert exc.value.code == "VALIDATION_FAILED"


def test_data_outside_the_protocol_range_is_invisible():
    m = restrict_market(market(), market().sessions[10], market().sessions[40])
    assert m.sessions[0] == market().sessions[10] and m.sessions[-1] == market().sessions[40]
    assert all(m.sessions[0] <= b.session_date <= m.sessions[-1] for i in m.instruments for b in m.bars_of(i))
    assert m.dataset_checksum == market().dataset_checksum


def test_the_test_window_opens_only_after_the_selection_is_frozen():
    from finplan_model.strategies import build_strategy

    mkt = market()
    ev = SplitEvaluator(mkt, daily_config(), None, list(mkt.instruments), {"validation": (mkt.sessions[10], mkt.sessions[40]), "test": (mkt.sessions[41], mkt.sessions[80])})
    with pytest.raises(FinplanError) as exc:
        ev.run(build_strategy("equal_weight"), "test")
    assert exc.value.code == "OPERATION_NOT_PERMITTED"
    checksum = ev.open_test({"selected": "equal_weight"})
    assert checksum.startswith("sha256:")
    ev.run(build_strategy("equal_weight"), "test")
    with pytest.raises(FinplanError):
        ev.open_test({"selected": "something else"})  # the selection is frozen once


# ----------------------------------------------------------------- end to end on tiny data
def test_every_family_is_compared_on_every_split_with_one_evaluator(outcome):
    comp = outcome.summary["comparison"]
    assert set(comp) == {"train", "validation", "test"}
    for split, section in comp.items():
        rows = section["strategies"]
        assert {r["strategy"]: r["family"] for r in rows} == FAMILIES
        assert {r["strategy"]: r["role"] for r in rows}["ppo"] == "rl"
        assert sum(r["selected"] for r in rows) == 1 and sum(r["incumbent"] for r in rows) == 1
        for r in rows:
            assert abs(sum(r["final_weights"].values()) + r["final_cash_weight"] - 1.0) < 1e-5 or r["strategy"] == "cash"
            assert all(w <= 1.0 + 1e-9 and w >= -1e-12 for w in r["final_weights"].values())  # long-only, max weight 1
    ev = outcome.summary["evaluation"]
    assert ev["rebalance_frequency"] == "daily" and ev["long_only"] is True and ev["max_weight"] == 1.0
    assert outcome.test_section is comp["test"]
    splits = outcome.summary["data"]["splits"]
    assert splits["train"]["end"] < splits["validation"]["start"] <= splits["validation"]["end"] < splits["test"]["start"]


def test_traditional_hyperparameters_are_chosen_on_validation(outcome):
    trad = outcome.summary["traditional"]
    for name, block in trad.items():
        scores = [g["validation"]["sharpe"] if g["validation"]["sharpe"] is not None else -math.inf for g in block["grid"]]
        assert block["chosen_grid_index"] == scores.index(max(scores))
        assert outcome.summary["selection"]["chosen_hyperparameters"][name] == block["chosen"]
    assert "lookback" in trad["min_variance"]["chosen"] and "risk_aversion" in trad["mean_variance"]["chosen"]


def test_rl_seeds_are_all_reported_with_their_spread_and_the_validation_choice(outcome):
    for algo in ("ppo", "sac"):
        block = outcome.summary["rl"][algo]
        assert block["status"] == "trained" and block["deterministic_evaluation"] is True
        seeds = block["seeds"]
        assert [s["seed"] for s in seeds] == [0, 1, 2]
        assert all({"train", "validation", "test"} <= set(s) for s in seeds)
        chosen = [s for s in seeds if s["selected"]]
        assert len(chosen) == 1 and chosen[0]["seed"] == block["chosen"]["seed"]
        best = max(seeds, key=lambda s: s["validation"]["sharpe"] if s["validation"]["sharpe"] is not None else -math.inf)
        assert chosen[0]["validation"]["sharpe"] == best["validation"]["sharpe"]
        stats = block["seed_statistics"]["test"]["total_return"]
        values = [s["test"]["total_return"] for s in seeds]
        assert stats["n"] == 3 and stats["min"] == min(values) and stats["max"] == max(values)
        assert stats["mean"] == pytest.approx(sum(values) / 3, abs=1e-6) and stats["std"] >= 0
        assert block["environment"]["reward_formula"].startswith("reward_t = reward_scale")
        assert block["grid"][0]["checkpoints"] and block["grid"][0]["configuration_id"].startswith("cfg_")


def test_rl06_training_rewards_are_a_separate_section(outcome):
    tr = outcome.summary["training_reward"]
    assert "never as portfolio performance" in tr["note"]
    for algo, runs in tr["runs"].items():
        assert len(runs) == 3
        for r in runs:
            assert set(r) <= {"reward", "seed", "status", "training_reward", "timesteps_trained", "best_step", "stopped_reason", "seconds"}
    text = json.dumps(outcome.summary["comparison"])
    assert "episode_reward" not in text and "reward" not in json.dumps([r["metrics"] for r in outcome.test_section["strategies"]])


def test_selection_is_frozen_before_test_and_test_runs_once(outcome):
    sel = outcome.summary["selection"]
    assert sel["selection_checksum"].startswith("sha256:")
    ranking = sel["validation_ranking"]
    eligible = [r for r in ranking if r["strategy"] != sel["incumbent"]]
    assert sel["selected"]["strategy"] == eligible[0]["strategy"] == outcome.selected.name
    access = outcome.summary["test_access"]
    # one test evaluation per family model plus the non-selected seeds of the chosen RL configurations
    assert access["split_evaluations"]["test"] == 8 + 2 * 2 and access["test_reuse"] is False


def test_promotion_check_criteria_v1_against_the_incumbent(outcome):
    gate = outcome.summary["promotion_check"]
    assert gate["incumbent"] == "buy_and_hold" and gate["incumbent_source"].startswith("fallback")
    assert gate["requires_user_approval"] is True and gate["result"] in ("pass", "fail")
    expect = gate["candidate_test"]["total_return"] > gate["incumbent_test"]["total_return"] and gate["candidate_test"]["max_drawdown"] <= gate["incumbent_test"]["max_drawdown"]
    assert (gate["result"] == "pass") == expect


def test_caveats_flag_thin_rl_data_and_hindsight_bias(outcome):
    kinds = {c["kind"]: c["text"] for c in outcome.summary["caveats"]}
    assert "250 training days is thin for reinforcement learning" in kinds["thin_rl_training_data"]
    assert "hindsight" in kinds["hindsight_and_survivorship"].lower() and "survivorship" in kinds["hindsight_and_survivorship"].lower()
    assert {"short_single_test_period", "validation_reuse", "reward_is_not_performance"} <= set(kinds)


def test_policies_are_stored_with_checksum_seed_and_configuration_id(outcome):
    assert len(outcome.policies) == 6  # 2 algorithms x 1 reward configuration x 3 seeds
    for meta, data in outcome.policies:
        assert meta["checksum"].startswith("sha256:") and meta["configuration_id"].startswith("cfg_") and isinstance(meta["seed"], int)
        assert data[:2] == b"PK"


def test_time_budget_skips_rl_runs_and_says_so():
    clock = iter([0.0] + [1e9] * 10_000)
    out = select_models(market(), daily_config(), protocol=tiny_protocol(), deadline=1.0, clock=lambda: next(clock))
    assert out.summary["rl"]["ppo"]["status"] == "not_trained_time_budget"
    assert {c["kind"] for c in out.summary["caveats"]} >= {"rl_time_budget"}
    assert {r["strategy"] for r in out.test_section["strategies"]} == set(FAMILIES) - {"ppo", "sac"}


def test_a_production_strategy_is_the_incumbent_when_it_is_comparable():
    p = tiny_protocol()
    p["rl"]["algorithms"] = []
    with pytest.raises(FinplanError):
        validate_protocol(p)  # at least one RL algorithm is part of the protocol
    p = tiny_protocol()
    p["rl"]["algorithms"] = ["ppo"]
    out = select_models(market(), daily_config(), protocol=p, incumbent="equal_weight", prior_test_accesses=1)
    gate = out.summary["promotion_check"]
    assert gate["incumbent"] == "equal_weight" and gate["incumbent_source"] == "production_strategy"
    assert out.summary["test_access"]["test_reuse"] is True
