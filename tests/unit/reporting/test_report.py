"""REP-01 (separate sections), REP-03 (compute cost), REP-04 (period labels and holdout access log),
REP-05 (variability across seeds), REP-06 (deterministic generation), REP-07 (holdout metrics
record); tasks 5.1, 5.3, 5.3a, 5.4 and 5.5."""

from __future__ import annotations

import copy

import pytest
from finplan_contracts.validate import validate

from finplan_model.core.errors import FinplanError
from finplan_model.reporting import COMPARABILITY_FIELDS, ReportConfig, build_report, report_from_stored, store_report, to_run_result_payload, validate_holdout_record

from .support import benchmark


@pytest.fixture(scope="module")
def bench():
    return benchmark([{"name": "min_variance", "params": {"lookback": 40}}, {"name": "scenario_cvar", "seeds": [1, 2, 3, 4, 5], "params": {"n_scenarios": 100, "lookback": 40}}])


@pytest.fixture(scope="module")
def report(bench):
    return build_report(bench["run"], cost_records=[bench["cost"]], holdout_access=bench["access"])


# --------------------------------------------------------------------- REP-01
def test_classical_report_marks_accuracy_and_rewards_not_applicable(report):
    s = report.content["sections"]
    assert set(s) == {"portfolio_performance", "model_accuracy", "training_rewards", "compute_cost"}
    assert s["model_accuracy"]["status"] == "not_applicable" and set(s["model_accuracy"]["by_strategy"].values()) == {"not_applicable"}
    assert s["training_rewards"]["status"] == "not_applicable"
    assert s["portfolio_performance"]["status"] == "filled" and s["compute_cost"]["status"] == "filled"
    labels = {d["label"] for d in report.content["strategies"]}
    assert labels == {"min_variance", "scenario_cvar", "cash", "buy_and_hold", "equal_weight"}
    assert report.content["auto_added_controls"] == ["cash", "buy_and_hold", "equal_weight"]


@pytest.mark.parametrize("key", ["blended_score", "combined_score", "composite", "score", "score_weights"])
def test_blended_score_request_fails(bench, key):
    with pytest.raises(FinplanError) as ei:
        build_report(bench["run"], config={key: {"net_cumulative_return": 0.5, "directional_hit_rate": 0.5}}, holdout_access=bench["access"])
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["reason"] == "blended_score_refused"
    with pytest.raises(FinplanError):
        ReportConfig.from_dict({key: True})


def test_accuracy_is_only_for_predictive_strategies_and_never_mixes_sections(bench):
    with pytest.raises(FinplanError):
        build_report(bench["run"], holdout_access=bench["access"], accuracy={"min_variance": {"validation": {"directional_hit_rate": 0.5}}})
    doc = copy.deepcopy(bench["run"].to_dict())
    doc["descriptors"]["min_variance"]["makes_predictions"] = True
    ok = build_report(doc, holdout_access=bench["access"], accuracy={"min_variance": {"validation": {"directional_hit_rate": 0.55}}})
    acc = ok.content["sections"]["model_accuracy"]
    assert acc["status"] == "filled" and acc["by_strategy"]["min_variance"] == "filled" and acc["by_strategy"]["cash"] == "not_applicable"
    with pytest.raises(FinplanError):
        build_report(doc, holdout_access=bench["access"], accuracy={"min_variance": {"validation": {"sharpe_ratio": 1.0}}})


# --------------------------------------------------------------------- REP-04
def test_validation_and_holdout_in_separately_labeled_columns_with_access_log(report, bench):
    periods = report.content["sections"]["portfolio_performance"]["periods"]
    assert set(periods) == {"walk_forward_fold", "validation", "holdout"}
    for ptype, rows in periods.items():
        assert rows and all(r["period_type"] == ptype and r["synthetic"] is True for r in rows)
    assert report.content["synthetic"] is True and report.content["period_labels"] == ["walk_forward_fold", "validation", "holdout"]
    h = report.content["holdout"]
    assert h["access_log"] and h["access_log"][0]["run_id"] == bench["ctx"].run_id and h["access_log"][0]["purpose"] == "holdout_evaluation"
    assert h["reused"] is False and h["bounds"]["start"] > periods["validation"][0]["end"]
    with pytest.raises(FinplanError) as ei:
        build_report(bench["run"])
    assert ei.value.details["pointer"] == "/holdout_access"


def test_reused_holdout_is_flagged_in_the_report(bench):
    access = dict(bench["access"], reused=True, reused_families=["classical_optimizer"])
    assert build_report(bench["run"], holdout_access=access).content["holdout"]["reused"] is True


# --------------------------------------------------------------------- REP-05
def test_cvar_with_five_seeds_shows_mean_and_range(report):
    var = report.content["variability"]
    assert set(var) == {"scenario_cvar"} and var["scenario_cvar"]["n_seeds"] == 5
    holdout = var["scenario_cvar"]["periods"]["holdout:holdout"]
    assert holdout["n_seeds"] == 5 and holdout["seeds"] == [1, 2, 3, 4, 5]
    rows = [r for r in report.content["sections"]["portfolio_performance"]["periods"]["holdout"] if r["strategy"] == "scenario_cvar"]
    assert sorted(r["seed"] for r in rows) == [1, 2, 3, 4, 5]  # every seed, not only the best
    for metric, stats in holdout["metrics"].items():
        vals = [r["metrics"][metric] for r in rows]
        assert stats["mean"] == pytest.approx(sum(vals) / 5) and stats["range"] == pytest.approx(max(vals) - min(vals))
    assert "walk_forward_fold:aggregate" in var["scenario_cvar"]["periods"]


# --------------------------------------------------------------------- REP-07
def test_holdout_record_equals_report_holdout_column(report, bench):
    records = bench["run"].holdout_records
    assert len(records) == 1 + 5 + 3  # min_variance, five CVaR seeds, three controls
    rows = {(r["strategy"], r["seed"]): r for r in report.content["sections"]["portfolio_performance"]["periods"]["holdout"]}
    for rec in records:
        row = rows[(rec["strategy_label"], rec["seed"])]
        assert rec["net_cumulative_return"] == row["metrics"]["net_cumulative_return"]
        assert rec["max_drawdown"] == row["metrics"]["max_drawdown"]
        for f in COMPARABILITY_FIELDS:
            assert rec[f] not in (None, "", {})
        assert rec["dataset_id"] == bench["dataset"].dataset_id and rec["holdout"] == bench["run"].holdout_bounds
        assert rec["evaluator_version"] == "1.0.0" and rec["run_id"] == bench["ctx"].run_id
    # records are stored as trusted artifact references in research storage
    refs = bench["run"].holdout_record_refs
    assert len(refs) == len(records) and all(validate(r, "artifact-ref").valid for r in refs)
    assert bench["env"].store.get_json(refs[0])["net_cumulative_return"] == records[0]["net_cumulative_return"]


@pytest.mark.parametrize("field", COMPARABILITY_FIELDS)
def test_holdout_record_missing_comparability_field_fails(bench, field):
    rec = dict(bench["run"].holdout_records[0])
    rec.pop(field)
    with pytest.raises(FinplanError) as ei:
        validate_holdout_record(rec)
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["missing_field"] == field


# --------------------------------------------------------------------- REP-03
def test_cost_section_shows_estimate_and_pending_actual(report, bench):
    (row,) = report.content["sections"]["compute_cost"]["rows"]
    assert row["estimated_cost"] == {"usd": 0.12, "label": "estimated", "source": "pre_flight_estimate"}
    assert row["actual_cost"] == {"status": "pending", "label": "actual"}
    assert row["instance_type"] == "ml.m5.xlarge" and row["instance_count"] == 1 and row["billed_runtime_seconds"] is None
    billed = build_report(bench["run"], holdout_access=bench["access"], cost_records=[dict(bench["cost"], actual_usd=0.04, billed_runtime_seconds=310)])
    (row,) = billed.content["sections"]["compute_cost"]["rows"]
    assert row["actual_cost"] == {"status": "available", "label": "actual", "usd": 0.04} and row["billed_runtime_seconds"] == 310


def test_cost_record_needs_a_run_id_and_estimate(bench):
    with pytest.raises(FinplanError):
        build_report(bench["run"], holdout_access=bench["access"], cost_records=[dict(bench["cost"], estimated_usd=None)])
    with pytest.raises(FinplanError):
        build_report(bench["run"], holdout_access=bench["access"], cost_records=[dict(bench["cost"], run_id="job-1")])


# --------------------------------------------------------------------- REP-06
def test_regenerated_report_has_identical_content_and_checksum(report, bench):
    store = bench["env"].store
    run_ref = store.put_json(bench["run"].to_dict(), kind="run_artifact", synthetic=True)
    again = report_from_stored(run_ref, store, cost_records=[bench["cost"]], holdout_access=bench["access"])
    assert again.content == report.content and again.report_checksum == report.report_checksum
    narrated = report_from_stored(run_ref, store, cost_records=[bench["cost"]], holdout_access=bench["access"], narrative=["Minimum variance had the lowest drawdown."])
    assert narrated.report_checksum == report.report_checksum and "narrative" not in narrated.content
    refs = store_report(narrated, store)
    assert validate(refs["report_ref"], "artifact-ref").valid and "narrative_ref" in refs
    assert store.get_json(refs["report_ref"])["report_checksum"] == report.report_checksum


def test_run_result_payload_sections_validate(report):
    payload = to_run_result_payload(report, "min_variance")
    assert validate(payload, "run-result-payload").valid
    assert payload["accuracy"] == {} and payload["compute_cost"] == {"estimated_usd": 0.12} and payload["synthetic"] is True
