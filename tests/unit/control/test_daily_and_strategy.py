"""Change add-daily-recommendation-and-on-demand-experiments: PSS-01..03 (production-strategy setting),
DRJ-01..03 and DRJ-05 (daily_recommendation submission and cost), UNV-03 (trigger-role kind
restriction, no schedule) and the job API resource policy for the platform trigger. Offline."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from finplan_contracts.iam import Request, evaluate
from finplan_contracts.validate import validate

from finplan_model.control.auth import Principal
from finplan_model.control.policies import job_api_resource_policy
from finplan_model.control.production_strategy import InMemoryStrategyParameter, StrategySelectionService, eligibility, parse_document
from finplan_model.control.selection import SelectionApi
from finplan_model.core.config import JobTypeConfig, daily_cost_problems, load_config
from tests.unit.control.support import SUBMITTER, Harness, arn, request

ROOT = Path(__file__).resolve().parents[3]
TRIGGER = arn("finplan-beta-financialplanning-daily-trigger-step-role")
OPERATOR = arn("finplan-beta-financialplanning-operator-pipeline-stage")
PLAN_WRITER = arn("finplan-beta-financelambdastool-plan-writer-role")
PLAN_ID = "pl_01KDVDNAZ83BAMMYCEGWF33DPM"
UNIVERSE = "finance/equity-etf-daily/research-universe"


def _evidence_run(strategy: str = "min_variance", job_type: str = "run_benchmark", state: str = "succeeded", dataset: str = UNIVERSE, purpose: str = "research") -> dict:
    return {
        "run_id": "run_01KDVDNBYGX5V5HY2JSK5XWKHC",
        "environment": "beta",
        "state": state,
        "job_type": job_type,
        "purpose": purpose,
        "dataset_id": dataset,
        "configuration": {"payload": {"strategy": strategy}},
        "submitted_at": "2026-01-04T09:00:00Z",
        "completed_at": "2026-01-04T09:20:00Z",
        "revision": 3,
        "compute_class": "cpu",
    }


def _seed(h: Harness, run: dict) -> None:
    h.store.create_run(run, [{"run_id": run["run_id"], "seq": 0, "from_state": None, "state": run["state"], "at": run["submitted_at"]}])


def _selection(h: Harness, value: str | None = None) -> tuple[StrategySelectionService, InMemoryStrategyParameter, list]:
    param = InMemoryStrategyParameter(value)
    audit: list = []
    return StrategySelectionService(h.cfg, param, h.store, h.clock, h.ids, audit_log=audit.append), param, audit


def _daily(**over):
    base = {"job_type": "daily_recommendation", "purpose": "production_candidate", "plan_id": PLAN_ID, "evaluation_window": None, "idempotency_key": "daily-beta-2026-01-05"}
    body = request(**{**base, **over})
    body["configuration"]["payload"].update(strategy="min_variance", objective="daily_recommendation")
    return body


def _set(svc, principal=OPERATOR, **over):
    body = {"action": "set", "strategy_id": "min_variance", "idempotency_key": "sel-0001", "confirmed_by_user": True, **over}
    return svc.handle(Principal.from_arn(principal), {k: v for k, v in body.items() if v is not None})


# ===================================================================== PSS-01
def test_pss01_get_when_nothing_selected_is_none():
    h = Harness()
    svc, _, _ = _selection(h)
    resp = svc.get()
    assert resp["strategy"] is None and resp["changed"] is False
    assert validate(resp, "tools/production-strategy-response").valid


def test_pss01_set_requires_confirmation_and_leaves_the_key_unchanged():
    h = Harness()
    _seed(h, _evidence_run())
    svc, param, audit = _selection(h)
    with pytest.raises(Exception) as ei:
        _set(svc, confirmed_by_user=None)
    assert ei.value.code == "PRECONDITION_FAILED" and ei.value.details["reason"] == "confirmation_required"
    assert param.value is None and param.writes == 0 and not audit


@pytest.mark.parametrize("principal", [SUBMITTER, TRIGGER, arn("finplan-beta-financeagent-runtime-role")])
def test_pss01_only_plan_writer_or_operator_may_set(principal):
    h = Harness()
    _seed(h, _evidence_run())
    svc, param, _ = _selection(h)
    with pytest.raises(Exception) as ei:
        _set(svc, principal=principal)
    assert ei.value.code == "FORBIDDEN" and param.value is None


def test_pss01_set_is_idempotent_and_clear_removes_the_key():
    h = Harness()
    _seed(h, _evidence_run())
    svc, param, audit = _selection(h)
    code, first = _set(svc, principal=PLAN_WRITER, on_behalf_of="test-user")
    assert code == 200 and first["changed"] is True
    again = _set(svc, principal=PLAN_WRITER, on_behalf_of="test-user")[1]
    assert again == first and param.writes == 1
    with pytest.raises(Exception) as ei:
        _set(svc, principal=PLAN_WRITER, strategy_id="equal_weight")  # same key, other request
    assert ei.value.code == "IDEMPOTENCY_KEY_REUSED"
    code, cleared = svc.handle(Principal.from_arn(OPERATOR), {"action": "clear", "idempotency_key": "clr-0001", "confirmed_by_user": True})
    assert code == 200 and cleared["changed"] is True and cleared["strategy"] is None and param.value is None
    assert svc.get()["strategy"] is None


# ===================================================================== PSS-02
def test_pss02_valid_selection_writes_a_contract_document_citing_the_evidence():
    h = Harness()
    _seed(h, _evidence_run())
    svc, param, _ = _selection(h)
    _, resp = _set(svc)
    doc = json.loads(param.value)
    assert validate(doc, "production-strategy").valid
    assert doc["strategy_id"] == "min_variance" and doc["environment"] == "beta"
    assert doc["selected_by"] == "finplan-beta-financialplanning-operator-pipeline-stage" and doc["selected_at"]
    assert "run_01KDVDNBYGX5V5HY2JSK5XWKHC" in doc["note"]
    assert resp["strategy"] == doc and validate(resp, "tools/production-strategy-response").valid


@pytest.mark.parametrize(
    ("strategy", "run", "rule"),
    [
        ("scenario_cvar", None, "no_evaluation_evidence"),
        ("min_variance", _evidence_run(state="failed"), "no_evaluation_evidence"),
        ("min_variance", _evidence_run(dataset="finance/etf-daily/SPY"), "no_evaluation_evidence"),
        ("min_variance", _evidence_run(purpose="production_candidate"), "no_evaluation_evidence"),
        ("not_a_strategy", None, "strategy_not_registered"),
    ],
)
def test_pss02_each_failing_rule_is_named(strategy, run, rule):
    h = Harness()
    if run:
        _seed(h, run)
    svc, param, _ = _selection(h)
    with pytest.raises(Exception) as ei:
        _set(svc, strategy_id=strategy)
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["rule"] == rule
    assert param.value is None


def test_pss02_controls_evaluated_in_a_universe_benchmark_count_as_evidence():
    h = Harness()
    _seed(h, _evidence_run(strategy="min_variance", job_type="run_benchmark"))
    assert eligibility("buy_and_hold", cfg=h.cfg, runs=[h.store.get_run("run_01KDVDNBYGX5V5HY2JSK5XWKHC")]).ok


def test_pss02_retired_and_learning_rules(monkeypatch):
    from finplan_model.control import production_strategy as ps

    h = Harness()
    runs = [_evidence_run()]
    monkeypatch.setattr(ps, "RETIRED_STRATEGIES", frozenset({"min_variance"}))
    assert eligibility("min_variance", cfg=h.cfg, runs=runs).rule == "strategy_retired"
    monkeypatch.setattr(ps, "RETIRED_STRATEGIES", frozenset())
    monkeypatch.setattr(ps, "_family", lambda s: "rl")
    assert eligibility("min_variance", cfg=h.cfg, runs=runs, model_version="mv_x", registry_status=lambda mv: "registered").rule == "strategy_not_promoted"
    assert eligibility("min_variance", cfg=h.cfg, runs=runs, model_version="mv_x", registry_status=lambda mv: "promoted").ok
    monkeypatch.setattr(ps, "supported_datasets", lambda s: ("etf-daily",))
    assert eligibility("min_variance", cfg=h.cfg, runs=runs).rule == "dataset_not_supported"


# ===================================================================== PSS-03 audit
def test_pss03_every_change_is_audited_with_old_new_user_and_channel():
    h = Harness()
    _seed(h, _evidence_run())
    _seed(h, {**_evidence_run(strategy="equal_weight"), "run_id": "run_01KDVDP88REHGPBXFX6CHX92KS"})
    svc, _, audit = _selection(h)
    _set(svc, principal=PLAN_WRITER, on_behalf_of="test-user")
    _set(svc, principal=OPERATOR, strategy_id="equal_weight", idempotency_key="sel-0002")
    svc.handle(Principal.from_arn(OPERATOR), {"action": "clear", "idempotency_key": "clr-0001", "confirmed_by_user": True})
    assert [a["action"] for a in h.store.audit] == ["set", "set", "clear"] == [a["action"] for a in audit]
    first, second, third = h.store.audit
    assert first["old_value"] is None and first["new_value"]["strategy_id"] == "min_variance" and first["user"] == "test-user" and first["channel"] == "tool"
    assert second["old_value"]["strategy_id"] == "min_variance" and second["new_value"]["strategy_id"] == "equal_weight" and second["channel"] == "operator"
    assert third["old_value"]["strategy_id"] == "equal_weight" and third["new_value"] is None
    assert "arn:" not in json.dumps(h.store.audit)


def test_selection_api_routes_and_envelopes():
    h = Harness()
    svc, _, _ = _selection(h)
    api = SelectionApi(svc)

    def call(method, body=None, principal=OPERATOR):
        ev = {"httpMethod": method, "path": "/v1/production-strategy", "headers": {}, "body": None if body is None else json.dumps(body), "requestContext": {"identity": {"userArn": principal}}}
        r = api.handle(ev)
        return r["statusCode"], json.loads(r["body"])

    assert call("GET") == (200, svc.get())
    code, body = call("PUT", {"action": "set", "strategy_id": "min_variance", "idempotency_key": "k1", "confirmed_by_user": True})
    assert code == 400 and body["code"] == "VALIDATION_FAILED" and body["details"]["rule"] == "no_evaluation_evidence"
    code, body = call("PUT", {"action": "set", "strategy_id": "min_variance", "idempotency_key": "k1"})
    assert code == 422 and body["code"] == "PRECONDITION_FAILED"
    code, body = call("PUT", {"action": "explode"})
    assert code == 400


def test_parse_document_treats_absent_empty_and_garbage_as_none():
    assert parse_document(None) is None and parse_document("") is None and parse_document("  ") is None
    assert parse_document("{not json") is None and parse_document("{}") is None and parse_document('{"strategy_id": ""}') is None
    assert parse_document('{"strategy_id": "cash"}') == {"strategy_id": "cash"}


# ===================================================================== DRJ-01..03, UNV-03
def _selected(h: Harness, strategy: str = "min_variance") -> None:
    _seed(h, _evidence_run(strategy=strategy))
    h.settings.strategy_raw = json.dumps({"strategy_id": strategy, "environment": "beta", "selected_at": "2026-01-04T10:00:00Z", "selected_by": "finplan-beta-financialplanning-operator-pipeline-stage"})


def test_drj01_tool_role_cannot_submit_daily_recommendation():
    h = Harness()
    _selected(h)
    with pytest.raises(Exception) as ei:
        h.service.submit_job(Principal.from_arn(SUBMITTER), _daily(), correlation_id="corr-test-daily-0001")
    assert ei.value.code == "FORBIDDEN"
    assert h.store.query_runs()[0] == [h.store.get_run("run_01KDVDNBYGX5V5HY2JSK5XWKHC")]


@pytest.mark.parametrize("job_type", ["run_benchmark", "run_backtest", "prepare_dataset", "report"])
def test_unv03_trigger_role_gets_forbidden_for_any_other_kind(job_type):
    h = Harness()
    with pytest.raises(Exception) as ei:
        h.service.submit_job(Principal.from_arn(TRIGGER), request(job_type=job_type), correlation_id="corr-test-daily-0002")
    assert ei.value.code == "FORBIDDEN"


def test_drj02_runs_only_the_configured_strategy_and_freezes_it():
    h = Harness()
    _selected(h)
    code, resp = h.service.submit_job(Principal.from_arn(TRIGGER), _daily(), correlation_id="corr-test-daily-0003")
    assert code == 202 and resp["state"] == "queued", resp  # under its own USD 0.15 auto-approve ceiling
    run = h.store.get_run(resp["run_id"])
    assert run["production_strategy"]["strategy_id"] == "min_variance" and run["production_strategy"]["configuration_id"] == resp["configuration_id"]
    assert run["plan_id"] == PLAN_ID and run["max_runtime_seconds"] == 1800 and run["instance_type"] == "ml.m5.xlarge" and run["budget_category"] == "cpu_research"
    # replay by idempotency key: same run
    assert h.service.submit_job(Principal.from_arn(TRIGGER), _daily(), correlation_id="corr-test-daily-0004")[1]["run_id"] == resp["run_id"]
    # the spec carries the frozen triple and the staging target
    h.service.dispatch()
    (req,) = h.created_jobs()
    assert req["StoppingCondition"]["MaxRuntimeInSeconds"] == 1800 and req["ProcessingResources"]["ClusterConfig"]["InstanceType"] == "ml.m5.xlarge"
    spec = h.run_io.get_spec(resp["run_id"], None)
    assert spec["staging"] == {"plan_id": PLAN_ID} and spec["production_strategy"]["strategy_id"] == "min_variance"


def test_drj02_override_attempt_fails_naming_strategy_id():
    h = Harness()
    _selected(h, "min_variance")
    body = _daily()
    body["configuration"]["payload"]["strategy"] = "mean_variance"
    with pytest.raises(Exception) as ei:
        h.service.submit_job(Principal.from_arn(TRIGGER), body, correlation_id="corr-test-daily-0005")
    assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["field"] == "strategy_id"


@pytest.mark.parametrize("raw", [None, "", "{oops", json.dumps({"strategy_id": "scenario_cvar", "environment": "beta", "selected_at": "2026-01-04T10:00:00Z", "selected_by": "x"})])
def test_drj03_no_or_ineligible_strategy_means_no_run(raw):
    h = Harness()
    h.settings.strategy_raw = raw
    with pytest.raises(Exception) as ei:
        h.service.submit_job(Principal.from_arn(TRIGGER), _daily(), correlation_id="corr-test-daily-0006")
    assert ei.value.code == "PRECONDITION_FAILED"
    assert ei.value.details["reason"] == ("strategy_not_eligible" if raw and raw.startswith('{"') else "no_production_strategy")
    h.service.dispatch()
    assert h.store.query_runs()[0] == [] and h.created_jobs() == []


def test_drj03_dry_run_also_refuses_without_strategy():
    h = Harness()
    with pytest.raises(Exception) as ei:
        h.service.submit_job(Principal.from_arn(TRIGGER), _daily(dry_run=True), correlation_id="corr-test-daily-0007")
    assert ei.value.details["reason"] == "no_production_strategy"


def test_daily_needs_production_candidate_and_plan_id():
    h = Harness()
    _selected(h)
    for body, ptr in ((_daily(purpose="research"), "/purpose"), (_daily(plan_id=None), "/plan_id")):
        with pytest.raises(Exception) as ei:
            h.service.submit_job(Principal.from_arn(TRIGGER), body, correlation_id="corr-test-daily-0008")
        assert ei.value.code == "VALIDATION_FAILED" and ei.value.details["pointer"] == ptr


def test_daily_budget_exceeded_is_reported():
    h = Harness(allocation={"cpu_research": 0.05, "gpu": 25})
    _selected(h)
    with pytest.raises(Exception) as ei:
        h.service.submit_job(Principal.from_arn(TRIGGER), _daily(), correlation_id="corr-test-daily-0009")
    assert ei.value.code == "BUDGET_EXCEEDED"
    assert [r["run_id"] for r in h.store.query_runs()[0]] == ["run_01KDVDNBYGX5V5HY2JSK5XWKHC"]


def test_platform_daily_submission_fixture_is_accepted_shape():
    """The contract's own daily-recommendation submission fixture validates against job-submission."""
    from finplan_contracts import schemas

    root = Path(schemas.__file__).parent / "data" / "fixtures" / "job-submission" / "valid" / "daily-recommendation.json"
    body = json.loads(root.read_text())
    assert body["job_type"] == "daily_recommendation" and body["purpose"] == "production_candidate" and body["plan_id"].startswith("pl_")


# ===================================================================== DRJ-05
def test_drj05_build_cost_check():
    for env in ("beta", "gamma", "prod"):
        cfg = load_config(env)
        jt = cfg.job_type("daily_recommendation")
        assert jt is not None and daily_cost_problems(jt) == []
        assert jt.cost_check_usd_per_hour * jt.max_runtime_seconds / 3600 <= 0.15
    raw = dict(load_config("beta").raw["job_types"]["daily_recommendation"])
    low = JobTypeConfig.from_dict("daily_recommendation", {**raw, "auto_approve_usd": 0.05})
    assert any("auto-approve" in p for p in daily_cost_problems(low))
    pricey = JobTypeConfig.from_dict("daily_recommendation", {**raw, "cost_check_usd_per_hour": 1.0})
    assert any("exceeds the USD" in p for p in daily_cost_problems(pricey))
    long = JobTypeConfig.from_dict("daily_recommendation", {**raw, "max_runtime_seconds": 3600})
    assert daily_cost_problems(long)


# ===================================================================== UNV-03 no schedule; API resource policy
def test_unv03_no_schedule_submits_jobs():
    """FinanceModel's only schedule is the dispatcher tick (it starts queued runs, never submits)."""
    from infra.stacks.control import DISPATCH_SCHEDULE_DESCRIPTION

    src = (ROOT / "infra" / "stacks" / "control.py").read_text()
    assert src.count("scheduler.CfnSchedule(") == 1 and "submit" not in DISPATCH_SCHEDULE_DESCRIPTION.lower()
    assert 'input="{}"' in src  # the tick carries no job request


def test_unv03_negative_template_fixture_is_detected():
    from finplan_model.control.production_strategy import DAILY_JOB_TYPE

    bad = {"Resources": {"S": {"Type": "AWS::Scheduler::Schedule", "Properties": {"Target": {"Input": json.dumps({"job_type": "run_benchmark"})}}}}}
    good = {"Resources": {"S": {"Type": "AWS::Scheduler::Schedule", "Properties": {"Target": {"Input": "{}"}}}}}

    def submits(template):
        return [k for k, r in template["Resources"].items() if r["Type"] in ("AWS::Scheduler::Schedule", "AWS::Events::Rule") and "job_type" in json.dumps(r)]

    assert submits(bad) == ["S"] and submits(good) == [] and DAILY_JOB_TYPE == "daily_recommendation"


ACCT = "<account-id>"
API = f"arn:aws:execute-api:us-east-2:{ACCT}:api1/api"


def _invoke(role: str, method: str, path: str) -> bool:
    pol = job_api_resource_policy("beta", invoker_role_patterns=["finplan-beta-financelambdastool-*", "finplan-beta-financialplanning-operator*"], api_resource=API, partition="aws", account=ACCT)
    res = evaluate(Request("execute-api:Invoke", f"{API}/{method}/{path}", {"aws:PrincipalArn": f"arn:aws:iam::{ACCT}:role/{role}"}), [pol])
    return res.allowed


def test_resource_policy_trigger_role_submit_and_status_only():
    trig = "finplan-beta-financialplanning-daily-trigger-step-role"
    assert _invoke(trig, "POST", "v1/jobs") and _invoke(trig, "GET", "v1/jobs/run_x")
    for method, path in (("GET", "v1/jobs"), ("POST", "v1/jobs/run_x/cancel"), ("POST", "v1/jobs/run_x/approve"), ("PUT", "v1/production-strategy"), ("GET", "v1/production-strategy")):
        assert not _invoke(trig, method, path), (method, path)
    assert not _invoke("finplan-gamma-financialplanning-daily-trigger-step-role", "POST", "v1/jobs")


def test_resource_policy_operator_pipeline_stage_and_strategy_writers():
    op = "finplan-beta-financialplanning-operator-pipeline-stage"
    assert _invoke(op, "POST", "v1/jobs") and _invoke(op, "GET", "v1/jobs/run_x") and _invoke(op, "PUT", "v1/production-strategy")
    assert _invoke("finplan-beta-financelambdastool-plan-writer-role", "PUT", "v1/production-strategy")
    assert not _invoke("finplan-beta-financelambdastool-submitter-role", "PUT", "v1/production-strategy")
    assert _invoke("finplan-beta-financelambdastool-reader-role", "GET", "v1/production-strategy")


def test_config_job_types_identical_across_environments():
    jts = [copy.deepcopy(load_config(e).raw["job_types"]["daily_recommendation"]) for e in ("beta", "gamma", "prod")]
    assert jts[0] == jts[1] == jts[2]
