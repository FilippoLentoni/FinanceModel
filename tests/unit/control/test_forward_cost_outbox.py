"""Real transaction/queue recovery semantics; no AWS, vendor calls or time sleeps."""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

import boto3
import pytest
from moto import mock_aws

from finplan_model.control.store import ConditionFailed, DynamoRunStore, InMemoryRunStore
from .test_dynamo_store_and_settings import TABLE, _create_table


@pytest.fixture(params=["memory", "dynamo"])
def store(request):
    if request.param == "memory":
        yield InMemoryRunStore()
    else:
        with mock_aws():
            client = boto3.client("dynamodb", region_name="us-east-2")
            _create_table(client)
            yield DynamoRunStore(client, TABLE)


def proposal(name="one", experiment="cost-test", previous=None):
    rid = "run_" + name
    binding = {"artifact_id": "run_artifact_" + name, "checksum": "sha256:" + "a" * 64}
    descriptor = {"mode": "forward_one_decision", "experiment_id": experiment,
        "approach_id": "jev_systemone", "portfolio_id": "pf_cost_test",
        "context_id": experiment + "-context", "binding_ref": binding, "binding_checksum": binding["checksum"]}
    run = {"run_id": rid, "state": "running", "revision": 0,
        "submitted_at": "2026-10-18T01:00:00Z", "forward_jev": descriptor}
    context = {k: descriptor[k] for k in ("context_id", "binding_ref", "binding_checksum")}
    context.update(managed_run_id=rid, previous_managed_run_id=previous)
    return run, context


def terminal(run, *, state="succeeded", at="2026-10-18T01:05:00Z"):
    new = {**run, "state": state, "revision": run["revision"] + 1, "completed_at": at}
    event = {"protocol": "finplan-managed-model-cost/1", "event_id": "managed:" + run["run_id"],
        "managed_run_id": run["run_id"], "occurred_at": at, "job_state": state,
        "invocation_state": "indeterminate" if state == "failed" else "completed",
        "compute_observed_usd": None, "compute_estimated_usd": 0.01,
        "compute_cost_basis": "configured rate estimate; not observed billing",
        "vendor_observed_usd": None, "vendor_estimated_usd": None,
        "vendor_cost_basis": "unavailable", "usage": None}
    event.update({k: run["forward_jev"][k] for k in ("experiment_id", "approach_id", "portfolio_id", "binding_ref", "binding_checksum")})
    return new, event


def reference(cost):
    from datetime import datetime, timezone
    from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
    slot = datetime.fromisoformat(cost["occurred_at"].replace("Z", "+00:00")).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    parts = [cost["experiment_id"], cost["approach_id"], slot, "managed_model_cost:" + cost["managed_run_id"]]
    return {"observation_id": "mo_" + sha256_checksum(canonical_json_bytes(parts)).split(":")[1],
        "checksum": sha256_checksum(canonical_json_bytes(cost))}


def test_terminal_transition_and_cost_are_atomic_and_private(store):
    run, _ = proposal()
    store.create_run(run, [{"seq": 0, "state": "running"}])
    new, cost = terminal(run)
    with pytest.raises(ConditionFailed):
        store.update_run(new, 7, [{"seq": 1, "state": "succeeded"}], forward_cost_event=cost)
    assert store.query_forward_cost_events("cost-test") == ([], None)
    assert store.get_run(run["run_id"])["state"] == "running"
    store.update_run(new, 0, [{"seq": 1, "state": "succeeded"}], forward_cost_event=cost)
    assert store.query_forward_cost_events("cost-test")[0] == [cost]
    assert [r["run_id"] for r in store.query_runs()[0]] == [run["run_id"]]
    assert len(store.list_events(run["run_id"])) == 2
    # A delayed duplicate loses the original revision and cannot repeat the event.
    with pytest.raises(ConditionFailed):
        store.update_run(new, 0, [], forward_cost_event=cost)
    assert store.query_forward_cost_events("cost-test")[0] == [cost]


def test_pending_forever_until_verified_delivery_then_same_ref_retry(store):
    run, _ = proposal()
    store.create_run(run, [])
    new, cost = terminal(run, state="failed")
    store.update_run(new, 0, forward_cost_event=cost)
    assert store.query_forward_cost_events("cost-test")[0][0]["vendor_observed_usd"] is None
    ref = reference(cost)
    for bad in ({**ref, "observation_id": "mo_" + "f" * 64}, {**ref, "checksum": "sha256:" + "d" * 64}):
        with pytest.raises(ValueError):
            store.ack_forward_cost_event("cost-test", cost["event_id"], bad, "2026-11-18T01:06:00Z")
        assert store.query_forward_cost_events("cost-test")[0] == [cost]
    store.ack_forward_cost_event("cost-test", cost["event_id"], ref, "2026-11-18T01:06:00Z")
    assert store.query_forward_cost_events("cost-test") == ([], None)
    store.ack_forward_cost_event("cost-test", cost["event_id"], ref, "2026-12-18T01:06:00Z")
    with pytest.raises(ConditionFailed):
        store.ack_forward_cost_event("another-experiment", cost["event_id"], ref, "2026-12-18T01:06:00Z")
    # Original terminal metadata can be updated without rewriting/reactivating its cost.
    store.update_run({**new, "revision": 2}, 1, forward_cost_event=cost)
    assert store.query_forward_cost_events("cost-test") == ([], None)


def test_ack_normalizes_key_time_without_rewriting_original_event(store):
    run, _ = proposal()
    store.create_run(run, [])
    new, cost = terminal(run, at="2026-10-18T01:05:00+00:00")
    store.update_run(new, 0, forward_cost_event=cost)
    assert store.query_forward_cost_events("cost-test")[0] == [cost]
    ref = reference(cost)
    assert ref["observation_id"] == reference({**cost, "occurred_at": "2026-10-18T01:05:00Z"})["observation_id"]
    store.ack_forward_cost_event("cost-test", cost["event_id"], ref, "2026-10-18T01:06:00Z")
    assert store.query_forward_cost_events("cost-test") == ([], None)


def test_changed_permanent_cost_rolls_back_new_revision(store):
    run, _ = proposal()
    store.create_run(run, [])
    new, cost = terminal(run)
    store.update_run(new, 0, forward_cost_event=cost)
    with pytest.raises(ConditionFailed):
        store.update_run({**new, "revision": 2}, 1, forward_cost_event={**cost, "compute_estimated_usd": 123.0})
    assert store.get_run(run["run_id"])["revision"] == 1


def test_scoped_bounded_pagination_and_no_outbox_in_public_listing(store):
    for name, exp in (("one", "cost-test"), ("two", "cost-test"), ("foreign", "other-test")):
        run, _ = proposal(name, exp)
        store.create_run(run, [])
        new, cost = terminal(run)
        store.update_run(new, 0, forward_cost_event=cost)
    page, token = store.query_forward_cost_events("cost-test", limit=1)
    assert len(page) == 1 and token
    with pytest.raises(ValueError):
        store.query_forward_cost_events("other-test", limit=1, next_token=token)
    second, _ = store.query_forward_cost_events("cost-test", limit=1, next_token=token)
    assert len(second) == 1 and second[0]["event_id"] != page[0]["event_id"]
    for invalid in (0, 101, True):
        with pytest.raises(ValueError):
            store.query_forward_cost_events("cost-test", limit=invalid)
    with pytest.raises(ValueError):
        store.query_forward_cost_events("cost-test", next_token="malformed")
    assert len(store.query_runs()[0]) == 3
    assert len(store.query_runs("succeeded")[0]) == 3


@pytest.mark.parametrize("mutation", [
    {"managed_run_id": "run_foreign"}, {"experiment_id": "foreign"},
    {"binding_checksum": "sha256:" + "f" * 64}, {"portfolio_id": "pf_foreign"},
    {"occurred_at": "2026-10-18T01:07:00Z"}, {"occurred_at": "2026-10-18T01:05:00"},
    {"vendor_observed_usd": -1}, {"compute_estimated_usd": float("nan")},
])
def test_invalid_cost_binding_or_value_never_publishes_head(store, mutation):
    run, _ = proposal()
    store.create_run(run, [])
    new, cost = terminal(run)
    with pytest.raises(ValueError):
        store.update_run(new, 0, forward_cost_event={**cost, **mutation})
    assert store.get_run(run["run_id"])["revision"] == 0
    assert store.query_forward_cost_events("cost-test") == ([], None)


def test_context_and_job_creation_are_one_cas_transaction(store):
    first, head = proposal()
    store.create_run(first, [], forward_context=head)
    second, next_head = proposal("two", previous=first["run_id"])
    idem = {"scope_key": "new-proposal", "ttl": 1}
    with pytest.raises(ConditionFailed):
        store.create_run(second, [], idem, forward_context={**next_head, "previous_managed_run_id": "run_missing"})
    assert store.get_run(second["run_id"]) is None
    assert store.get_idempotency(idem["scope_key"]) is None
    assert store.get_forward_context(head["context_id"]) == head
    store.create_run(second, [], idem, forward_context=next_head)
    assert store.get_forward_context(head["context_id"]) == next_head
    assert store.get_run(first["run_id"]) == first  # original job and evidence retained
    assert len(store.query_runs()[0]) == 2  # context heads never appear as jobs
    third, stale = proposal("three", previous=first["run_id"])
    with pytest.raises(ConditionFailed):
        store.create_run(third, [], forward_context=stale)
    assert store.get_run(third["run_id"]) is None


def test_context_competing_proposals_have_one_winner(store):
    # Both contenders observed no predecessor. DynamoDB transaction CAS and the
    # memory lock must couple the winning pointer with exactly its own job.
    def submit(name):
        run, context = proposal(name)
        try:
            store.create_run(run, [], forward_context=context)
            return run["run_id"]
        except ConditionFailed:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = [r for r in pool.map(submit, ("one", "two")) if r]
    assert len(winners) == 1
    assert store.get_forward_context("cost-test-context")["managed_run_id"] == winners[0]
    assert len(store.query_runs()[0]) == 1
