"""Contract tests: DS-02 (dataset lineage record), DS-08 (synthetic fixtures), DS-12 (provenance
copied from snapshot lineage). Shared contract fixtures are read from the pinned package."""

from __future__ import annotations

from datetime import date

from finplan_contracts.validate import validate

from finplan_model.core.context import RunContext
from finplan_model.core.errors import contract_version
from finplan_model.datasets import fixture_calendar, synthetic_etf_observations
from finplan_model.datasets.fixtures import MockSnapshotProvider, contract_fixture
from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.datasets import InMemoryDatasetCatalog, prepare_dataset

FIXTURE = contract_fixture("input-snapshot", "calendar-version-quality-details-observation-summary.json")
CFG = {
    "instrument": {"tickers": ["SPY"]},
    "availability": {"mode": "session_close"},
    "features": {"lookback_sessions": 1, "label_horizon_sessions": 0},
    "splits": {"train": {"start": "2026-01-02", "end": "2026-01-05"}, "validation": {"start": "2026-01-07", "end": "2026-01-07"}, "holdout": {"start": "2026-01-09", "end": "2026-01-09"}, "embargo_sessions": 0},
}


def _prepared():
    ctx = RunContext.for_tests(seed=3, start="2026-02-01T00:00:00Z")
    cal = fixture_calendar(date(2025, 1, 1), date(2027, 12, 31))
    assert cal.version == FIXTURE["lineage"]["calendar_version"]
    prov = MockSnapshotProvider(clock=ctx.clock, calendar=cal)
    lin = FIXTURE["lineage"]
    sid = prov.publish(
        synthetic_etf_observations("SPY", cal.between(date(2026, 1, 2), date(2026, 1, 9)), seed=1),
        retrieved_at=lin["retrieved_at"],
        provider=lin["provider"],
        lineage_extra={k: lin[k] for k in ("provider_library", "library_version", "calendar_version")},
    )
    store = InMemoryArtifactStore()
    prep = prepare_dataset([sid], CFG, reader=prov.reader, store=store, catalog=InMemoryDatasetCatalog(), ctx=ctx.with_run(ctx.ids.run_id()))
    return prov, sid, prep, store


def test_synthetic_snapshot_built_from_contract_fixtures_validates():
    prov, sid, _prep, _store = _prepared()
    content = prov.reader.load(sid)
    assert validate(content.snapshot.record, "input-snapshot").valid
    assert validate(content.payload, "snapshot-payload").valid
    for obs in content.payload["observations"]:
        assert validate(obs, "observation").valid and obs["synthetic"] is True
    assert content.snapshot.synthetic is True and content.manifest["synthetic"] is True


def test_lineage_record_traces_to_snapshots_and_checksums():
    prov, sid, prep, _store = _prepared()
    rec = prep.record
    snap = prov.reader.resolve(sid)
    (entry,) = rec["input_snapshots"]
    assert entry["input_snapshot_id"] == sid and entry["manifest_checksum"] == snap.manifest_checksum
    # provenance copied verbatim from the snapshot lineage (shared fixture values)
    for k in ("provider", "provider_library", "library_version", "retrieved_at"):
        assert entry["provenance"][k] == FIXTURE["lineage"][k]
    assert entry["provenance"]["calendar"]["version"] == FIXTURE["lineage"]["calendar_version"]
    assert rec["domain"] == FIXTURE["domain"] and rec["domain_schema_version"] == FIXTURE["domain_schema_version"]
    assert rec["contract_version"] == contract_version()
    assert rec["configuration_id"].startswith("cfg_") and rec["created_at"] == "2026-02-01T00:00:00Z"
    assert rec["synthetic"] is True


def test_research_dataset_reference_is_a_contract_trusted_reference():
    _prov, _sid, prep, store = _prepared()
    ref = prep.ref
    assert validate(ref, "artifact-ref").valid
    assert ref["kind"] == "research_dataset" and ref["owner"] == "financemodel"
    assert ref["artifact_id"] == prep.dataset_id and ref["checksum"] == prep.manifest_checksum
    assert store.exists(ref)


def test_results_reference_datasets_only_through_the_lineage_record():
    _prov, sid, prep, _store = _prepared()
    rec = prep.record
    doc = {
        "run_id": rec["created_by_run_id"],
        "completion_status": "succeeded",
        "solution_status": "not_applicable",
        "artifacts": [rec["dataset_ref"]],
        "artifacts_complete": True,
        "configuration_id": rec["configuration_id"],
        "input_snapshot_id": sid,
        "dataset_checksum": rec["manifest_checksum"],
        "domain": rec["domain"],
        "domain_schema_version": rec["domain_schema_version"],
        "evaluator_version": "1.0.0",
        "synthetic": True,
    }
    result = validate(doc, "job-result")
    assert result.valid, [i.message for i in result.issues]
