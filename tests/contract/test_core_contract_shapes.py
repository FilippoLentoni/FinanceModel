"""Contract tests: documents produced by the shared core validate against the pinned contracts."""

from __future__ import annotations

import pytest
from finplan_contracts.validate import validate

from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.core.errors import ErrorCode, FinplanError
from finplan_model.core.outcome import failed_job_result, require_valid
from finplan_model.core.platform import build_synthetic_snapshot

_DETAILS = {
    ErrorCode.VALIDATION_FAILED: {"pointer": "/a"},
    ErrorCode.INVALID_IDENTIFIER: {"field": "run_id"},
    ErrorCode.UNSUPPORTED_CONTRACT_VERSION: {"served_contract_majors": [0]},
}


@pytest.mark.parametrize("code", list(ErrorCode))
def test_every_registered_code_produces_a_valid_envelope(code):
    env = FinplanError(code, "synthetic test error", details=_DETAILS.get(code, {"reason": "test"})).to_envelope("corr-12345678", synthetic=True)
    assert env["code"] == code.value
    assert validate(env, "error").valid


def test_failed_job_result_validates(ctx):
    rctx = ctx.with_run(ctx.ids.run_id())
    doc = failed_job_result(rctx, FinplanError.internal("container exited", reason="crash"))
    assert validate(doc, "job-result").valid
    assert doc["artifacts_complete"] is False and doc["synthetic"] is True


def test_artifact_refs_validate():
    ref = InMemoryArtifactStore().put_json({"a": 1}, kind="research_dataset", synthetic=True, domain="finance")
    assert validate(ref.to_dict(), "artifact-ref").valid


def test_synthetic_snapshot_record_and_payload_validate():
    payload = {"dataset_id": "finance/etf-daily/SPY", "instruments": [{"instrument_id": "SPY", "asset_class": "etf", "currency": "USD"}], "observations": [{"instrument_id": "SPY", "session_date": "2026-01-05", "kind": "completed_daily", "close": 100.0}]}
    rec, blobs = build_synthetic_snapshot(payload, input_snapshot_id="snap_01KDVDNAZ83BAMMYCEGWF33DPM")
    assert validate(rec, "input-snapshot").valid
    assert rec["synthetic"] is True and rec["status"] == "approved"


def test_require_valid_raises_contract_codes():
    with pytest.raises(FinplanError) as ei:
        require_valid({"run_id": "pl_01KDVDNAZ83BAMMYCEGWF33DPM", "completion_status": "failed", "artifacts": [], "artifacts_complete": False}, "job-result")
    assert ei.value.code in ("INVALID_IDENTIFIER", "VALIDATION_FAILED")
