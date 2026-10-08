"""Task 7.6: every example in docs/job-interface.md validates against the pinned contracts, and the
documented operations, job types, error mapping and strategies match the implementation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from finplan_contracts.canonical import configuration_id
from finplan_contracts.leak_scan import scan_text
from finplan_contracts.validate import validate

from finplan_model.control.api import HTTP_STATUS, ROUTES
from finplan_model.control.validation import PLANNED_JOB_TYPES, find_storage_location
from finplan_model.core.config import load_config
from finplan_model.jobs.strategy_resolver import PHASE1_STRATEGIES

DOC = Path(__file__).resolve().parents[3] / "docs" / "job-interface.md"
TEXT = DOC.read_text(encoding="utf-8")
EXAMPLES = [(m.group(1), json.loads(m.group(2))) for m in re.finditer(r"<!-- example: ([a-z/-]+) -->\s*```json\n(.*?)\n```", TEXT, re.S)]


def test_doc_has_examples_for_every_operation_shape():
    kinds = {k for k, _ in EXAMPLES}
    assert {"job-submission", "tools/submit-experiment-response", "job-status", "job-result", "error"} <= kinds
    assert len(EXAMPLES) >= 8


@pytest.mark.parametrize("schema, doc", EXAMPLES, ids=[f"{i}-{k}" for i, (k, _) in enumerate(EXAMPLES)])
def test_example_validates_against_the_contract(schema, doc):
    res = validate(doc, schema)
    assert res.valid, [i.message for i in res.issues]
    assert find_storage_location(doc) is None


def test_submission_example_is_accepted_by_the_service_and_ids_match():
    from tests.unit.control.support import SUBMITTER, Harness

    from finplan_model.control.auth import Principal

    sub = next(d for k, d in EXAMPLES if k == "job-submission")
    h = Harness()
    code, resp = h.service.submit_job(Principal.from_arn(SUBMITTER), sub, correlation_id="corr-docs-0001")
    assert code == 200 and resp["dry_run"] is True
    assert resp["cost_estimate"]["estimated_usd_upper_bound"] == pytest.approx(0.0725)
    assert configuration_id(sub["configuration"]) == resp["configuration_id"]


def test_documented_routes_statuses_and_types_match_the_code():
    for _method, _rx, op in ROUTES:
        assert f"`{op}`" in TEXT
    for code, status in HTTP_STATUS.items():
        assert re.search(rf"\| `{code}` \| {status} \|", TEXT), code
    for jt in load_config("beta").raw["job_types"]:
        assert f"`{jt}`" in TEXT
    for jt in PLANNED_JOB_TYPES:
        assert f"`{jt}`" in TEXT
    for s in PHASE1_STRATEGIES:
        assert f"`{s}`" in TEXT


def test_doc_has_no_leaks():
    assert scan_text(TEXT, str(DOC)) == []
