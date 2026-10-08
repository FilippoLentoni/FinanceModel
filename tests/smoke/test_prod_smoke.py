"""Prod smoke (DEP-05, DEP-03; platform lesson L5): runs as the prod stage role with its real
credentials. The job interface answers ``list_jobs`` and its contract envelopes (nothing is
recorded, no SageMaker job is started), the control-plane bundle runs (one dispatcher tick), and the
prod release manifest records the approver and the same artifact digest as the gamma manifest of the
same release (read from the release ledger in the pipeline store: the prod stage role cannot read
gamma parameters).
"""

from __future__ import annotations

import json
import os

from finplan_model.core.aws_clients import s3_client
from tests.deployed import deployed, requires_deployed
from tests.harness import OFFLINE_MARKER

pytestmark = requires_deployed


def test_smoke_runs_with_real_credentials_not_the_offline_fakes():
    assert OFFLINE_MARKER not in os.environ and os.environ.get("AWS_ACCESS_KEY_ID") != "testing"
    assert os.environ.get("AWS_EC2_METADATA_DISABLED") != "true"


def test_prod_manifest_digest_equals_gamma():
    d = deployed()
    prod = d.manifest()
    assert prod.get("approved_by") and prod.get("approved_at")
    store = os.environ.get("FINPLAN_PIPELINE_STORE")
    assert store, "the stage project names the pipeline store"
    body = s3_client(d.region, session=d.session).get_object(Bucket=store, Key=f"releases/{prod['release_id']}/manifests/gamma.json")["Body"].read()
    assert json.loads(body)["artifact_digest"] == prod["artifact_digest"]


def test_control_plane_bundle_runs():
    from infra.stacks import naming as n

    d = deployed()
    resp = d.session.client("lambda").invoke(FunctionName=n.function_name(d.env, n.DISPATCHER), InvocationType="RequestResponse", Payload=b"{}")
    payload = json.loads(resp["Payload"].read() or b"null")
    assert "FunctionError" not in resp, payload


def test_list_jobs_and_contract_envelopes_only():
    from tests.integration.job_suite import run_contract_checks

    d = deployed()
    run_contract_checks(d.api_call())
