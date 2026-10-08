"""Deployed beta/gamma suite (DEP-04, DEP-05, ENV-06, ENV-07, JOB-02..JOB-08, CTL-05; gamma: ENV-03,
WS-01, WS-05; platform lesson L5).

Runs in the stage project only (``FINPLAN_TARGET_ENV``, set by ``scripts/stage_runner.py``), as
``finplan-<env>-financemodel-pipeline-stage-role`` with its REAL credentials (never the offline
fakes), against the deployed environment: every reference comes from SSM, every API call is SigV4.
The stage fails when fewer than one test executes.

* real credentials, release manifest and published references;
* the control-plane Lambdas run from their dependency bundle (a synchronous dispatcher tick: the
  platform's first deploy failed at init with ``No module named 'finplan_contracts'``);
* through the job API (always deployed since contracts 1.0.0): contract envelopes (``list_jobs``,
  ``VALIDATION_FAILED``, ``NOT_FOUND``) and ONE tiny fixture ``run_backtest`` submitted, replayed,
  refused with ``IDEMPOTENCY_KEY_REUSED``, polled, cancelled and read back
  (:mod:`tests.integration.job_suite`). With the default auto-approve threshold 0 the run never
  starts a SageMaker job, so the suite costs Lambda, API Gateway and DynamoDB requests only;
* gamma only: isolation denials towards prod.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import pytest

from finplan_model.core.aws_clients import s3_client
from tests.deployed import deployed, requires_deployed
from tests.harness import OFFLINE_MARKER

pytestmark = requires_deployed


def test_stage_runs_with_real_credentials_not_the_offline_fakes():
    """Regression (platform lesson L5): the deployed suites keep the stage role's credentials."""
    assert OFFLINE_MARKER not in os.environ, "the offline harness was applied to a deployed suite"
    assert os.environ.get("AWS_ACCESS_KEY_ID") != "testing"
    assert os.environ.get("AWS_CONFIG_FILE") != os.devnull and os.environ.get("AWS_SHARED_CREDENTIALS_FILE") != os.devnull
    assert os.environ.get("AWS_EC2_METADATA_DISABLED") != "true"


def test_suite_runs_as_the_stage_role():
    d = deployed()
    arn = d.session.client("sts").get_caller_identity()["Arn"]
    assert f"/finplan-{d.env}-financemodel-pipeline-stage-role/" in arn, "the suite runs as the stage role"


def test_release_manifest_and_references_published():
    from finplan_contracts import ssm as contract_ssm
    from finplan_contracts.validate import validate

    d = deployed()
    manifest = d.manifest()
    assert validate(manifest, "release-manifest").valid
    assert manifest["repo"] == "financemodel" and manifest["environment"] == d.env
    for key, name in manifest["outputs"].items():
        assert contract_ssm.parse(name).repo == "financemodel", key
        assert d.param(name), f"{name} is listed in the manifest but not published"
    jobs = [k for k in manifest["outputs"] if k.startswith("job-") and k not in ("job-role-ref", "job-api-role-ref", "job-endpoint")]
    assert jobs, "no job definition is published"
    for key in jobs:
        doc = json.loads(d.param(manifest["outputs"][key]) or "{}")
        assert "@sha256:" in doc["image_uri"], "job definitions reference the image by digest"
    assert d.param(d.own("config", "budget-enforced-role-names")), "budget-enforced role names are published for the platform's budget action"
    if os.environ.get("FINPLAN_RELEASE_ID"):
        assert manifest["release_id"] == os.environ["FINPLAN_RELEASE_ID"]


def test_control_plane_lambdas_run_from_their_bundle():
    """One synchronous dispatcher tick: the bundle imports, SSM settings and the control table are
    reachable. Starts nothing that a scheduled tick would not start."""
    from infra.stacks import naming as n

    d = deployed()
    resp = d.session.client("lambda").invoke(FunctionName=n.function_name(d.env, n.DISPATCHER), InvocationType="RequestResponse", Payload=b"{}")
    payload = json.loads(resp["Payload"].read() or b"null")
    assert "FunctionError" not in resp, payload
    assert isinstance(payload, dict) and {"started", "reconciled", "schedule"} <= set(payload), payload


def test_job_api_contract_envelopes():
    from tests.integration.job_suite import run_contract_checks

    d = deployed()
    steps = run_contract_checks(d.api_call())
    print("\n".join(steps))


def test_job_api_fixture_job_lifecycle():
    from tests.integration.job_suite import run_job_lifecycle

    d = deployed()
    call = d.api_call()
    snapshot = d.integration_snapshot_id()
    run_key = os.environ.get("FINPLAN_RELEASE_ID", "") or datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    res = run_job_lifecycle(call, snapshot_id=snapshot, run_key=f"{d.env}-{run_key[-12:]}")
    print(res.run_id, res.states, "\n".join(res.steps))
    # The suite's run is cancelled: one synchronous tick re-plans the dispatcher schedule (design D1),
    # so the environment is left disarmed unless other runs are pending.
    from infra.stacks import naming as n

    resp = d.session.client("lambda").invoke(FunctionName=n.function_name(d.env, n.DISPATCHER), InvocationType="RequestResponse", Payload=b"{}")
    payload = json.loads(resp["Payload"].read() or b"null")
    assert "FunctionError" not in resp and payload.get("schedule"), payload
    print("dispatcher schedule after the suite:", payload["schedule"])


@pytest.mark.skipif(os.environ.get("FINPLAN_TARGET_ENV") != "gamma", reason="isolation suite runs in gamma")
def test_gamma_cannot_reach_prod():
    from botocore.exceptions import ClientError

    d = deployed()
    with pytest.raises(ClientError) as denied:
        d.session.client("ssm").get_parameter(Name="/finplan/prod/financemodel/release/manifest")
    assert denied.value.response["Error"]["Code"] == "AccessDeniedException"
    account = d.session.client("sts").get_caller_identity()["Account"]
    with pytest.raises(ClientError) as head:
        s3_client(d.region, session=d.session).head_bucket(Bucket=f"finplan-prod-financemodel-research-workspace-{account}")
    # 403: prod exists and gamma is refused. 404: prod is not deployed yet (first run, before the
    # prod approval), so there is nothing to reach; the SSM denial above still proves the boundary.
    assert head.value.response["Error"]["Code"] in ("403", "AccessDenied", "404", "NoSuchBucket")
