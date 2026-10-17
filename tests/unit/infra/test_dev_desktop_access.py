"""Beta operator exception cannot escape beta or weaken immutable registry records."""
import copy

import pytest
from finplan_contracts import iam

from finplan_model.control.auth import Principal
from finplan_model.control.policies import job_api_resource_policy
from finplan_model.core.config import EnvConfig, load_config, validate_config
from finplan_model.core.errors import FinplanError
from infra.stacks.policies import control_table_policy, registry_bucket_policy, research_bucket_policy

ARN = "arn:aws:iam::<account-id>:role/finplan-devdesktop"
ADMIN = {"Statement": [{"Effect": "Allow", "Action": "*", "Resource": "*"}]}


def decision(policy, action, resource, **context):
    return iam.evaluate(iam.Request(action, resource, {"aws:PrincipalArn": ARN, **context}), [ADMIN, policy]).decision


@pytest.mark.parametrize("env", ["beta", "gamma", "prod"])
def test_storage_and_api_access(env, assembly):
    cfg = load_config(env)
    expected = iam.ALLOWED if env == "beta" else iam.EXPLICIT_DENY
    table = "arn:aws:dynamodb:us-east-2:<account-id>:table/control"
    policy = control_table_policy(env, table, dev_desktop_roles=cfg.dev_desktop_role_names, partition="aws", account="<account-id>")
    for action in ("dynamodb:Scan", "dynamodb:PutItem", "dynamodb:UpdateItem"):
        assert decision(policy, action, table) == expected
    bucket = "arn:aws:s3:::example-research-bucket"
    for builder in (registry_bucket_policy, research_bucket_policy):
        policy = {"Statement": builder(env, bucket, dev_desktop_roles=cfg.dev_desktop_role_names, partition="aws", account="<account-id>")}
        assert decision(policy, "s3:GetObject", bucket + "/example") == expected
    policy = job_api_resource_policy(env, invoker_role_patterns=[], dev_desktop_roles=cfg.dev_desktop_role_names, partition="aws", account="<account-id>")
    for path in ("POST/v1/jobs/r/approve", "PUT/v1/production-strategy"):
        assert decision(policy, "execute-api:Invoke", "execute-api:/live/" + path) == expected
    # Synth wiring, rather than only testing pure builders.
    text = str(assembly.templates[f"finplan-{env}-financemodel-storage"])
    assert ("role/finplan-devdesktop" in text) == (env == "beta")
    text = str(assembly.templates[f"finplan-{env}-financemodel-control"])
    assert ("role/finplan-devdesktop" in text) == (env == "beta")


def test_registry_integrity():
    bucket = "arn:aws:s3:::example-registry-bucket"
    policy = {"Statement": registry_bucket_policy("beta", bucket, dev_desktop_roles=("finplan-devdesktop",), partition="aws", account="<account-id>")}
    for action in ("s3:PutObject", "s3:DeleteObject", "s3:DeleteObjectVersion"):
        assert decision(policy, action, bucket + "/example") == iam.EXPLICIT_DENY


@pytest.mark.parametrize("env", ["gamma", "prod"])
def test_reject_exception_outside_beta(env):
    raw = copy.deepcopy(load_config(env).raw)
    raw["dev_desktop_access"] = {"role_names": ["finplan-devdesktop"]}
    assert validate_config(env, raw)
    assert EnvConfig(env, raw).dev_desktop_role_names == ()


def test_beta_operator_approval_keeps_submitter_separation():
    desktop = Principal.from_arn(ARN)
    roles = load_config("beta").dev_desktop_role_names
    desktop.require_approver("other-approver", submitter="other-principal", operator_role_names=roles)
    with pytest.raises(FinplanError, match="own run"):
        desktop.require_approver("other-approver", submitter=desktop.arn, operator_role_names=roles)
    with pytest.raises(FinplanError, match="only the approver"):
        desktop.require_approver("other-approver", submitter="other-principal", operator_role_names=())


def test_beta_handler_approval_and_cancel():
    from tests.unit.control.support import Harness

    h = Harness()
    _, submitted = h.submit()
    desktop = Principal.from_arn(ARN)
    result = h.service.approve_run(desktop, submitted["run_id"], {})
    assert result["state"] == "queued"
    h.service.cancel_job(desktop, submitted["run_id"], {"idempotency_key": "desktop-cancel"})
    assert h.run(submitted["run_id"])["state"] == "cancelled"
    _, own = h.submit(principal=ARN, idempotency_key="desktop-own-run")
    with pytest.raises(FinplanError, match="own run"):
        h.service.approve_run(desktop, own["run_id"], {})
    assert h.created_jobs() == []


def test_beta_handler_selection_keeps_confirmation():
    from tests.unit.control.support import Harness
    from tests.unit.control.test_daily_and_strategy import _selection

    service, parameter, _ = _selection(Harness())
    with pytest.raises(FinplanError, match="confirm"):
        service.handle(Principal.from_arn(ARN), {"action": "clear", "idempotency_key": "desktop-clear"})
    assert parameter.writes == 0
    status, result = service.handle(Principal.from_arn(ARN), {"action": "clear", "idempotency_key": "desktop-clear", "confirmed_by_user": True})
    assert status == 200 and result["strategy"] is None


def test_beta_advisory_activation_keeps_confirmation():
    from finplan_model.rl.advisory import activate
    from tests.unit.control.support import Harness

    with pytest.raises(FinplanError, match="requires user intent"):
        activate(Harness().service, Principal.from_arn(ARN), {})
