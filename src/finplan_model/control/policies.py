"""IAM policy documents for the job API (consumed by ``infra``, task 10.2/10.3; CTL-08, JOB-01, JOB-07).

* :func:`job_api_resource_policy` - the API Gateway resource policy: only the granted invoker roles
  may call the API at all (others get ``403`` before the Lambda runs, JOB-01 "Unauthorized caller");
  ``POST /v1/jobs/*/approve`` is denied to every principal except the human approver role, and
  explicitly denied to tool-wrapper, agent, pipeline and FinanceModel service roles (CTL-08). The
  handler re-checks the same rules by role name (:mod:`finplan_model.control.auth`).
* :func:`approver_identity_policy` - what the approver role may do: read the job API and approve.
* :func:`gateway_responses` - API Gateway gateway-response templates so that refusals made by API
  Gateway itself (no or invalid SigV4 signature, resource-policy deny) are contract error envelopes
  (``UNAUTHORIZED`` / ``FORBIDDEN``) like the handler's own errors.

Documents use ``${AWS::Partition}`` / ``${AWS::AccountId}`` placeholders (``Fn::Sub``), never an
account literal. ``api_resource`` defaults to the API Gateway resource-policy shorthand
``execute-api:/*`` (any stage); tests pass a concrete placeholder ARN to simulate requests with
:func:`finplan_contracts.iam.evaluate`.
"""

from __future__ import annotations

from typing import Any

__all__ = ["DENIED_APPROVER_ROLE_PATTERNS", "approver_identity_policy", "approver_role_name", "gateway_responses", "job_api_resource_policy"]

#: Role-name patterns explicitly denied ``approve_run`` (agents, tool wrappers, pipelines, service roles).
DENIED_APPROVER_ROLE_PATTERNS = (
    "finplan-*-financelambdastool-*",
    "finplan-*-financeagent-*",
    "*pipeline*",
    "*codebuild*",
    "*deploy*",
    "finplan-*-financemodel-job*",
    "finplan-*-financemodel-dispatcher*",
    "finplan-*-financemodel-state-handler*",
)


def approver_role_name(env: str) -> str:
    """``finplan-<env>-financemodel-approver-role`` (published at ``config/approver-role-ref``)."""
    return f"finplan-{env}-financemodel-approver-role"


def _role_arn(name_pattern: str, partition: str, account: str) -> str:
    return f"arn:{partition}:iam::{account}:role/{name_pattern}"


def job_api_resource_policy(
    env: str,
    *,
    invoker_role_patterns: list[str],
    approver_role: str | None = None,
    api_resource: str = "execute-api:/*",
    partition: str = "${AWS::Partition}",
    account: str = "${AWS::AccountId}",
) -> dict[str, Any]:
    approver = approver_role or approver_role_name(env)
    approver_arn = _role_arn(approver, partition, account)
    invokers = sorted({_role_arn(p, partition, account) for p in invoker_role_patterns} | {approver_arn})
    approve = f"{api_resource}/POST/v1/jobs/*/approve"
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowGrantedInvokers",
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": "execute-api:Invoke",
                "Resource": f"{api_resource}/*",
                "Condition": {"ArnLike": {"aws:PrincipalArn": invokers}},
            },
            {
                "Sid": "DenyApproveExceptApproverRole",
                "Effect": "Deny",
                "Principal": {"AWS": "*"},
                "Action": "execute-api:Invoke",
                "Resource": approve,
                "Condition": {"ArnNotEquals": {"aws:PrincipalArn": approver_arn}},
            },
            {
                "Sid": "DenyApproveToolAgentPipelineRoles",
                "Effect": "Deny",
                "Principal": {"AWS": "*"},
                "Action": "execute-api:Invoke",
                "Resource": approve,
                "Condition": {"ArnLike": {"aws:PrincipalArn": [_role_arn(p, partition, account) for p in DENIED_APPROVER_ROLE_PATTERNS]}},
            },
        ],
    }


def approver_identity_policy(api_resource: str) -> dict[str, Any]:
    """Identity policy of the human approver role: read the job API and approve runs, nothing else."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {"Sid": "ReadJobs", "Effect": "Allow", "Action": "execute-api:Invoke", "Resource": [f"{api_resource}/GET/v1/jobs", f"{api_resource}/GET/v1/jobs/*"]},
            {"Sid": "ApproveRuns", "Effect": "Allow", "Action": "execute-api:Invoke", "Resource": f"{api_resource}/POST/v1/jobs/*/approve"},
            {"Sid": "CancelRuns", "Effect": "Allow", "Action": "execute-api:Invoke", "Resource": f"{api_resource}/POST/v1/jobs/*/cancel"},
        ],
    }


def gateway_responses(contract_version: str) -> dict[str, dict[str, Any]]:
    """``{response_type: {"status_code", "template"}}`` for ``AWS::ApiGateway::GatewayResponse``.

    ``$context.requestId`` stands in for the correlation id (the handler is never reached).
    """

    def body(code: str, message: str) -> str:
        return (
            '{"code":"' + code + '","message":"' + message + '","retryable":false,"details":{},'
            '"correlation_id":"apigw-$context.requestId","contract_version":"' + contract_version + '"}'
        )

    return {
        "MISSING_AUTHENTICATION_TOKEN": {"status_code": "401", "template": body("UNAUTHORIZED", "the caller could not be authenticated")},
        "INVALID_SIGNATURE": {"status_code": "401", "template": body("UNAUTHORIZED", "the caller could not be authenticated")},
        "EXPIRED_TOKEN": {"status_code": "401", "template": body("UNAUTHORIZED", "the caller could not be authenticated")},
        "ACCESS_DENIED": {"status_code": "403", "template": body("FORBIDDEN", "the caller is not allowed to perform the operation")},
        "THROTTLED": {"status_code": "429", "template": body("RATE_LIMITED", "the request was throttled; retry later").replace('"retryable":false', '"retryable":true')},
    }
