"""Caller identity and authorization inside the job API (on top of IAM, spec experiment-job-interface
"Published job interface reference", "Run purpose"; job-execution-controls "Human approval").

API Gateway authenticates every call with SigV4 (IAM); its resource policy allows only the granted
principals to invoke each route (an unauthorized principal gets ``403`` from API Gateway before the
Lambda runs, see :mod:`finplan_model.control.policies`). The handler additionally:

* requires a caller identity (``requestContext.identity.userArn``); none -> ``UNAUTHORIZED``;
* scopes idempotency by the normalized principal (an assumed-role session ARN becomes its role ARN,
  so retries from a new session of the same role replay);
* allows ``production_candidate`` only for principals listed in
  ``/finplan/<env>/financemodel/config/production-candidate-principals`` -> else ``FORBIDDEN``;
* allows ``approve_run`` only for the human approver role named at
  ``/finplan/<env>/financemodel/config/approver-role-ref``, never for the submitting principal and
  never for an agent, tool-wrapper or pipeline role (checked by name, defense in depth).

Only role **names** are ever written to responses (``approved_by``); ARNs (which carry the account)
stay inside the run store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from finplan_model.core.errors import ErrorCode, FinplanError

__all__ = ["Principal", "denied_approver_reason", "normalize_principal", "role_name_of"]

#: Role-name patterns that may never approve a run (agents, tool wrappers, pipelines, builds).
_DENIED_APPROVER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tool_wrapper_role", re.compile(r"^finplan-[a-z]+-financelambdastool-")),
    ("agent_role", re.compile(r"^finplan-[a-z]+-financeagent-")),
    ("pipeline_role", re.compile(r"(^|-)(pipeline|codebuild|codepipeline|deploy|cfn-exec|bootstrap)(-|$)")),
    ("job_role", re.compile(r"^finplan-[a-z]+-financemodel-(job|job-api|dispatcher|state-handler)(-|$)")),
    ("aws_service_role", re.compile(r"^AWSServiceRole")),
)


def role_name_of(arn: str) -> str | None:
    """Role name of a role ARN (``...:role/<path/>name``) or an assumed-role session ARN."""
    parts = arn.split(":", 5)
    if len(parts) != 6 or parts[0] != "arn":
        return None
    service, resource = parts[2], parts[5]
    if service == "iam" and resource.startswith("role/"):
        return resource.rsplit("/", 1)[-1]
    if service == "sts" and resource.startswith("assumed-role/"):
        segs = resource.split("/")
        return segs[1] if len(segs) >= 3 else None
    return None


def normalize_principal(arn: str) -> str:
    """Assumed-role session ARN -> role ARN (stable idempotency scope); other ARNs unchanged."""
    parts = arn.split(":", 5)
    name = role_name_of(arn)
    if name and len(parts) == 6 and parts[2] == "sts":
        return f"arn:{parts[1]}:iam::{parts[4]}:role/{name}"
    return arn


def denied_approver_reason(role_name: str) -> str | None:
    for reason, pattern in _DENIED_APPROVER_PATTERNS:
        if pattern.search(role_name):
            return reason
    return None


@dataclass(frozen=True)
class Principal:
    """The authenticated caller of one API request."""

    arn: str
    role_name: str | None

    @classmethod
    def from_arn(cls, raw: str | None) -> "Principal":
        if not raw or not isinstance(raw, str) or not raw.startswith("arn:"):
            raise FinplanError(ErrorCode.UNAUTHORIZED, "the caller could not be authenticated")
        return cls(arn=normalize_principal(raw), role_name=role_name_of(raw))

    @property
    def display_name(self) -> str:
        """Safe name for responses and logs (role name, never an ARN)."""
        return self.role_name or "unknown-principal"

    def require_production_candidate_grant(self, granted_role_names: list[str]) -> None:
        if not self.role_name or self.role_name not in set(granted_role_names):
            raise FinplanError(ErrorCode.FORBIDDEN, "the caller is not granted the production_candidate purpose", details={"purpose": "production_candidate"})

    def require_approver(self, approver_role_name: str | None, *, submitter: str) -> None:
        """Approval only by the configured human approver role, never by the submitter or a tool/agent/pipeline role."""
        name = self.role_name or ""
        reason = denied_approver_reason(name) if name else "unknown_principal"
        if reason is not None:
            raise FinplanError(ErrorCode.FORBIDDEN, "this principal may not approve runs", details={"reason": reason})
        if self.arn == submitter:
            raise FinplanError(ErrorCode.FORBIDDEN, "the submitting principal may not approve its own run", details={"reason": "submitter_cannot_approve"})
        if not approver_role_name or name != approver_role_name:
            raise FinplanError(ErrorCode.FORBIDDEN, "only the approver role may approve runs", details={"reason": "not_approver_role"})
