"""Shared CDK building blocks for every FinanceModel stack (FOUNDATION-owned).

* :class:`ModelStack` - base class for per-environment stacks: applies the contract cost-allocation
  tags (``project``, ``owner-repo`` = ``financemodel``, ``environment``) to everything in the stack,
  and the environment permission boundary ``finplan-<env>-permission-boundary`` (created by the
  FinancialPlanning tooling stack) to **every** IAM role in the stack, CDK-generated ones included.
  Research job roles that read approved snapshots and write the staging prefix override it with the
  research boundary (:func:`research_boundary`). Each resource still needs its own
  ``logical-role`` tag: call :func:`tag_role`.
* :func:`resource_name` / :func:`model_role_name` - ``finplan-<env>-financemodel-<logical>`` names
  (the naming the boundaries' name-based denies and the platform grants rely on).
* :func:`role_arn_pattern` - ``arn:${Partition}:iam::${AccountId}:role/<pattern>`` from tokens; no
  account literal ever appears in a file.
* :class:`StageContext` - what optional stack modules receive from ``infra/app.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import aws_cdk as cdk
from aws_cdk import Aws, Stack, Tags
from aws_cdk import aws_iam as iam
from constructs import Construct, IConstruct
from finplan_contracts import boundaries as contract_boundaries
from finplan_contracts import ssm as contract_ssm

from finplan_model.core.config import EnvConfig

__all__ = [
    "REPO",
    "REPO_ROOT",
    "ModelStack",
    "StageContext",
    "model_role_name",
    "research_boundary",
    "resource_name",
    "role_arn_pattern",
    "ssm_name",
    "stack_name",
    "tag_role",
]

REPO = "financemodel"
REPO_ROOT = Path(__file__).resolve().parents[2]


def resource_name(env: str, logical: str, suffix: str | None = None) -> str:
    return contract_boundaries.resource_name(env, REPO, logical, suffix)


def stack_name(env: str, part: str) -> str:
    return f"finplan-{env}-{REPO}-{part}"


def model_role_name(env: str, logical: str) -> str:
    name = resource_name(env, logical, "role")
    if len(name) > 64:
        raise ValueError(f"role name {name!r} exceeds 64 characters")
    return name


def role_arn_pattern(role_name_pattern: str) -> str:
    partition, account = Aws.PARTITION, Aws.ACCOUNT_ID
    return f"arn:{partition}:iam::{account}:role/{role_name_pattern}"


def ssm_name(env: str, category: str, name: str) -> str:
    return contract_ssm.build(env, REPO, category, name)


def tag_role(construct: IConstruct, logical_role: str) -> None:
    """Set the ``logical-role`` tag (ownership-matrix key) on a construct tree."""
    Tags.of(construct).add("logical-role", logical_role)


def research_boundary(scope: Construct, cid: str, env: str) -> iam.IManagedPolicy:
    """The research permission boundary (approved-snapshot reads, staging writes only; ENV-04)."""
    return iam.ManagedPolicy.from_managed_policy_name(scope, cid, contract_boundaries.research_boundary_name(env))


class ModelStack(Stack):
    """Base per-environment stack: contract tags, environment permission boundary, prod protection."""

    def __init__(self, scope: Construct, construct_id: str, *, cfg: EnvConfig, description: str, **kwargs: Any) -> None:
        super().__init__(
            scope,
            construct_id,
            env=cdk.Environment(region=cfg.region),
            description=description,
            termination_protection=cfg.env == "prod",
            **kwargs,
        )
        self.cfg = cfg
        self.env_name = cfg.env
        base = contract_ssm.cost_allocation_tags(REPO, cfg.env, "placeholder")
        for key in ("project", "owner-repo", "environment"):
            Tags.of(self).add(key, base[key])
        boundary = iam.ManagedPolicy.from_managed_policy_name(self, "EnvPermissionBoundary", contract_boundaries.boundary_name(cfg.env))
        iam.PermissionsBoundary.of(self).apply(boundary)


@dataclass
class StageContext:
    """Passed to ``add_to_stage(stage, ctx)`` of every optional per-environment stack module."""

    cfg: EnvConfig
    shared: Mapping[str, Any]
    stacks: dict[str, Stack] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
