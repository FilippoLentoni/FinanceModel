#!/usr/bin/env python3
"""FinanceModel CDK app: one ``Stage`` per environment (beta, gamma, prod) plus account-level stacks.

Synthesis is offline: stacks are account-agnostic (``AWS::AccountId`` is a deploy-time pseudo
parameter), configuration comes from ``config/<env>.json`` and ``config/shared.json``, and no
context lookups are used. Run ``npx aws-cdk@2 synth`` (``cdk.json``: ``uv run python infra/app.py``);
select environments with ``-c envs=beta,gamma`` (default: all three).

Hook contract for stack modules owned by other task groups
-----------------------------------------------------------
* per-environment modules (:data:`ENV_MODULES`, for example ``infra.stacks.storage``,
  ``infra.stacks.control``) expose ``add_to_stage(stage: aws_cdk.Stage, ctx: StageContext) -> None``.
  They create their stacks from :class:`infra.stacks.common.ModelStack` (tags + boundary), register
  them in ``ctx.stacks[<name>]`` and pass values to later modules through ``ctx.extras``. Modules run
  in the listed order.
* account-level modules (:data:`APP_MODULES`, for example ``infra.stacks.pipeline``) expose
  ``add_to_app(app: aws_cdk.App, shared: Mapping, stages: dict[str, StageContext]) -> None``.

A module that does not exist yet is skipped; an import error *inside* an existing module fails the
synth. While no module adds a stack to an environment, that environment gets one resource-free
placeholder stack (``finplan-<env>-financemodel-skeleton``) so the skeleton synthesizes (task 1.1).
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import aws_cdk as cdk  # noqa: E402

from finplan_model.core.config import DEPLOYED_ENVIRONMENTS, load_config, load_shared_config  # noqa: E402
from infra.stacks.common import ModelStack, StageContext, stack_name  # noqa: E402

ENV_MODULES = ("infra.stacks.storage", "infra.stacks.tables", "infra.stacks.roles", "infra.stacks.jobs", "infra.stacks.control", "infra.stacks.registry", "infra.stacks.release")
APP_MODULES = ("infra.stacks.tooling", "infra.stacks.pipeline")

__all__ = ["APP_MODULES", "ENV_MODULES", "build_app", "optional_module"]


def optional_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            return None
        raise


def build_app(app: cdk.App | None = None, envs: Iterable[str] | None = None, *, env_modules: Iterable[str] = ENV_MODULES, app_modules: Iterable[str] = APP_MODULES, placeholder: bool = True) -> cdk.App:
    app = app or cdk.App()
    selected = list(envs) if envs is not None else _selected_envs(app)
    shared = load_shared_config()
    stages: dict[str, StageContext] = {}
    for env in selected:
        cfg = load_config(env)
        stage = cdk.Stage(app, env.capitalize())
        ctx = StageContext(cfg=cfg, shared=shared)
        for name in env_modules:
            mod = optional_module(name)
            if mod is not None:
                mod.add_to_stage(stage, ctx)
        if placeholder and not any(isinstance(c, cdk.Stack) for c in stage.node.children):
            # Skeleton only: an environment no module contributes to yet gets one resource-free
            # stack so `cdk synth` has something to synthesize. It disappears as soon as any
            # stack module adds a stack to the stage.
            ctx.stacks["skeleton"] = ModelStack(stage, "Skeleton", cfg=cfg, description="FinanceModel skeleton placeholder (no resources); replaced by the stack modules", stack_name=stack_name(env, "skeleton"), analytics_reporting=True)
        stages[env] = ctx
    for name in app_modules:
        mod = optional_module(name)
        if mod is not None:
            mod.add_to_app(app, shared, stages)
    return app


def _selected_envs(app: cdk.App) -> list[str]:
    raw = app.node.try_get_context("envs")
    if not raw:
        return list(DEPLOYED_ENVIRONMENTS)
    envs = [e.strip() for e in str(raw).split(",") if e.strip()]
    unknown = [e for e in envs if e not in DEPLOYED_ENVIRONMENTS]
    if unknown:
        raise SystemExit(f"unknown environments in -c envs: {unknown}")
    return envs


if __name__ == "__main__":
    build_app().synth()
