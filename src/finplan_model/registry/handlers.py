"""Lambda entry point of the registry lookup (``finplan-<env>-financemodel-job-registry-lookup``).

Environment (values set by the stack, never committed): ``FINPLAN_ENVIRONMENT`` and
``FINPLAN_REGISTRY_BUCKET`` (the environment's model-registry bucket). Read-only: its role may only
read the registry (:func:`infra.stacks.policies.registry_lookup_policy`).
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["lookup_handler"]

_API: Any = None


def _api() -> Any:
    global _API
    if _API is None:
        from finplan_model.core.aws_clients import s3_client

        from .api import RegistryApi
        from .registry import ModelRegistry, S3RegistryStore

        _API = RegistryApi(ModelRegistry(S3RegistryStore(s3_client(), os.environ["FINPLAN_REGISTRY_BUCKET"])))
    return _API


def lookup_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    return _api().handle(event)
