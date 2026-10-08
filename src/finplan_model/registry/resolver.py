"""Wiring of the model registry into the control plane, the job container and the deploy
(tasks 9.2, 9.3; REG-01, REG-04).

* :func:`model_version_resolver` - the ``ServiceDeps.model_version_resolver`` hook of the control
  plane: ``(strategy, image_digest) -> model_version``. It registers the strategy implementation
  (idempotent, so it returns the existing ``model_version`` for a known identity) using the
  parameter schema version from :func:`finplan_model.strategies.registry.descriptor`. The minted
  ``model_version`` goes into the run spec, so every result carries it (REG-04).
* :func:`lineage_result_hook` - a ``ServiceDeps.result_hooks`` entry that records run lineage
  (``runs/<run_id>.json``) for every terminal result that names a ``model_version`` (idempotent).
* :func:`record_spec_lineage` - the same from a run spec (job container, before staging).
* :func:`seed_baselines` - the deploy-time registration of the controls and classical optimizers
  with the release's image digest (design D9; ``scripts/release.py``).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from finplan_model.core.errors import FinplanError

from .registry import ModelIdentity, ModelRegistry

__all__ = ["lineage_result_hook", "model_version_resolver", "record_spec_lineage", "seed_baselines"]


def _schema_version(strategy: str) -> str | None:
    try:
        from finplan_model.strategies.registry import BASELINES, descriptor
    except ImportError:  # pragma: no cover - strategies ship in the same package
        return None
    if strategy not in BASELINES:
        return None
    return str(descriptor(strategy)["param_schema_version"])


def model_version_resolver(registry: ModelRegistry, *, actor: str) -> Callable[[str | None, str | None], str | None]:
    """``ServiceDeps.model_version_resolver``: register (or find) the strategy implementation."""

    def resolve(strategy: str | None, image_digest: str | None) -> str | None:
        if not strategy or not image_digest:
            return None
        version = _schema_version(strategy)
        if version is None:
            return None
        record, _created = registry.register(ModelIdentity(strategy, image_digest, version), actor=actor)
        return str(record["model_version"])

    return resolve


def lineage_result_hook(registry: ModelRegistry, *, actor: str) -> Callable[[Mapping[str, Any], dict[str, Any]], dict[str, Any]]:
    """``ServiceDeps.result_hooks`` entry: record ``run_id`` -> ``model_version`` lineage (write-once)."""

    def hook(run: Mapping[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        mv = result.get("model_version") or run.get("model_version")
        run_id = result.get("run_id") or run.get("run_id")
        if mv and run_id:
            try:
                registry.record_run(str(run_id), str(mv), actor=actor, details=run)
            except FinplanError as exc:
                if exc.code != "NOT_FOUND":  # an unregistered model_version is reported by the lookup, not here
                    raise
        return result

    return hook


def record_spec_lineage(registry: ModelRegistry, spec: Mapping[str, Any], *, actor: str) -> dict[str, Any] | None:
    """Record the run's lineage from its run spec (the job container calls this before staging)."""
    mv = spec.get("model_version")
    if not mv:
        return None
    return registry.record_run(str(spec["run_id"]), str(mv), actor=actor, details=spec)


def seed_baselines(registry: ModelRegistry, image_digest: str, *, actor: str, strategies: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    """Register every baseline strategy for this release's image digest; ``{strategy: record}`` (idempotent)."""
    from finplan_model.strategies.registry import BASELINES, descriptor

    out: dict[str, dict[str, Any]] = {}
    for name in sorted(strategies or BASELINES):
        record, created = registry.register(ModelIdentity(name, image_digest, str(descriptor(name)["param_schema_version"])), actor=actor)
        out[name] = {**record, "created": created}
    return out
