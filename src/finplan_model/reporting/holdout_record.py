"""Holdout metrics record for promotion (REP-07; task 5.3a; user decision 15c, 2026-10-07).

For every holdout evaluation the common evaluator stores a machine-readable record with the
**net-of-costs cumulative return** and the **maximum drawdown**, together with every comparability
field the promotion check needs to compare a candidate with its incumbent deterministically:

``dataset_id`` and dataset manifest checksum, the holdout bounds, the full simulation configuration
with its ``simulation_configuration_id``, the cost model with its ``cost_model_id``, and the
``evaluator_version`` (plus the image digest and the evaluation's ``result_checksum``).

The values are taken from the evaluation result itself, so they are exactly the numbers in the
benchmark report's holdout column. A record missing any comparability field is refused with
``VALIDATION_FAILED`` naming the field. Records are stored as artifacts of kind
``holdout_metrics_record`` (trusted reference, content-addressed).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

from finplan_model.core.artifacts import ArtifactRef, ArtifactStore
from finplan_model.core.errors import FinplanError
from finplan_model.core.ids import require_id

__all__ = ["COMPARABILITY_FIELDS", "HOLDOUT_RECORD_KIND", "RECORD_VERSION", "build_holdout_record", "store_holdout_record", "validate_holdout_record"]

RECORD_VERSION = "holdout-metrics-record-v1"
HOLDOUT_RECORD_KIND = "holdout_metrics_record"
COMPARABILITY_FIELDS = ("dataset_id", "dataset_manifest_checksum", "holdout", "simulation_configuration", "simulation_configuration_id", "cost_model", "cost_model_id", "evaluator_version")
METRIC_FIELDS = ("net_cumulative_return", "max_drawdown")
_CFG_RE = re.compile(r"^cfg_[0-9a-f]{64}\Z")
_CK_RE = re.compile(r"^sha256:[0-9a-f]{64}\Z")


def validate_holdout_record(doc: Mapping[str, Any]) -> dict[str, Any]:
    for f in ("record_version", "strategy", *COMPARABILITY_FIELDS, *METRIC_FIELDS, "synthetic", "result_checksum"):
        if f not in doc or doc[f] is None or doc[f] == "" or doc[f] == {}:
            raise FinplanError.validation(f"holdout metrics record lacks comparability field {f}", pointer=f"/{f}", missing_field=f)
    for f in METRIC_FIELDS:
        v = doc[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise FinplanError.validation(f"{f} must be a finite number", pointer=f"/{f}")
    h = doc["holdout"]
    if not isinstance(h, Mapping) or not h.get("start") or not h.get("end") or str(h["start"]) > str(h["end"]):
        raise FinplanError.validation("holdout bounds must be {start, end}", pointer="/holdout")
    for f in ("simulation_configuration_id", "cost_model_id"):
        if not _CFG_RE.match(str(doc[f])):
            raise FinplanError.validation(f"{f} must be a configuration_id", pointer=f"/{f}")
    for f in ("dataset_manifest_checksum", "result_checksum"):
        if not _CK_RE.match(str(doc[f])):
            raise FinplanError.validation(f"{f} must be a sha256 checksum", pointer=f"/{f}")
    if doc.get("run_id") is not None:
        require_id("run_id", doc["run_id"])
    if doc.get("model_version") is not None:
        require_id("model_version", doc["model_version"])
    return dict(doc)


def build_holdout_record(result: Mapping[str, Any], *, dataset: Mapping[str, Any], holdout: Mapping[str, str], simulation_configuration: Mapping[str, Any], run_id: str | None, model_version: str | None, seed: int | None = None) -> dict[str, Any]:
    """The record of one holdout evaluation (``result`` is an ``EvaluationResult.to_dict()``)."""
    ident = result["identity"]
    m = result["metrics"]
    sim_cfg = dict(simulation_configuration)
    doc: dict[str, Any] = {
        "record_version": RECORD_VERSION,
        "run_id": run_id,
        "model_version": model_version,
        "strategy": result["strategy"],
        "seed": seed,
        "dataset_id": ident.get("dataset_id") or dataset.get("dataset_id"),
        "dataset_manifest_checksum": ident.get("dataset_checksum") or dataset.get("manifest_checksum"),
        "holdout": {"start": str(holdout["start"]), "end": str(holdout["end"])},
        "evaluation_window": dict(result["evaluation_window"]),
        "simulation_configuration": sim_cfg,
        "simulation_configuration_id": ident["simulation_configuration_id"],
        "cost_model": {"fees": sim_cfg.get("fees"), "spread": sim_cfg.get("spread"), "slippage": sim_cfg.get("slippage")},
        "cost_model_id": ident["cost_model_id"],
        "evaluator_version": ident["evaluator_version"],
        "image_digest": ident.get("image_digest"),
        "net_cumulative_return": m["net_cumulative_return"],
        "max_drawdown": m["max_drawdown"],
        "synthetic": bool(result["synthetic"]),
        "result_checksum": result["result_checksum"],
    }
    if doc["dataset_id"] != dataset.get("dataset_id"):
        raise FinplanError.validation("holdout result was not evaluated on the named dataset", pointer="/dataset_id")
    return validate_holdout_record(doc)


def store_holdout_record(record: Mapping[str, Any], store: ArtifactStore) -> ArtifactRef:
    doc = validate_holdout_record(record)
    return store.put_json(doc, kind=HOLDOUT_RECORD_KIND, synthetic=True if doc["synthetic"] else None, domain="finance")  # type: ignore[attr-defined]
