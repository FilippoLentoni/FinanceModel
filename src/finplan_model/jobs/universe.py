"""Research-universe input (``finance/equity-etf-daily/research-universe``, contracts 1.1.0) for the
existing experiment kinds and the ``daily_recommendation`` job (specs universe-research-input and
daily-recommendation-job).

* :func:`universe_block` / :func:`is_universe_content`: a snapshot payload with a ``universe`` block.
* :func:`adjusted_observation`: the ``adj_close`` return basis. Prices are scaled by
  ``adj_close / close`` so returns follow the adjusted series (dividends and splits included).
* :func:`cash_assumption`: the snapshot's declared cash assumption (``zero_nominal`` -> cash earns
  0 % a year); any other assumption is refused (``VALIDATION_FAILED``) rather than guessed.
* :func:`bias_section`: the mandatory "Hindsight and survivorship bias" section, reproducing the
  snapshot's ``bias_disclosures`` unchanged and stating the cash assumption. A universe snapshot
  without disclosures fails closed (``VALIDATION_FAILED``).
* :func:`full_allocation`: target weights for every universe instrument (zeros included) plus cash,
  summing to 1.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from finplan_model.core.errors import FinplanError

__all__ = [
    "BIAS_SECTION_TITLE",
    "CASH_RATES",
    "adjusted_observation",
    "bias_disclosures_of",
    "bias_section",
    "cash_assumption",
    "full_allocation",
    "is_universe_content",
    "universe_block",
]

BIAS_SECTION_TITLE = "Hindsight and survivorship bias"
#: Declared cash assumptions -> annual cash rate used by the simulator.
CASH_RATES = {"zero_nominal": 0.0}


def universe_block(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    block = payload.get("universe")
    return block if isinstance(block, Mapping) else None


def is_universe_content(content: Any) -> bool:
    return universe_block(getattr(content, "payload", {}) or {}) is not None


def adjusted_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    """The observation on the ``adj_close`` basis (OHLC scaled by ``adj_close / close``)."""
    out = dict(obs)
    adj, close = obs.get("adj_close"), obs.get("close")
    if adj is None or close in (None, 0):
        return out
    factor = float(adj) / float(close)
    out["close"] = float(adj)
    for k in ("open", "high", "low"):
        if out.get(k) is not None:
            out[k] = float(out[k]) * factor
    return out


def bias_disclosures_of(content: Any) -> list[dict[str, Any]]:
    """The snapshot's disclosures, unchanged (record first, payload as fallback)."""
    record = getattr(getattr(content, "snapshot", None), "record", {}) or {}
    found = record.get("bias_disclosures") or (getattr(content, "payload", {}) or {}).get("bias_disclosures") or []
    return copy.deepcopy(list(found))


def cash_assumption(content: Any) -> dict[str, Any]:
    block = universe_block(getattr(content, "payload", {}) or {}) or {}
    cash = [i for i in block.get("instruments", []) if i.get("kind") == "cash"]
    if not cash:
        return {"instrument_id": None, "return_assumption": "none", "annual_rate": 0.0, "weight_allowed": False}
    item = cash[0]
    assumption = str(item.get("return_assumption") or "")
    if assumption not in CASH_RATES:
        raise FinplanError.validation("unsupported cash return assumption", pointer="/universe/instruments", return_assumption=assumption[:64])
    return {"instrument_id": item["instrument_id"], "return_assumption": assumption, "annual_rate": CASH_RATES[assumption], "weight_allowed": True}


def bias_section(content: Any) -> dict[str, Any]:
    """Mandatory report and summary section of every universe run; fails closed without disclosures."""
    disclosures = bias_disclosures_of(content)
    if not disclosures:
        raise FinplanError.validation("the universe snapshot carries no bias disclosures", pointer="/bias_disclosures", reason="bias_disclosures_missing")
    block = universe_block(getattr(content, "payload", {}) or {}) or {}
    return {
        "title": BIAS_SECTION_TITLE,
        "bias_disclosures": disclosures,
        "cash_assumption": cash_assumption(content),
        "return_basis": block.get("return_basis", "adj_close"),
    }


def full_allocation(final_weights: Mapping[str, float], cash: float, instruments: Sequence[str]) -> dict[str, Any]:
    """Weights for every instrument (zeros included) plus cash, normalized to sum to 1."""
    rows = {i: max(0.0, float(final_weights.get(i, 0.0))) for i in instruments}
    cash = max(0.0, float(cash))
    total = sum(rows.values()) + cash
    if total <= 0:
        rows, cash, total = {i: 0.0 for i in instruments}, 1.0, 1.0
    weights = [{"instrument_id": i, "weight": round(rows[i] / total, 12)} for i in instruments]
    cash_w = round(max(0.0, 1.0 - sum(w["weight"] for w in weights)), 12)
    return {"weights": weights, "cash_weight": cash_w}
