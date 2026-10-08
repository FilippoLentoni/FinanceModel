#!/usr/bin/env python3
"""Instance prices for the pre-flight cost estimate, fetched from the AWS Price List API (design D4;
``docs/operations.md``, "Configuration an operator writes").

The control plane refuses a paid job without a current price (``PRECONDITION_FAILED``): it reads
``/finplan/<env>/financemodel/config/instance-prices``::

    {"retrieved_at": "<RFC 3339>", "currency": "USD", "usd_per_hour": {"ml.m5.xlarge": <price>},
     "source": "AWS Price List API", "service_code": "AmazonSageMaker", "component": "Processing",
     "region": "<region>"}

No price is ever written in this repository. This module asks the Price List API (``pricing``,
served from us-east-1) for the on-demand hourly price of every instance type the environment's job
types may use, as a SageMaker **Processing** instance in the deployment region, and writes the
document as the ``bootstrap`` writer (:func:`finplan_contracts.ssm.check_write`).

* :func:`ensure_instance_prices` (used by ``scripts/bootstrap.py`` after the tooling deploy): writes
  the parameter of an environment only when it is absent, unreadable, missing an instance type, or
  older than ``cost.price_max_age_days``; a current operator value is left alone.
* CLI (prices go stale after 30 days, so an operator refreshes them)::

      uv run python scripts/instance_prices.py               # show what would be written (read-only)
      uv run python scripts/instance_prices.py --write       # write absent or stale prices
      uv run python scripts/instance_prices.py --write --force --env beta

A price the API does not return stops the write for that environment (nothing is guessed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from finplan_contracts import ssm as contract_ssm  # noqa: E402

from finplan_model.core.clock import parse_utc, utc_iso  # noqa: E402
from finplan_model.core.config import DEPLOYED_ENVIRONMENTS, EnvConfig, load_config  # noqa: E402

__all__ = [
    "COMPONENT",
    "PRICING_REGION",
    "SERVICE_CODE",
    "PriceUnavailable",
    "PricesDecision",
    "decide",
    "ensure_instance_prices",
    "fetch_prices",
    "instance_types",
    "main",
    "parameter_name",
]

REPO = "financemodel"
SERVICE_CODE = "AmazonSageMaker"
#: The SageMaker usage the control plane starts (``CreateProcessingJob``).
COMPONENT = "Processing"
UNIT = "Hrs"
#: The Price List API is served from us-east-1 (it prices every region).
PRICING_REGION = "us-east-1"


class PriceUnavailable(RuntimeError):
    """The Price List API returned no on-demand hourly price for an instance type."""


def parameter_name(env: str) -> str:
    return contract_ssm.build(env, REPO, "config", "instance-prices")


def instance_types(cfg: EnvConfig) -> list[str]:
    """Every instance type a deployed job type of ``cfg`` may request."""
    out: set[str] = set()
    for jt in cfg.job_types.values():
        if jt.deployed:
            out.update(jt.instance_types or (jt.default_instance_type,))
    return sorted(out)


def _hourly_usd(price_list: Sequence[Any]) -> float | None:
    best: float | None = None
    for item in price_list:
        prod = json.loads(item) if isinstance(item, str) else item
        attrs = (prod.get("product") or {}).get("attributes") or {}
        if attrs.get("component") != COMPONENT:
            continue
        for term in ((prod.get("terms") or {}).get("OnDemand") or {}).values():
            for dim in (term.get("priceDimensions") or {}).values():
                if str(dim.get("unit", "")) != UNIT:
                    continue
                try:
                    usd = float((dim.get("pricePerUnit") or {}).get("USD"))
                except (TypeError, ValueError):
                    continue
                if usd > 0 and (best is None or usd < best):
                    best = usd
    return best


def fetch_prices(pricing: Any, region: str, types: Iterable[str], *, now: datetime | None = None) -> dict[str, Any]:
    """The ``instance-prices`` document for ``types`` in ``region`` (Price List API, on-demand)."""
    prices: dict[str, float] = {}
    for itype in sorted(set(types)):
        resp = pricing.get_products(
            ServiceCode=SERVICE_CODE,
            Filters=[
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                {"Type": "TERM_MATCH", "Field": "instanceName", "Value": itype},
                {"Type": "TERM_MATCH", "Field": "component", "Value": COMPONENT},
            ],
            FormatVersion="aws_v1",
            MaxResults=100,
        )
        usd = _hourly_usd(resp.get("PriceList") or [])
        if usd is None:
            raise PriceUnavailable(f"the Price List API returned no on-demand {COMPONENT} price for {itype} in {region}")
        prices[itype] = usd
    return {
        "retrieved_at": utc_iso(now or datetime.now(UTC)),
        "currency": "USD",
        "usd_per_hour": prices,
        "source": "AWS Price List API",
        "service_code": SERVICE_CODE,
        "component": COMPONENT,
        "region": region,
    }


@dataclass(frozen=True)
class PricesDecision:
    env: str
    parameter: str
    action: str  # "keep" | "write"
    reason: str


def decide(cfg: EnvConfig, current: str | None, *, now: datetime, force: bool = False) -> PricesDecision:
    """Whether the bootstrap may (re)write ``cfg``'s instance prices (module docstring)."""
    name = parameter_name(cfg.env)
    if force:
        return PricesDecision(cfg.env, name, "write", "forced refresh")
    if current is None:
        return PricesDecision(cfg.env, name, "write", "absent")
    try:
        doc = json.loads(current)
        retrieved = parse_utc(str(doc["retrieved_at"]))
        have = set((doc.get("usd_per_hour") or {}).keys())
    except (ValueError, KeyError, TypeError):
        return PricesDecision(cfg.env, name, "write", "unreadable")
    missing = sorted(set(instance_types(cfg)) - have)
    if missing:
        return PricesDecision(cfg.env, name, "write", f"no price for {', '.join(missing)}")
    max_age = timedelta(days=float(cfg.raw["cost"]["price_max_age_days"]))
    if now - retrieved > max_age:
        return PricesDecision(cfg.env, name, "write", f"older than {cfg.raw['cost']['price_max_age_days']} days")
    return PricesDecision(cfg.env, name, "keep", f"current (retrieved {doc['retrieved_at']})")


def _get(ssm: Any, name: str) -> str | None:
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception as exc:
        if getattr(exc, "response", {}).get("Error", {}).get("Code") == "ParameterNotFound":
            return None
        raise


def ensure_instance_prices(
    ssm: Any,
    pricing: Any,
    *,
    envs: Iterable[str] = DEPLOYED_ENVIRONMENTS,
    configs: Mapping[str, EnvConfig] | None = None,
    write: bool = True,
    force: bool = False,
    now: datetime | None = None,
    out: Callable[[str], None] = print,
) -> list[PricesDecision]:
    """Write absent or stale ``instance-prices`` from the Price List API (one fetch per region)."""
    now = now or datetime.now(UTC)
    decisions: list[PricesDecision] = []
    fetched: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    for env in envs:
        cfg = (configs or {}).get(env) or load_config(env, ROOT / "config")
        d = decide(cfg, _get(ssm, parameter_name(env)), now=now, force=force)
        decisions.append(d)
        if d.action == "keep":
            out(f"[OK] {d.parameter}: {d.reason}; left unchanged")
            continue
        types = tuple(instance_types(cfg))
        key = (cfg.region, types)
        if key not in fetched:
            fetched[key] = fetch_prices(pricing, cfg.region, types, now=now)
        doc = fetched[key]
        value = json.dumps(doc, sort_keys=True, separators=(",", ":"))
        if not write:
            out(f"[DRY-RUN] {d.parameter} ({d.reason}) would be written: {value}")
            continue
        decision = contract_ssm.check_write(d.parameter, contract_ssm.Writer(REPO, "bootstrap"))
        if not decision.allowed:
            raise PermissionError(f"{d.parameter}: " + "; ".join(decision.reasons))
        ssm.put_parameter(Name=d.parameter, Value=value, Type="String", Overwrite=True, Description="On-demand SageMaker Processing hourly prices (AWS Price List API), written by the FinanceModel bootstrap")
        out(f"[WROTE] {d.parameter} ({d.reason}): {', '.join(f'{k} {v:g} USD/h' for k, v in doc['usd_per_hour'].items())}, retrieved {doc['retrieved_at']}")
    return decisions


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - needs AWS credentials
    ap = argparse.ArgumentParser(description="Fetch SageMaker Processing instance prices from the AWS Price List API into /finplan/<env>/financemodel/config/instance-prices.")
    ap.add_argument("--env", action="append", choices=DEPLOYED_ENVIRONMENTS, help="environment (repeatable; default: all)")
    ap.add_argument("--write", action="store_true", help="write absent or stale parameters (default: show only)")
    ap.add_argument("--force", action="store_true", help="refresh even a current value")
    args = ap.parse_args(argv)
    import boto3

    region = str(json.loads((ROOT / "config" / "shared.json").read_text(encoding="utf-8"))["region"])
    session = boto3.session.Session(region_name=region)
    try:
        ensure_instance_prices(session.client("ssm"), session.client("pricing", region_name=PRICING_REGION), envs=args.env or DEPLOYED_ENVIRONMENTS, write=args.write, force=args.force)
    except (PriceUnavailable, PermissionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
