"""Native BTC/ETH hourly cohort on authoritative isolated books; no provider calls.

Reads the already approved platform's original hourly candles. A genuine completed
hourly close is the declared execution quote, never represented as a minute quote
or a hindsight bar-open fill. Equity defaults and frozen policies are unchanged.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
from datetime import UTC, datetime
import boto3
from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.platform import HttpPlatformClient
from finplan_model.crypto_research import PRODUCTS, validate_dataset
from . import hourly
from .crypto import crypto_adapter
from .registry import APPROACHES
from .research import stamp

CRYPTO_POLICY = {**copy.deepcopy(hourly.POLICY), "version": "beta-forward-crypto-hourly/1",
    "market_calendar": "CRYPTO_24X7", "execution_timing": "next_hourly_crypto_completed_quote",
    "quote_protocol": "native_completed_1h_coinbase_close/1", "execution_price_freshness_seconds": 1200,
    "price_freshness_seconds": 1200, "fees": {"proportional_bps": 60.0, "fixed_per_trade": 0.0},
    "spread": {"half_spread_bps": 10.0}, "slippage": {"model": "none", "coefficient_bps": 0.0},
    "execution_cost_definition": "Common kernel:60bps reference-notional fee+10bps spread; distinct from legacy slippage-adjusted fee basis",
    "cohort_scope": "BTC_USD_ETH_USD_native_hourly_24x7; never ranked against equity cohorts"}


def crypto_registry():
    return [a.to_dict() for a in APPROACHES if a.adapter == "crypto_native"]


def source_from_dataset(dataset, *, acquired_at, slot, allow_synthetic=False):
    at = datetime.fromisoformat(acquired_at.replace("Z", "+00:00"))
    data = validate_dataset(copy.deepcopy(dataset), at)
    if data["market_data_synthetic"] and not allow_synthetic:
        raise ValueError("forward crypto requires actual approved platform source")
    if datetime.fromisoformat(data["as_of"].replace("Z", "+00:00")) != datetime.fromisoformat(slot.replace("Z", "+00:00")):
        raise ValueError("matching completed hourly source is not available for this slot")
    age = (at - datetime.fromisoformat(data["as_of"].replace("Z", "+00:00"))).total_seconds()
    if not 0 <= age <= CRYPTO_POLICY["price_freshness_seconds"]:
        raise ValueError("native completed hourly quote exceeds declared freshness")
    quotes = []
    for instrument in PRODUCTS:
        row = data["bars"][instrument][-1]
        quotes.append({"instrument_id": instrument, "price": row["close"], "volume": row["volume"],
            "volume_units": "base_asset_units_during_actual_completed_hour", "currency": "USD", "status": "fresh",
            "frequency": "1h", "provider": "Coinbase Exchange public completed candles", "venue_calendar": "CRYPTO_24X7",
            "quoted_at": data["as_of"], "bar_started_at": row["start"], "bar_ended_at": data["as_of"],
            "available_at": acquired_at, "retrieved_at": acquired_at, "platform_available_at": data["available_at"],
            "availability_basis": "actual acquisition of immutable central dataset; original platform availability separately retained",
            "freshness_seconds": age, "freshness_limit_seconds": CRYPTO_POLICY["price_freshness_seconds"],
            "dataset_checksum": data["dataset_checksum"], "source_provenance": data["sources"],
            "usage": "declared native hourly close for crypto valuation/execution; never a minute quote"})
    return {"dataset_id": data["dataset_id"], "checksum": data["dataset_checksum"],
        "investable_universe": list(PRODUCTS), "native_crypto_dataset": data, "registry_approaches": crypto_registry(),
        "observations": [], "execution_observations": [], "news": [], "raw_news": [],
        "current_quotes": {"quotes": quotes, "available_at": acquired_at, "status": "captured_native_hourly",
            "collection_owner": "financialplanning", "consumer_acquired_at": acquired_at},
        "trigger": "controlled_replay" if allow_synthetic else os.environ.get("FINPLAN_EXPERIMENT_TRIGGER", "manual_validation"),
        "model_native_cadence": "hourly_crypto_24x7", "acquired_at": acquired_at,
        "code_versions": {"financemodel": os.environ.get("FINPLAN_MODEL_SOURCE_COMMIT", "not_supplied"),
            "financialplanning": os.environ.get("FINPLAN_PLATFORM_SOURCE_COMMIT", "not_supplied"),
            "financeagent": os.environ.get("FINPLAN_AGENT_SOURCE_COMMIT", "not_supplied")},
        "limitations": [CRYPTO_POLICY["cohort_scope"], CRYPTO_POLICY["execution_cost_definition"],
            "Completed hourly close is carried for at most1200seconds; original quote time remains explicit"]}


def load_source(slot, *, session=None):
    session = session or boto3.Session(region_name="us-east-2")
    if session.region_name != "us-east-2":
        raise ValueError("beta crypto restricted to us-east-2")
    endpoint = session.client("ssm").get_parameter(Name="/finplan/beta/financialplanning/api/plan-endpoint")["Parameter"]["Value"]
    client = HttpPlatformClient(endpoint, region="us-east-2", credentials=session.get_credentials())
    data = client.get_crypto_dataset()
    return source_from_dataset(data, acquired_at=stamp(), slot=slot)


def run(slot, experiment, output, *, backend=None, source=None, now=None, account_missed=False):
    if source is None:
        try:
            source = load_source(slot) if not account_missed else {"investable_universe": list(PRODUCTS),
                "registry_approaches": crypto_registry(), "observations": [], "execution_observations": [], "historical_slot_missing": True}
        except Exception as exc:
            source = {"investable_universe": list(PRODUCTS), "registry_approaches": crypto_registry(),
                "observations": [], "execution_observations": [], "acquisition_failure": type(exc).__name__ + ": " + str(exc),
                "trigger": os.environ.get("FINPLAN_EXPERIMENT_TRIGGER", "manual_validation")}
    return hourly.run_hourly(slot, experiment, output, backend=backend, source=source, policies={}, research={"news": []}, now=now,
        external_adapters={"crypto_native": crypto_adapter}, account_missed=account_missed, comparison_policy=CRYPTO_POLICY)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--account-missed", action="store_true")
    args = parser.parse_args()
    result = run(args.slot, args.experiment, args.output, account_missed=args.account_missed)
    print(canonical_json_bytes(result).decode())


if __name__ == "__main__":
    main()
