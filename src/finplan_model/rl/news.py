"""Versioned, point-in-time news metadata for research, never retroactive features.

Deterministic headline lexicon features are hypotheses, not calibrated sentiment.
The actual retrieval time is an availability lower bound even when a provider
reports an earlier first-seen time. Revised content receives a new identity.
"""
from __future__ import annotations

import math
import re
from datetime import timedelta

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.clock import parse_utc, utc_iso
from finplan_model.core.errors import FinplanError

EXTRACTOR = "headline-lexicon/1"
POSITIVE = frozenset(("beats", "upgrade", "growth", "profit", "rises", "record"))
NEGATIVE = frozenset(("misses", "downgrade", "loss", "falls", "lawsuit", "recall"))
UNCERTAIN = frozenset(("uncertain", "may", "could", "risk", "warning"))


def freeze_sources(sources, *, retrieved_at):
    from finplan_model.classical.research import safe_url
    retrieved = parse_utc(retrieved_at)
    records = []
    for source in sources[:50]:
        url = safe_url(source.get("url"))
        title = str(source.get("title", ""))[:500]
        if not url or not title:
            continue
        seen = source.get("first_seen_at", source.get("published_at"))
        try:
            observed = parse_utc(seen)
        except (ValueError, TypeError):
            continue
        if observed > retrieved:
            continue
        tokens = set(re.findall(r"[a-z]+", title.lower()))
        sentiment = (len(tokens & POSITIVE) - len(tokens & NEGATIVE)) / max(1, len(tokens & (POSITIVE | NEGATIVE)))
        record = {"url": url, "title": title, "instrument_ids": sorted(set(source.get("instrument_ids", []))),
                  "source": str(source.get("source", "public_news_metadata")),
                  "first_seen_at": utc_iso(observed), "retrieved_at": utc_iso(retrieved),
                  "available_at": utc_iso(max(observed, retrieved)), "extractor_version": EXTRACTOR,
                  "original_publication_verified": False,
                  "features": {"sentiment": sentiment, "uncertainty": float(bool(tokens & UNCERTAIN)),
                               "earnings_event": float(bool(tokens & {"earnings", "guidance"}))}}
        record["content_checksum"] = sha256_checksum(canonical_json_bytes({"url": url, "title": title}))
        record["record_checksum"] = sha256_checksum(canonical_json_bytes(record))
        records.append(record)
    return records


def features_at(records, instrument, *, decision_time, max_age_hours=72):
    cutoff = parse_utc(decision_time)
    eligible = []
    for record in records:
        content = {k: v for k, v in record.items() if k != "record_checksum"}
        if sha256_checksum(canonical_json_bytes(content)) != record.get("record_checksum"):
            raise FinplanError.precondition("news record changed after freezing", reason="news_integrity")
        if record.get("extractor_version") != EXTRACTOR:
            raise FinplanError.precondition("unknown news extractor", reason="news_extractor_mismatch")
        available = max(parse_utc(record["available_at"]), parse_utc(record["retrieved_at"]), parse_utc(record["first_seen_at"]))
        if instrument in record["instrument_ids"] and cutoff - timedelta(hours=max_age_hours) <= available <= cutoff:
            eligible.append(record)
    if not eligible:
        return {"status": "missing", "sentiment": 0., "count": 0, "age_hours": 0., "uncertainty": 0., "earnings_event": 0., "missing": 1.}
    n = len(eligible)
    result = {"status": "available", "sentiment": sum(r["features"]["sentiment"] for r in eligible) / n,
              "count": n, "age_hours": min((cutoff - parse_utc(r["available_at"])).total_seconds() / 3600 for r in eligible),
              "uncertainty": sum(r["features"]["uncertainty"] for r in eligible) / n,
              "earnings_event": max(r["features"]["earnings_event"] for r in eligible), "missing": 0.}
    if not all(math.isfinite(v) for v in result.values() if isinstance(v, (int, float))):
        raise FinplanError.precondition("nonfinite news feature", reason="news_features_invalid")
    return result


def require_coverage(records, instruments, decision_times):
    for timestamp in decision_times:
        for instrument in instruments:
            if features_at(records, instrument, decision_time=timestamp)["missing"]:
                raise FinplanError.precondition("news training lacks frozen point-in-time coverage", reason="news_coverage_missing", instrument=instrument)
