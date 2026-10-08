"""Shared outbound payload check for data derived from Yahoo Finance snapshots (DS-14; task 3.9a).

User decision 15d (2026-10-07): data derived from Yahoo Finance snapshots may leave FinanceModel
research storage toward an external service (TypeSafe) only

* from runs with a **research purpose** (``research``, ``tuning``, ``holdout_evaluation``), and
* as **derived, bucketed text descriptors** - never raw price or volume series or bulk data.

:func:`check_outbound_payload` runs before anything is sent and raises ``VALIDATION_FAILED``
(``outbound_payload_rejected``) when the payload carries:

* a run of consecutive numbers longer than ``max_consecutive_numbers`` (in text, or as a JSON array);
* any number equal to a raw price or volume observation of the source dataset (also when rounded to
  the number of decimals it was written with);
* a table (Markdown, HTML or delimiter-separated rows) or an attachment (binary content, base64
  blobs, file or attachment fields);
* more than ``max_bytes`` of content (bulk data).

A run whose purpose is not a research purpose is refused with ``OPERATION_NOT_PERMITTED``.
Rejections are logged with the run ID, the reason codes and counts - **never the payload values**.
:func:`guarded_send` calls the sender only after the check passes, so a rejected payload makes no
network request. The Jev strategy (next change) applies it before every request.
"""

from __future__ import annotations

import base64
import binascii
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

from finplan_model.core.artifacts import canonical_json_bytes
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError

__all__ = ["OutboundPolicy", "RESEARCH_PURPOSES", "check_outbound_payload", "guarded_send", "raw_values_of"]

RESEARCH_PURPOSES = ("research", "tuning", "holdout_evaluation")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?%?(?![A-Za-z0-9_])")
_SEPARATOR_RE = re.compile(r"^[\s,;|\t/]*$")
_MD_TABLE_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", re.MULTILINE)
_HTML_TABLE_RE = re.compile(r"<\s*(table|tr|td|th)\b", re.IGNORECASE)
_B64_RE = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")
_ATTACHMENT_KEYS = re.compile(r"(?i)^(attachments?|files?|file_content|blob|bytes|data_uri|base64|csv|parquet|dataframe|table|rows|series|prices|volumes|ohlcv|bars|observations)$")
T = TypeVar("T")


@dataclass(frozen=True)
class OutboundPolicy:
    max_consecutive_numbers: int = 3
    max_bytes: int = 16_384
    max_table_rows: int = 2


@dataclass
class _Findings:
    reasons: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str, n: int = 1) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + n


def raw_values_of(observations: Iterable[Mapping[str, Any]]) -> set[float]:
    """Raw price and volume values of a dataset's observations (what must never leave)."""
    out: set[float] = set()
    for o in observations:
        for k in ("open", "high", "low", "close", "adj_close", "volume"):
            v = o.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                out.add(float(v))
    return out


def _parse_number(token: str) -> tuple[float, int]:
    t = token.rstrip("%").replace(",", "")
    decimals = len(t.split(".", 1)[1].split("e")[0].split("E")[0]) if "." in t else 0
    return float(t), decimals


def _matches_raw(value: float, decimals: int, raw: set[float], raw_rounded: dict[int, set[float]]) -> bool:
    if value in raw:
        return True
    if abs(value) < 10 and decimals < 2:
        return False  # small integers (bucket indexes, horizons) are too ambiguous to be raw values
    rounded = raw_rounded.get(decimals)
    if rounded is None:
        rounded = raw_rounded[decimals] = {round(v, decimals) for v in raw}
    return round(value, decimals) in rounded


def _scan_text(text: str, policy: OutboundPolicy, raw: set[float], raw_rounded: dict[int, set[float]], f: _Findings) -> None:
    if _MD_TABLE_RE.search(text) or _HTML_TABLE_RE.search(text):
        f.add("table")
    rows = [ln for ln in text.splitlines() if ln.count(",") >= 2 or ln.count("\t") >= 2 or ln.count("|") >= 3]
    if len(rows) > policy.max_table_rows:
        f.add("table")
    if _B64_RE.search(text):
        f.add("attachment")
    matches = list(_NUMBER_RE.finditer(text))
    run, longest = 0, 0
    prev_end = None
    for m in matches:
        value, decimals = _parse_number(m.group(0))
        if _matches_raw(value, decimals, raw, raw_rounded):
            f.add("raw_value")
        if prev_end is not None and _SEPARATOR_RE.match(text[prev_end : m.start()]):
            run += 1
        else:
            run = 1
        longest = max(longest, run)
        prev_end = m.end()
    if longest > policy.max_consecutive_numbers:
        f.add("numeric_run")


def _walk(node: Any, policy: OutboundPolicy, raw: set[float], raw_rounded: dict[int, set[float]], f: _Findings) -> None:
    if isinstance(node, (bytes, bytearray, memoryview)):
        f.add("attachment")
    elif isinstance(node, str):
        _scan_text(node, policy, raw, raw_rounded, f)
    elif isinstance(node, bool) or node is None:
        return
    elif isinstance(node, (int, float)):
        if math.isfinite(float(node)) and _matches_raw(float(node), 0 if isinstance(node, int) else 6, raw, raw_rounded):
            f.add("raw_value")
    elif isinstance(node, Mapping):
        for k, v in node.items():
            if isinstance(k, str) and _ATTACHMENT_KEYS.match(k):
                f.add("attachment")
            _walk(v, policy, raw, raw_rounded, f)
    elif isinstance(node, (list, tuple)):
        nums = 0
        for v in node:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                nums += 1
                if nums > policy.max_consecutive_numbers:
                    f.add("numeric_run")
                    nums = -(10**9)  # count the run once
            else:
                nums = 0
            _walk(v, policy, raw, raw_rounded, f)
        if len(node) > policy.max_table_rows and all(isinstance(v, (list, tuple, Mapping)) for v in node):
            f.add("table")
    else:
        f.add("attachment")


def check_outbound_payload(payload: Any, *, ctx: RunContext, raw_values: Iterable[float] = (), policy: OutboundPolicy | None = None, destination: str = "external") -> None:
    """Raise unless ``payload`` is a research-purpose, descriptor-only payload (see module doc)."""
    policy = policy or OutboundPolicy()
    if ctx.purpose not in RESEARCH_PURPOSES:
        ctx.log("outbound_payload_rejected", destination=destination, reasons={"purpose": 1}, purpose=ctx.purpose)
        raise FinplanError.not_permitted("Yahoo-derived data leaves FinanceModel only from research-purpose runs", reason="outbound_purpose_not_permitted", purpose=ctx.purpose)
    f = _Findings()
    try:
        size = len(canonical_json_bytes(payload)) if not isinstance(payload, (bytes, bytearray)) else len(payload)
    except (TypeError, ValueError):
        size = policy.max_bytes + 1
        f.add("attachment")
    if size > policy.max_bytes:
        f.add("bulk")
    if isinstance(payload, str) and len(payload) > 64:
        try:
            base64.b64decode(payload, validate=True)
            f.add("attachment")
        except (binascii.Error, ValueError):
            pass
    _walk(payload, policy, set(float(v) for v in raw_values), {}, f)
    if f.reasons:
        # Logged without any payload value: reason codes and counts only.
        ctx.log("outbound_payload_rejected", destination=destination, reasons=dict(sorted(f.reasons.items())), size_bytes=size)
        raise FinplanError.validation("outbound payload rejected: only derived, bucketed text descriptors may leave FinanceModel", pointer="", reason="outbound_payload_rejected", violations=sorted(f.reasons))
    ctx.log("outbound_payload_checked", destination=destination, size_bytes=size)


def guarded_send(payload: Any, send: Callable[[Any], T], *, ctx: RunContext, raw_values: Iterable[float] = (), policy: OutboundPolicy | None = None, destination: str = "external") -> T:
    """Check ``payload`` and only then call ``send(payload)`` (a rejected payload is never sent)."""
    check_outbound_payload(payload, ctx=ctx, raw_values=raw_values, policy=policy, destination=destination)
    return send(payload)
