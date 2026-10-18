"""Offline Claude Code reviewer protocol of the ``direct_claude_news`` book.

Claude Code runs only on the dev box and cannot be called from Lambda. The exchange is
two immutable DynamoDB records in the FinancialPlanning ``research_input`` table:

1. **llm_prompt** (written by the forward runner, ``prepare_llm_prompt``, at the research
   slots 09:00 / 15:00 America/New_York, after FinancialPlanning stored that slot's
   ``research_source``): a normal :class:`ModelingEvidence` research_input record,
   ``kind="llm_prompt"``, ``approach_id="direct_claude_news"``, ``slot=<research slot>``,
   deterministic ``input_id = mi_ + sha256(RFC8785([experiment, "direct_claude_news",
   research_slot, "llm_prompt"]))``. Its payload (S3 artifact, also mirrored to DynamoDB
   as ``content#<artifact_key>#<part>`` items, gzip of RFC 8785 JSON) is
   :data:`PROMPT_SCHEMA_VERSION`; ``payload["prompt"]`` is the exact UTF-8 text to send on
   stdin and ``payload["prompt_checksum"]`` its ``sha256:<hex>``. The prompt is exactly the
   reviewer prompt :func:`finplan_model.forward.research.prepare_review` builds for the
   Codex reviewer (interim, no forecast), from the newest ``research_source`` at or before
   the research slot and the CURRENT holdings of the experiment's direct_claude_news book,
   followed by the response JSON schema (Codex receives it as ``--output-schema``).

2. **llm_response** (written by ``ops/llm-autoschedule/claude_review.py`` on the dev box,
   stdlib + boto3 only): one raw DynamoDB item, NOT a ModelingEvidence record (it has no
   GSI attributes, like the ``artifact_content`` items)::

     pk            S  "llm_response#" + response_id
     record_type   S  "llm_response"
     schema_version S "finplan-llm-response/1"
     response_id   S  mi_ + sha256(json.dumps([prompt_id, "llm_response"], separators=(",", ":")))
     prompt_id     S  input_id of the llm_prompt record
     experiment_id S, approach_id S ("direct_claude_news"), research_slot S (UTC "...Z")
     responded_at  S  UTC "...Z" when the CLI finished
     body          S  canonical JSON (sort_keys, separators (",", ":"), ensure_ascii=False)
     body_checksum S  "sha256:" + sha256(body UTF-8)

   ``body`` holds: schema_version, response_id, prompt_id, prompt_input_checksum (the
   prompt record's ``checksum``), prompt_checksum, experiment_id, approach_id,
   research_slot, requested_at, responded_at, claude_version, command (argv without the
   prompt), exit_code, status (``completed`` | ``cli_failed`` | ``invalid_json``),
   final_text, final_text_checksum, parsed (the JSON object or null), parse_error,
   cli_result (subtype, is_error, num_turns, duration_ms, total_cost_usd, usage, model_usage).
   The item is written once (``attribute_not_exists(pk)``) and must stay below 350 KB.

The decision slot (09:45 / 15:45) uses the newest response whose research slot and
``responded_at`` are at or before the decision cutoff and no older than 7 hours, after
strict validation (:func:`validate_allocation`). Otherwise the book records
``unavailable`` with the searched slots. The model input captures the exact prompt and
response references (record IDs and checksums) and the exact prompt text.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from datetime import timedelta
from typing import Any, Callable, Mapping

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum

from .research import MAX_REVIEW_BYTES, VERSION as REVIEW_PROMPT_VERSION, prepare_review, stamp
from .schedule import format_slot, parse_slot, research_slots_between, validate_research_slot

APPROACH_ID = "direct_claude_news"
PROMPT_KIND = "llm_prompt"
PROMPT_SCHEMA_VERSION = "finplan-llm-prompt/1"
RESPONSE_RECORD_TYPE = "llm_response"
RESPONSE_SCHEMA_VERSION = "finplan-llm-response/1"
RESPONSE_KEY_PREFIX = "llm_response#"
MAX_RESPONSE_AGE_SECONDS = 7 * 3600
#: Research slots searched backwards for the newest research_source of a prompt.
RESEARCH_LOOKBACK = timedelta(days=4)
TOOL_POLICY = {"tools": "none", "cli_flags": ["-p", "--output-format", "json", "--tools", "", "--strict-mcp-config", "--safe-mode", "--no-session-persistence", "--disable-slash-commands"],
               "web_search": "disabled", "mcp_servers": [], "session_persistence": False,
               "input": "prompt on stdin", "working_directory": "empty temporary directory"}
SCHEMA_PREAMBLE = "\nRESPONSE JSON SCHEMA (return exactly one JSON object conforming to it, no prose, no code fence):\n"


def _identifier(prefix: str, *parts: str) -> str:
    return prefix + "_" + hashlib.sha256(canonical_json_bytes(list(parts))).hexdigest()


def prompt_id(experiment: str, research_slot: str) -> str:
    return _identifier("mi", experiment, APPROACH_ID, research_slot, PROMPT_KIND)


def response_id(prompt_input_id: str) -> str:
    return _identifier("mi", prompt_input_id, RESPONSE_RECORD_TYPE)


def response_key(prompt_input_id: str) -> str:
    return RESPONSE_KEY_PREFIX + response_id(prompt_input_id)


def research_source_id(experiment: str, research_slot: str) -> str:
    """FinancialPlanning's research_source record (collect_modeling_sources.py --kind research)."""
    return _identifier("mi", experiment + "-interim", "__research_source__", research_slot, "research_source")


def _latest_research_source(backend: Any, experiment: str, research_slot: str) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    end = parse_slot(research_slot)
    for slot in research_slots_between(format_slot(end - RESEARCH_LOOKBACK), research_slot):
        rid = research_source_id(experiment, slot)
        if backend.repo.get("research_input", rid):
            captured = backend.read_input(rid)
            return slot, captured["metadata"], captured["payload"]
    return None


def prepare_llm_prompt(backend: Any, experiment: str, research_slot: str, *, comparison_policy: Mapping[str, Any], cohort: str | None = None,
                       load_inputs: Callable[..., tuple[dict[str, Any], dict[str, Any]]] | None = None, now: str | None = None) -> dict[str, Any]:
    """Persist the exact Claude reviewer prompt for ``research_slot`` (idempotent)."""
    from .hourly import _portfolio_holdings, _processing_versions, load_current_inputs

    slot = validate_research_slot(research_slot)
    pid = prompt_id(experiment, slot)
    existing = backend.repo.get("research_input", pid)
    if existing:
        return {"status": "exists", "experiment_id": experiment, "research_slot": slot, "prompt_id": pid, "checksum": existing.doc["checksum"]}
    found = _latest_research_source(backend, experiment, slot)
    if found is None:
        return {"status": "unavailable", "experiment_id": experiment, "research_slot": slot, "prompt_id": pid,
                "reason": "no research_source collected at or before the research slot (4-day lookback)"}
    source_slot, source_meta, data = found
    if data.get("status") == "failed" or data.get("acquisition_failure"):
        return {"status": "unavailable", "experiment_id": experiment, "research_slot": slot, "prompt_id": pid, "research_source_slot": source_slot,
                "reason": "research acquisition failed: " + str(data.get("acquisition_failure", data.get("reason")))}
    at = now or stamp()
    source, _ = (load_inputs or load_current_inputs)(cohort=cohort)
    source = {**source,
              "news": [n for n in data.get("news", []) if n.get("published_at") and n["published_at"] <= at and n.get("available_at", "9999") <= at
                       and n.get("quality") != "quarantined_missing_publication_time"],
              "research_universe": data.get("universe"), "research_coverage": data.get("coverage", []), "research_workers": data.get("workers", []),
              "research_available_at": data.get("available_at"), "availability_cutoff": at, "decision_time": at, "scheduled_slot": slot}
    book = backend.ensure_portfolio(experiment, APPROACH_ID, comparison_policy["initial_capital"], source["investable_universe"], at)
    holdings = _portfolio_holdings(book, source)
    prepared = prepare_review(source, holdings, allow_forecast=False, destination="direct_claude")
    schema = prepared["response_schema"]
    prompt = prepared["prompt"] + SCHEMA_PREAMBLE + canonical_json_bytes(schema).decode()
    if len(prompt.encode()) > MAX_REVIEW_BYTES:
        return {"status": "unavailable", "experiment_id": experiment, "research_slot": slot, "prompt_id": pid,
                "reason": "reviewer prompt exceeds the declared bounded byte budget"}
    eligible = sorted({row["url"] for row in prepared["selected"] if row.get("url")} | ({prepared["actual_source"]["research_universe"]["source_url"]} if isinstance(prepared["actual_source"]["research_universe"].get("source_url"), str) and prepared["actual_source"]["research_universe"]["source_url"] else set()))
    payload = {"schema_version": PROMPT_SCHEMA_VERSION, "experiment_id": experiment, "approach_id": APPROACH_ID, "research_slot": slot,
               "status": "pending", "availability_cutoff": at, "prepared_at": at,
               "prompt": prompt, "prompt_checksum": sha256_checksum(prompt.encode()), "prompt_bytes": len(prompt.encode()),
               "review_request": {"boundary": "claude_code_print", "prompt_version": REVIEW_PROMPT_VERSION, "messages": [{"role": "user", "content": prompt}],
                                  "schema": schema, "forecast_scope": "interim_nonexecuted_research_proposal", "actual_source": prepared["actual_source"],
                                  "review_prompt_checksum": sha256_checksum(prepared["prompt"].encode()),
                                  "review_prompt_semantics": "exact research.prepare_review prompt (as sent to the Codex reviewer), followed by the response schema"},
               "eligible_citations": eligible,
               "research_source_ref": {"input_id": source_meta["input_id"], "checksum": source_meta["checksum"], "slot": source_slot},
               "holdings": holdings, "portfolio_id": book["portfolio_id"], "portfolio_revision": book["revision"],
               "tool_policy": copy.deepcopy(TOOL_POLICY), "max_response_age_seconds": MAX_RESPONSE_AGE_SECONDS,
               "response_contract": {"record_type": RESPONSE_RECORD_TYPE, "schema_version": RESPONSE_SCHEMA_VERSION, "key": response_key(pid), "response_id": response_id(pid)},
               "code_versions": _processing_versions(), "trigger": os.environ.get("FINPLAN_EXPERIMENT_TRIGGER", "manual_validation")}
    item = backend.capture_input(experiment, APPROACH_ID, slot, payload, None, at, kind=PROMPT_KIND)
    return {"status": "prepared", "experiment_id": experiment, "research_slot": slot, "prompt_id": item["input_id"], "checksum": item["checksum"],
            "prompt_checksum": payload["prompt_checksum"], "research_source_slot": source_slot, "news_count": len(prepared["selected"])}


# ----------------------------------------------------------------------------- responses
def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_allocation(parsed: Any, universe: list[str], eligible_citations: list[str] | None = None) -> dict[str, Any]:
    """Strict LLM allocation: only the universe, weights in [0, 1], sum <= 1, remainder cash."""
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    rows = parsed.get("target_weights")
    if not isinstance(rows, list):
        raise ValueError("target_weights must be a list")
    weights: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"instrument_id", "weight"}:
            raise ValueError("each target weight must be exactly {instrument_id, weight}")
        instrument, weight = row["instrument_id"], row["weight"]
        if not isinstance(instrument, str) or instrument not in universe:
            raise ValueError(f"instrument outside the experiment universe: {instrument!r}")
        if instrument in weights:
            raise ValueError(f"duplicate instrument {instrument}")
        if not _number(weight) or not 0 <= weight <= 1:
            raise ValueError(f"weight of {instrument} must be a finite number in [0, 1]")
        weights[instrument] = float(weight)
    total = sum(weights.values())
    if total > 1 + 1e-9:
        raise ValueError("target weights sum above one")
    cash = max(0.0, 1.0 - total)
    supplied = parsed.get("cash_weight")
    if supplied is not None and (not _number(supplied) or not 0 <= supplied <= 1 or abs(supplied - cash) > 1e-4):
        raise ValueError("cash_weight must equal one minus the target weights")
    status = parsed.get("status", "recommended")
    if status not in ("recommended", "hold"):
        raise ValueError("status must be recommended or hold")
    rationale = parsed.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("a rationale is required")
    citations = parsed.get("citations", [])
    if not isinstance(citations, list) or any(not isinstance(c, str) for c in citations):
        raise ValueError("citations must be a list of URLs")
    if eligible_citations is not None and set(citations) - set(eligible_citations):
        raise ValueError("response cites a source that was not supplied")
    forecast = parsed.get("forecast")
    if forecast is not None and (not isinstance(forecast, dict) or forecast.get("status") != "not_available"):
        raise ValueError("an interim research response may not supply a portfolio forecast")
    return {"status": status, "target_weights": [{"instrument_id": i, "weight": weights.get(i, 0.0)} for i in universe],
            "cash_weight": cash, "rationale": rationale, "citations": list(citations)}


def decode_response_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Body of a raw llm_response DynamoDB item (attribute-value form), checksum verified."""
    def text(name: str) -> str:
        value = item.get(name)
        if not isinstance(value, Mapping) or not isinstance(value.get("S"), str):
            raise ValueError(f"llm_response item lacks string attribute {name}")
        return value["S"]
    if text("record_type") != RESPONSE_RECORD_TYPE or text("schema_version") != RESPONSE_SCHEMA_VERSION:
        raise ValueError("not a finplan-llm-response/1 item")
    raw = text("body")
    if "sha256:" + hashlib.sha256(raw.encode()).hexdigest() != text("body_checksum"):
        raise ValueError("llm_response body checksum mismatch")
    body = json.loads(raw)
    for name in ("response_id", "prompt_id", "experiment_id", "approach_id", "research_slot", "responded_at"):
        if body.get(name) != text(name):
            raise ValueError(f"llm_response attribute {name} does not match its body")
    if text("pk") != RESPONSE_KEY_PREFIX + body["response_id"]:
        raise ValueError("llm_response key does not match its body")
    return body


def _default_reader(backend: Any) -> Callable[[str], Mapping[str, Any] | None]:
    def read(key: str) -> Mapping[str, Any] | None:
        table = backend.repo.names["research_input"]
        return backend.repo.client.get_item(TableName=table, Key={"pk": {"S": key}}, ConsistentRead=True).get("Item")
    return read


def select_response(backend: Any, experiment: str, cutoff: str, universe: list[str], *, read_item: Callable[[str], Mapping[str, Any] | None] | None = None) -> dict[str, Any]:
    """Newest valid response at or before ``cutoff`` (<= 7 h old) with its exact references."""
    read_item = read_item or _default_reader(backend)
    deadline = parse_slot(cutoff)
    searched: list[dict[str, Any]] = []
    for slot in research_slots_between(format_slot(deadline - timedelta(seconds=MAX_RESPONSE_AGE_SECONDS)), format_slot(deadline)):
        pid = prompt_id(experiment, slot)
        record = backend.repo.get("research_input", pid)
        if not record:
            searched.append({"research_slot": slot, "prompt_id": pid, "status": "no_prompt"})
            continue
        prompt_doc = record.doc
        key = response_key(pid)
        item = read_item(key)
        if not item:
            searched.append({"research_slot": slot, "prompt_id": pid, "status": "no_response"})
            continue
        try:
            body = decode_response_item(item)
            if (body["prompt_id"], body["experiment_id"], body["approach_id"], body["research_slot"], body["response_id"]) != (pid, experiment, APPROACH_ID, slot, response_id(pid)):
                raise ValueError("response belongs to another prompt")
            if body.get("prompt_input_checksum") != prompt_doc["checksum"]:
                raise ValueError("response was produced for different prompt bytes")
            responded, requested = parse_slot(body["responded_at"]), parse_slot(body["requested_at"])
            if responded > deadline:
                raise ValueError("response became available after the decision cutoff")
            if requested < parse_slot(prompt_doc["captured_at"]) or responded < requested:
                raise ValueError("response timestamps precede the prompt")
            if body.get("status") != "completed" or body.get("exit_code") != 0:
                raise ValueError("reviewer did not complete: " + str(body.get("status")))
            if body.get("final_text_checksum") != "sha256:" + hashlib.sha256(str(body.get("final_text")).encode()).hexdigest():
                raise ValueError("final text checksum mismatch")
            prompt_payload = backend.read_input(pid)["payload"]
            if body.get("prompt_checksum") != prompt_payload["prompt_checksum"] or prompt_payload["prompt_checksum"] != sha256_checksum(prompt_payload["prompt"].encode()):
                raise ValueError("response prompt checksum does not match the stored prompt")
            allocation = validate_allocation(body.get("parsed"), universe, prompt_payload.get("eligible_citations"))
        except (ValueError, KeyError, TypeError) as exc:
            searched.append({"research_slot": slot, "prompt_id": pid, "status": "invalid_response", "reason": str(exc)})
            continue
        searched.append({"research_slot": slot, "prompt_id": pid, "status": "selected"})
        return {"status": "selected", "research_slot": slot, "prompt_id": pid, "prompt_doc": prompt_doc, "prompt_payload": prompt_payload,
                "response_key": key, "response_body": body, "body_checksum": item["body_checksum"]["S"], "allocation": allocation, "searched": searched}
    return {"status": "unavailable", "searched": searched}


def claude_adapter(backend: Any, experiment: str, *, read_item: Callable[[str], Mapping[str, Any] | None] | None = None):
    """``external_adapters['claude_offline']`` of :func:`finplan_model.forward.runtime.evaluate_approach`."""
    def adapter(base, source, holdings, before):
        cutoff = base["availability_cutoff"]
        universe = list(source["investable_universe"])
        found = select_response(backend, experiment, cutoff, universe, read_item=read_item)
        if found["status"] != "selected":
            before({"framework": "claude_code_cli_offline", "boundary": "not_invoked_no_eligible_llm_response", "invocation_state": "not_invoked",
                    "max_response_age_seconds": MAX_RESPONSE_AGE_SECONDS, "searched": found["searched"], "actual_source": {"news": []}, "messages": []})
            reasons = "; ".join(f"{row['research_slot']}: {row['status']}" + (f" ({row['reason']})" if row.get("reason") else "") for row in found["searched"]) or "no research slot in window"
            return {"status": "unavailable", "reason": "no valid Claude review response within 7 h before the cutoff: " + reasons, "forecast": {"available": False}}
        prompt_payload, body, doc = found["prompt_payload"], found["response_body"], found["prompt_doc"]
        before({"framework": "claude_code_cli_offline", "boundary": "offline_claude_review_response", "invocation_state": "invoked_offline_before_cutoff",
                "prompt_ref": {"input_id": found["prompt_id"], "checksum": doc["checksum"], "artifact_key": doc.get("artifact_key"), "kind": PROMPT_KIND,
                               "research_slot": found["research_slot"], "captured_at": doc["captured_at"], "prompt_checksum": prompt_payload["prompt_checksum"]},
                "response_ref": {"table": "research_input", "pk": found["response_key"], "response_id": body["response_id"], "body_checksum": found["body_checksum"],
                                 "final_text_checksum": body["final_text_checksum"], "requested_at": body["requested_at"], "responded_at": body["responded_at"],
                                 "claude_version": body.get("claude_version")},
                "messages": copy.deepcopy(prompt_payload["review_request"]["messages"]),
                "actual_source": copy.deepcopy(prompt_payload["review_request"]["actual_source"]),
                "research_source_ref": copy.deepcopy(prompt_payload["research_source_ref"]),
                "holdings_at_prompt": {"portfolio_id": prompt_payload["portfolio_id"], "revision": prompt_payload["portfolio_revision"]},
                "tool_policy": copy.deepcopy(prompt_payload["tool_policy"]), "searched": found["searched"]})
        allocation = found["allocation"]
        cli = body.get("cli_result") or {}
        return {**allocation, "allocation_origin": "Claude Code offline review of the stored prompt " + found["prompt_id"],
                "provider_final_response_text": body["final_text"], "provider_final_response_checksum": body["final_text_checksum"],
                "reviewer_invocation": {"reviewer": "offline_claude_code_cli", "claude_version": body.get("claude_version"), "requested_at": body["requested_at"],
                                        "responded_at": body["responded_at"], "research_slot": found["research_slot"], "prompt_id": found["prompt_id"],
                                        "response_id": body["response_id"], "num_turns": cli.get("num_turns"), "duration_ms": cli.get("duration_ms")},
                "usage": {"cost_usd": None, "reported_api_equivalent_usd": cli.get("total_cost_usd"),
                          "cost_basis": "existing Claude Code subscription on the dev box; the CLI-reported figure is an API-equivalent estimate"},
                "holdings_revision_at_prompt": prompt_payload["portfolio_revision"],
                "forecast": {"available": False, "status": "not_available", "reason": "A target allocation is not a portfolio value forecast"}}
    return adapter
