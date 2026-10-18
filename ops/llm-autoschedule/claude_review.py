#!/usr/bin/env python3
"""Answer the newest pending direct_claude_news prompt with Claude Code (dev box only).

Standalone on purpose: stdlib + boto3, no finplan imports, so it runs under its systemd timer
from any checkout. The FinanceModel forward runner (Lambda) writes the prompt; this script
writes the response; the runner's next decision slot reads it. Protocol and item shapes:
``finplan_model.forward.llm_review`` (kept in sync by tests/unit/forward/test_claude_review_script.py).

Work discovery (one prompt per run, idempotent):

1. Query the FinancialPlanning ``research_input`` table's ``approach-index`` for
   ``approach_key = "<experiment>#direct_claude_news"``, newest ``time_key`` first, and take
   the newest item with ``kind = "llm_prompt"`` (time_key starts with its research slot).
2. Skip it when its research slot is older than ``--max-age-seconds`` (default 6 h; the runner
   accepts responses up to 7 h old) or when ``llm_response#<response_id>`` already exists.
3. Read the prompt record (strongly consistent ``GetItem`` on ``pk = <prompt input_id>``),
   then its payload from the DynamoDB content copy ``content#<artifact_key>#<part>``
   (gzip of RFC 8785 JSON, ``sha256`` verified against the record's ``checksum``), and verify
   ``payload.prompt_checksum``.
4. Run ``claude -p --output-format json --tools "" --strict-mcp-config --safe-mode
   --no-session-persistence --disable-slash-commands`` with the prompt on stdin, in an empty
   temporary directory, all tools disabled.
5. Write ONE item (``attribute_not_exists(pk)``): ``pk = "llm_response#" + response_id``,
   ``response_id = "mi_" + sha256(json.dumps([prompt_id, "llm_response"], separators=(",", ":")))``,
   with the canonical JSON ``body`` and its ``body_checksum`` (see :func:`build_item`).

Exit codes: 0 answered or nothing to do, 1 failure (the item is still written when the CLI
itself failed, so the runner reports a clear reason), 2 usage error.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone

APPROACH_ID = "direct_claude_news"
PROMPT_KIND = "llm_prompt"
RECORD_TYPE = "llm_response"
SCHEMA_VERSION = "finplan-llm-response/1"
KEY_PREFIX = "llm_response#"
DEFAULT_TABLE = "finplan-beta-financialplanning-research-input"
DEFAULT_EXPERIMENT = "beta-competition-v1"
CLAUDE_FLAGS = ["-p", "--output-format", "json", "--tools", "", "--strict-mcp-config", "--safe-mode", "--no-session-persistence", "--disable-slash-commands"]
MAX_ITEM_BYTES = 350_000
CLI_RESULT_FIELDS = ("subtype", "is_error", "num_turns", "duration_ms", "total_cost_usd", "usage", "stop_reason", "terminal_reason", "permission_denials")


def stamp(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("aware timestamp required")
    return parsed.astimezone(timezone.utc)


def sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def response_id(prompt_id: str) -> str:
    # Same as finplan_model.forward.llm_review.response_id (RFC 8785 of an ASCII string list).
    return "mi_" + hashlib.sha256(json.dumps([prompt_id, RECORD_TYPE], separators=(",", ":")).encode()).hexdigest()


def find_newest_prompt(client, table: str, experiment: str):
    """Newest llm_prompt index row of the experiment's direct_claude_news book, or None."""
    token = None
    for _ in range(5):
        kwargs = {"TableName": table, "IndexName": "approach-index", "KeyConditionExpression": "approach_key = :k",
                  "ExpressionAttributeValues": {":k": {"S": f"{experiment}#{APPROACH_ID}"}}, "ScanIndexForward": False, "Limit": 25}
        if token:
            kwargs["ExclusiveStartKey"] = token
        page = client.query(**kwargs)
        for item in page.get("Items", []):
            if item.get("kind", {}).get("S") == PROMPT_KIND:
                return item
        token = page.get("LastEvaluatedKey")
        if not token:
            return None
    return None


def read_prompt(client, table: str, prompt_id: str):
    """(record doc, verified payload) of an llm_prompt record."""
    item = client.get_item(TableName=table, Key={"pk": {"S": prompt_id}}, ConsistentRead=True).get("Item")
    if not item:
        raise RuntimeError(f"prompt record {prompt_id} not found")
    doc = json.loads(item["doc"]["S"])
    if doc.get("kind") != PROMPT_KIND or doc.get("approach_id") != APPROACH_ID or doc.get("input_id") != prompt_id:
        raise RuntimeError("record is not a direct_claude_news llm_prompt")
    key, chunks, total = doc["artifact_key"], [], None
    while total is None or len(chunks) < total:
        part = client.get_item(TableName=table, Key={"pk": {"S": f"content#{key}#{len(chunks):04d}"}}, ConsistentRead=True).get("Item")
        if not part:
            raise RuntimeError("prompt content is not mirrored in DynamoDB")
        total = int(part["parts"]["N"])
        chunks.append(bytes(part["data"]["B"]))
    raw = gzip.decompress(b"".join(chunks))
    if sha256(raw) != doc["checksum"]:
        raise RuntimeError("prompt content checksum mismatch")
    payload = json.loads(raw)
    if sha256(payload["prompt"].encode()) != payload["prompt_checksum"]:
        raise RuntimeError("prompt text checksum mismatch")
    return doc, payload


def parse_final_text(text: str):
    """The JSON object in Claude's final text (a surrounding code fence is tolerated)."""
    body = text.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    value = json.loads(body)
    if not isinstance(value, dict):
        raise ValueError("final text is not a JSON object")
    return value


def run_claude(prompt: str, *, claude: str = "claude", timeout: int = 900, runner=subprocess.run) -> dict:
    version = runner([claude, "--version"], capture_output=True, text=True, timeout=30, check=False).stdout.strip()
    workdir = tempfile.mkdtemp(prefix="claude-review-")
    try:
        requested = stamp()
        proc = runner([claude, *CLAUDE_FLAGS], input=prompt, capture_output=True, text=True, timeout=timeout, cwd=workdir, check=False)
        responded = stamp()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return {"claude_version": version, "requested_at": requested, "responded_at": responded, "exit_code": proc.returncode, "stdout": proc.stdout}


def build_item(doc: dict, payload: dict, outcome: dict, *, command=None) -> dict:
    """The llm_response DynamoDB item (attribute-value form) for one CLI outcome."""
    prompt_id = doc["input_id"]
    rid = response_id(prompt_id)
    status, final_text, parsed, parse_error, cli = "completed", "", None, None, {}
    try:
        result = json.loads(outcome["stdout"])
        cli = {k: result.get(k) for k in CLI_RESULT_FIELDS if k in result}
        cli["model_usage"] = {name: {k: v for k, v in (usage or {}).items() if k in ("inputTokens", "outputTokens", "costUSD", "canonicalModel")}
                              for name, usage in (result.get("modelUsage") or {}).items()}
        final_text = result.get("result") or ""
        if outcome["exit_code"] != 0 or result.get("is_error"):
            status = "cli_failed"
    except (TypeError, ValueError):
        status, parse_error = "cli_failed", "CLI output is not JSON"
    if status == "completed":
        try:
            parsed = parse_final_text(final_text)
        except ValueError as exc:
            status, parse_error = "invalid_json", str(exc)[:500]
    body = {"schema_version": SCHEMA_VERSION, "response_id": rid, "prompt_id": prompt_id, "prompt_input_checksum": doc["checksum"],
            "prompt_checksum": payload["prompt_checksum"], "experiment_id": payload["experiment_id"], "approach_id": APPROACH_ID,
            "research_slot": payload["research_slot"], "requested_at": outcome["requested_at"], "responded_at": outcome["responded_at"],
            "claude_version": outcome["claude_version"], "command": list(command or ["claude", *CLAUDE_FLAGS]), "exit_code": outcome["exit_code"],
            "status": status, "final_text": final_text, "final_text_checksum": sha256(final_text.encode()), "parsed": parsed,
            "parse_error": parse_error, "cli_result": cli}
    raw = canonical(body)
    if len(raw.encode()) > MAX_ITEM_BYTES:
        body.update(final_text="", final_text_checksum=sha256(b""), parsed=None, status="invalid_json", parse_error="response exceeds the 350 KB item bound")
        raw = canonical(body)
    s = lambda v: {"S": v}  # noqa: E731
    return {"pk": s(KEY_PREFIX + rid), "record_type": s(RECORD_TYPE), "schema_version": s(SCHEMA_VERSION), "response_id": s(rid),
            "prompt_id": s(prompt_id), "experiment_id": s(body["experiment_id"]), "approach_id": s(APPROACH_ID),
            "research_slot": s(body["research_slot"]), "responded_at": s(body["responded_at"]), "body": s(raw), "body_checksum": s(sha256(raw.encode()))}


def review_once(client, *, table: str = DEFAULT_TABLE, experiment: str = DEFAULT_EXPERIMENT, max_age_seconds: int = 6 * 3600,
                now: datetime | None = None, run=run_claude) -> dict:
    row = find_newest_prompt(client, table, experiment)
    if row is None:
        return {"status": "idle", "reason": "no llm_prompt record"}
    prompt_id = row["pk"]["S"]
    slot = row["time_key"]["S"].split("#", 1)[0]
    if parse_time(slot) < (now or datetime.now(timezone.utc)) - timedelta(seconds=max_age_seconds):
        return {"status": "idle", "reason": "newest prompt is stale", "prompt_id": prompt_id, "research_slot": slot}
    key = KEY_PREFIX + response_id(prompt_id)
    if client.get_item(TableName=table, Key={"pk": {"S": key}}, ConsistentRead=True).get("Item"):
        return {"status": "idle", "reason": "already answered", "prompt_id": prompt_id}
    doc, payload = read_prompt(client, table, prompt_id)
    outcome = run(payload["prompt"])
    item = build_item(doc, payload, outcome)
    try:
        client.put_item(TableName=table, Item=item, ConditionExpression="attribute_not_exists(pk)")
    except Exception as exc:  # noqa: BLE001 - a concurrent writer answered first
        if getattr(exc, "response", {}).get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            raise
        return {"status": "idle", "reason": "answered concurrently", "prompt_id": prompt_id}
    body = json.loads(item["body"]["S"])
    return {"status": body["status"], "prompt_id": prompt_id, "response_key": key, "research_slot": body["research_slot"], "responded_at": body["responded_at"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--table", default=os.environ.get("FINPLAN_RESEARCH_INPUT_TABLE", DEFAULT_TABLE))
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--max-age-seconds", type=int, default=6 * 3600)
    ap.add_argument("--claude", default=os.environ.get("CLAUDE_BIN", "claude"))
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--region", default="us-east-2")
    args = ap.parse_args(argv)
    import boto3

    client = boto3.client("dynamodb", region_name=args.region)
    result = review_once(client, table=args.table, experiment=args.experiment, max_age_seconds=args.max_age_seconds,
                         run=lambda prompt: run_claude(prompt, claude=args.claude, timeout=args.timeout))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in ("idle", "completed") else 1


if __name__ == "__main__":
    sys.exit(main())
