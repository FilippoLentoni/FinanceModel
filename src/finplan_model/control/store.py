"""Run metadata store: run records, append-only transition events, idempotency records and leases
(design D1, D3). Research run metadata only; never authoritative plan state.

Two implementations share :class:`RunStore`:

* :class:`InMemoryRunStore` - unit tests and local runs (one lock, deep copies);
* :class:`DynamoRunStore` - the deployed single-table DynamoDB design (:data:`TABLE_SPEC`), written
  with conditional writes and transactions so late or duplicate events can never move a run twice
  and an idempotency record and its run are created atomically.

Item layout (``pk`` / ``sk``):

=====================================  ===========================  =============================
item                                   pk                           sk
=====================================  ===========================  =============================
run record (JSON ``doc``, ``revision``) ``RUN#<run_id>``             ``RUN``
transition event (append-only)          ``RUN#<run_id>``             ``EVT#<seq:06d>``
idempotency record (``ttl``)            ``IDEM#<sha256 of scope>``   ``IDEM``
lease slot                              ``LEASE#<env>#<class>``      ``SLOT#<n>``
=====================================  ===========================  =============================

Run records carry ``gsi1pk = STATE#<state>`` / ``gsi2pk = RUNS`` with ``<submitted_at>#<run_id>``
sort keys (indexes ``by_state`` and ``by_submitted``) for the queue, ``list_jobs`` and budget sums.

Private forward context heads use ``FORWARD_CONTEXT#<context_id> / HEAD`` and
are CAS-created in the same transaction as their managed job. Permanent cost
outbox items use ``MANAGED_COST#<event_id> / EVENT``; only undelivered items carry
``gsi1pk = FORWARD_COST#<experiment_id>``. They have no TTL or public RUNS index.
Acknowledgement retains the original event and removes only its pending index.
"""

from __future__ import annotations

import copy
import base64
import hashlib
import json
import math
import threading
from datetime import datetime, timezone
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ConditionFailed",
    "DynamoRunStore",
    "InMemoryRunStore",
    "RunStore",
    "TABLE_SPEC",
    "idempotency_scope_key",
]

#: The DynamoDB table the control plane needs (consumed by ``infra`` task 10.3).
TABLE_SPEC: dict[str, Any] = {
    "logical_role": "runs-table",
    "partition_key": {"name": "pk", "type": "S"},
    "sort_key": {"name": "sk", "type": "S"},
    "global_secondary_indexes": [
        {"name": "by_state", "partition_key": {"name": "gsi1pk", "type": "S"}, "sort_key": {"name": "gsi1sk", "type": "S"}, "projection": "ALL"},
        {"name": "by_submitted", "partition_key": {"name": "gsi2pk", "type": "S"}, "sort_key": {"name": "gsi2sk", "type": "S"}, "projection": "ALL"},
    ],
    "ttl_attribute": "ttl",
    "billing_mode": "PAY_PER_REQUEST",
    "point_in_time_recovery": True,
    "stream": None,
}


class ConditionFailed(Exception):
    """A conditional write lost (the item changed or already exists)."""


def idempotency_scope_key(principal: str, environment: str, operation: str, idempotency_key: str) -> str:
    """Stable store key for the contract idempotency scope ``(principal, environment, operation, key)``."""
    raw = "\x1f".join((principal, environment, operation, idempotency_key)).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sort_key(run: Mapping[str, Any]) -> str:
    return f"{run['submitted_at']}#{run['run_id']}"


def _cost_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("managed cost timestamps require a timezone")
    return parsed.astimezone(timezone.utc)


def _cost_key(event: Mapping[str, Any]) -> str:
    return _cost_time(event["occurred_at"]).isoformat(timespec="microseconds") + "#" + event["event_id"]


def _cost_scope(experiment: str) -> str:
    if not isinstance(experiment, str) or not 1 <= len(experiment) <= 256 or any(ord(c) < 32 for c in experiment):
        raise ValueError("invalid managed cost experiment scope")
    return "FORWARD_COST#" + experiment


def _cost_event(run: Mapping[str, Any], event: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Validate private outbox identity before the terminal run transaction."""
    from .states import is_terminal
    from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
    descriptor = run.get("forward_jev") or run.get("forward_qwen") or {}
    if (not is_terminal(run["state"]) or event.get("protocol") != "finplan-managed-model-cost/1"
            or event.get("event_id") != "managed:" + run["run_id"]
            or event.get("managed_run_id") != run["run_id"] or event.get("job_state") != run["state"]):
        raise ValueError("managed cost event must match its terminal managed run")
    for key in ("experiment_id", "approach_id", "portfolio_id", "binding_checksum", "binding_ref"):
        if not descriptor.get(key) or event.get(key) != descriptor[key]:
            raise ValueError("managed cost event does not match approved forward binding")
    _cost_scope(event["experiment_id"])
    if not (_cost_time(run["submitted_at"]) <= _cost_time(event["occurred_at"]) <= _cost_time(run["completed_at"])):
        raise ValueError("managed cost event time lies outside its actual job interval")
    for key in ("compute_observed_usd", "compute_estimated_usd", "vendor_observed_usd", "vendor_estimated_usd"):
        value = event.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
            raise ValueError("managed costs must be finite nonnegative values or unavailable")
    raw = canonical_json_bytes(dict(event))
    if len(raw) > 16 * 1024:
        raise ValueError("managed cost outbox metadata exceeds 16 KiB; use immutable artifact references")
    return copy.deepcopy(dict(event)), sha256_checksum(raw)


def _cost_token(experiment: str, cursor: Any) -> str:
    return base64.urlsafe_b64encode(json.dumps({"scope": _cost_scope(experiment), "cursor": cursor}, separators=(",", ":")).encode()).decode()


def _cost_cursor(experiment: str, token: str | None, limit: int) -> Any:
    _cost_scope(experiment)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("managed cost query page must contain 1..100 records")
    if token is None:
        return None
    try:
        if not isinstance(token, str) or len(token) > 8192:
            raise ValueError()
        value = json.loads(base64.b64decode(token, altchars=b"-_", validate=True))
        if value["scope"] != _cost_scope(experiment) or not value.get("cursor"):
            raise ValueError()
        return value["cursor"]
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        raise ValueError("managed cost cursor has invalid query scope") from exc


def _cost_ack(event: Mapping[str, Any], observation_ref: Mapping[str, Any], recorded_at: str) -> dict[str, Any]:
    from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
    checksum = observation_ref.get("checksum")
    if (set(observation_ref) != {"observation_id", "checksum"} or not isinstance(observation_ref["observation_id"], str)
            or not observation_ref["observation_id"].startswith("mo_") or not isinstance(checksum, str)
            or len(checksum) != 71 or not checksum.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in checksum[7:])):
        raise ValueError("managed cost acknowledgement requires an immutable observation reference")
    slot = _cost_time(event["occurred_at"]).isoformat().replace("+00:00", "Z")
    expected_id = "mo_" + sha256_checksum(canonical_json_bytes([event["experiment_id"], event["approach_id"],
        slot, "managed_model_cost:" + event["managed_run_id"]])).split(":", 1)[1]
    if observation_ref["observation_id"] != expected_id or checksum != sha256_checksum(canonical_json_bytes(dict(event))):
        raise ValueError("managed cost acknowledgement does not bind the original event bytes")
    if _cost_time(recorded_at) < _cost_time(event["occurred_at"]):
        raise ValueError("managed cost acknowledgement precedes actual event")
    return {"observation_ref": dict(observation_ref), "recorded_at": recorded_at}


def _forward_context_doc(run: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    descriptor = run.get("forward_jev") or run.get("forward_qwen") or {}
    if (context.get("managed_run_id") != run["run_id"] or not context.get("context_id")
            or not isinstance(context["context_id"], str) or len(context["context_id"]) > 256):
        raise ValueError("invalid forward managed context identity")
    for key in ("context_id", "binding_ref", "binding_checksum"):
        if not descriptor.get(key) or context.get(key) != descriptor[key]:
            raise ValueError("forward context differs from managed run binding")
    if "previous_managed_run_id" not in context:
        raise ValueError("forward context requires explicit predecessor identity")
    previous = context["previous_managed_run_id"]
    if previous is not None and (not isinstance(previous, str) or not previous or previous == run["run_id"]):
        raise ValueError("invalid forward context predecessor")
    return copy.deepcopy(dict(context))


@runtime_checkable
class RunStore(Protocol):
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...

    def create_run(self, run: Mapping[str, Any], events: Sequence[Mapping[str, Any]], idempotency: Mapping[str, Any] | None = None, *, forward_context: Mapping[str, Any] | None = None) -> None:
        """Create a run (revision 0) with its first events and, atomically, its idempotency record.

        Raises :class:`ConditionFailed` when the run or the idempotency record already exists."""
        ...

    def get_forward_context(self, context_id: str) -> dict[str, Any] | None: ...

    def update_run(self, run: Mapping[str, Any], expected_revision: int, events: Sequence[Mapping[str, Any]] = (), *, forward_cost_event: Mapping[str, Any] | None = None) -> None:
        """Replace the run if its stored revision equals ``expected_revision`` and append ``events``."""
        ...

    def query_forward_cost_events(self, experiment_id: str, *, limit: int = 100, next_token: str | None = None) -> tuple[list[dict[str, Any]], str | None]: ...

    def ack_forward_cost_event(self, experiment_id: str, event_id: str, observation_ref: Mapping[str, Any], recorded_at: str) -> None: ...

    def list_events(self, run_id: str) -> list[dict[str, Any]]: ...

    def query_runs(self, state: str | None = None, *, limit: int = 100, after: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
        """Runs in submission order (optionally one state); ``after`` is the previous page's cursor."""
        ...

    def get_idempotency(self, scope_key: str) -> dict[str, Any] | None: ...

    def put_idempotency(self, scope_key: str, record: Mapping[str, Any], ttl_epoch: int) -> None:
        """Create an idempotency record; :class:`ConditionFailed` if one exists."""
        ...

    def lease_slots(self, lease_key: str) -> list[dict[str, Any]]: ...

    def acquire_slot(self, lease_key: str, slot: int, holder: str, expires_at: str, acquired_at: str) -> bool: ...

    def renew_slot(self, lease_key: str, slot: int, holder: str, expires_at: str) -> bool: ...

    def release_slot(self, lease_key: str, slot: int, holder: str) -> bool: ...


def iter_runs(store: RunStore, state: str | None = None, page: int = 200) -> Iterator[dict[str, Any]]:
    after: str | None = None
    while True:
        runs, after = store.query_runs(state, limit=page, after=after)
        yield from runs
        if after is None:
            return


# ===================================================================== in memory
class InMemoryRunStore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.runs: dict[str, dict[str, Any]] = {}
        self.events: dict[str, dict[int, dict[str, Any]]] = {}
        self.idempotency: dict[str, dict[str, Any]] = {}
        self.leases: dict[str, dict[int, dict[str, Any]]] = {}
        self.audit: list[dict[str, Any]] = []
        self.forward_cost_events: dict[str, dict[str, Any]] = {}
        self.forward_contexts: dict[str, dict[str, Any]] = {}

    def append_audit(self, record: Mapping[str, Any]) -> None:
        """Append-only audit trail (production-strategy changes)."""
        with self._lock:
            self.audit.append(copy.deepcopy(dict(record)))

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self.runs.get(run_id)
            return copy.deepcopy(r) if r is not None else None

    def _append(self, run_id: str, events: Sequence[Mapping[str, Any]]) -> None:
        bucket = self.events.setdefault(run_id, {})
        for ev in events:
            if int(ev["seq"]) in bucket:
                raise ConditionFailed(f"event {ev['seq']} already exists")
        for ev in events:
            bucket[int(ev["seq"])] = copy.deepcopy(dict(ev))

    def get_forward_context(self, context_id: str) -> dict[str, Any] | None:
        with self._lock:
            return copy.deepcopy(self.forward_contexts.get(context_id))

    def create_run(self, run: Mapping[str, Any], events: Sequence[Mapping[str, Any]], idempotency: Mapping[str, Any] | None = None, *, forward_context: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            if run["run_id"] in self.runs:
                raise ConditionFailed("run exists")
            if idempotency is not None and idempotency["scope_key"] in self.idempotency:
                raise ConditionFailed("idempotency record exists")
            context = _forward_context_doc(run, forward_context) if forward_context is not None else None
            if context:
                old = self.forward_contexts.get(context["context_id"])
                if (old or {}).get("managed_run_id") != context["previous_managed_run_id"]:
                    raise ConditionFailed("forward context predecessor changed")
            self.runs[run["run_id"]] = copy.deepcopy(dict(run))
            self._append(run["run_id"], events)
            if idempotency is not None:
                self.idempotency[idempotency["scope_key"]] = copy.deepcopy(dict(idempotency))
            if context:
                self.forward_contexts[context["context_id"]] = context

    def update_run(self, run: Mapping[str, Any], expected_revision: int, events: Sequence[Mapping[str, Any]] = (), *, forward_cost_event: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            cur = self.runs.get(run["run_id"])
            if cur is None or int(cur["revision"]) != int(expected_revision):
                raise ConditionFailed("revision changed")
            bucket = self.events.get(run["run_id"], {})
            if any(int(ev["seq"]) in bucket for ev in events):
                raise ConditionFailed("event exists")
            cost = _cost_event(run, forward_cost_event) if forward_cost_event is not None else None
            if cost:
                old_cost = self.forward_cost_events.get(cost[0]["event_id"])
                if old_cost and old_cost["checksum"] != cost[1]:
                    raise ConditionFailed("managed cost event is immutable")
            self.runs[run["run_id"]] = copy.deepcopy(dict(run))
            self._append(run["run_id"], events)
            if cost:
                self.forward_cost_events.setdefault(cost[0]["event_id"], {"doc": cost[0], "checksum": cost[1]})

    def query_forward_cost_events(self, experiment_id: str, *, limit: int = 100, next_token: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
        after = _cost_cursor(experiment_id, next_token, limit)
        if after is not None and not isinstance(after, str):
            raise ValueError("invalid managed cost memory cursor")
        with self._lock:
            rows = sorted((v["doc"] for v in self.forward_cost_events.values()
                if v["doc"]["experiment_id"] == experiment_id and not v.get("ack")), key=_cost_key)
            if after:
                rows = [r for r in rows if _cost_key(r) > after]
            page = rows[:limit]
            token = _cost_token(experiment_id, _cost_key(page[-1])) if len(rows) > limit and page else None
            return copy.deepcopy(page), token

    def ack_forward_cost_event(self, experiment_id: str, event_id: str, observation_ref: Mapping[str, Any], recorded_at: str) -> None:
        _cost_scope(experiment_id)
        with self._lock:
            value = self.forward_cost_events.get(event_id)
            if not value or value["doc"]["experiment_id"] != experiment_id:
                raise ConditionFailed("managed cost event missing or outside experiment")
            ack = _cost_ack(value["doc"], observation_ref, recorded_at)
            if value.get("ack"):
                if value["ack"]["observation_ref"] != ack["observation_ref"]:
                    raise ConditionFailed("managed cost acknowledgement is immutable")
                return
            value["ack"] = ack

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy(v) for _, v in sorted(self.events.get(run_id, {}).items())]

    def query_runs(self, state: str | None = None, *, limit: int = 100, after: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
        with self._lock:
            rows = sorted((r for r in self.runs.values() if state is None or r["state"] == state), key=_sort_key)
            if after is not None:
                rows = [r for r in rows if _sort_key(r) > after]
            page = rows[:limit]
            nxt = _sort_key(page[-1]) if len(rows) > limit and page else None
            return [copy.deepcopy(r) for r in page], nxt

    def get_idempotency(self, scope_key: str) -> dict[str, Any] | None:
        with self._lock:
            r = self.idempotency.get(scope_key)
            return copy.deepcopy(r) if r is not None else None

    def put_idempotency(self, scope_key: str, record: Mapping[str, Any], ttl_epoch: int) -> None:
        with self._lock:
            if scope_key in self.idempotency:
                raise ConditionFailed("idempotency record exists")
            self.idempotency[scope_key] = copy.deepcopy({**dict(record), "scope_key": scope_key, "ttl": ttl_epoch})

    def lease_slots(self, lease_key: str) -> list[dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy({**v, "slot": k}) for k, v in sorted(self.leases.get(lease_key, {}).items())]

    def acquire_slot(self, lease_key: str, slot: int, holder: str, expires_at: str, acquired_at: str) -> bool:
        with self._lock:
            slots = self.leases.setdefault(lease_key, {})
            if slot in slots:
                return False
            slots[slot] = {"holder": holder, "expires_at": expires_at, "acquired_at": acquired_at, "heartbeat_at": acquired_at}
            return True

    def renew_slot(self, lease_key: str, slot: int, holder: str, expires_at: str) -> bool:
        with self._lock:
            cur = self.leases.get(lease_key, {}).get(slot)
            if cur is None or cur["holder"] != holder:
                return False
            cur["expires_at"] = expires_at
            cur["heartbeat_at"] = expires_at
            return True

    def release_slot(self, lease_key: str, slot: int, holder: str) -> bool:
        with self._lock:
            slots = self.leases.get(lease_key, {})
            cur = slots.get(slot)
            if cur is None or cur["holder"] != holder:
                return False
            del slots[slot]
            return True


# ===================================================================== DynamoDB
def _s(v: str) -> dict[str, str]:
    return {"S": v}


def _n(v: int | float) -> dict[str, str]:
    return {"N": str(v)}


class DynamoRunStore:
    """The single-table DynamoDB implementation (low-level client; values as JSON strings).

    ``client`` is a ``boto3.client("dynamodb")`` (moto in tests). Only ``GetItem``, ``PutItem``,
    ``UpdateItem``, ``DeleteItem``, ``Query`` and ``TransactWriteItems`` are used; no scans and no
    stream actions.
    """

    def __init__(self, client: Any, table_name: str) -> None:
        self.client = client
        self.table = table_name

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _is_condition_failure(exc: Exception) -> bool:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "") if hasattr(exc, "response") else ""
        return code in ("ConditionalCheckFailedException", "TransactionCanceledException")

    def _run_item(self, run: Mapping[str, Any]) -> dict[str, Any]:
        sk = _sort_key(run)
        return {
            "pk": _s(f"RUN#{run['run_id']}"),
            "sk": _s("RUN"),
            "doc": _s(json.dumps(run, sort_keys=True, separators=(",", ":"))),
            "revision": _n(int(run["revision"])),
            "state": _s(str(run["state"])),
            "gsi1pk": _s(f"STATE#{run['state']}"),
            "gsi1sk": _s(sk),
            "gsi2pk": _s("RUNS"),
            "gsi2sk": _s(sk),
        }

    def _event_put(self, run_id: str, ev: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "Put": {
                "TableName": self.table,
                "Item": {"pk": _s(f"RUN#{run_id}"), "sk": _s(f"EVT#{int(ev['seq']):06d}"), "doc": _s(json.dumps(dict(ev), sort_keys=True, separators=(",", ":")))},
                "ConditionExpression": "attribute_not_exists(pk)",
            }
        }

    def _transact(self, items: list[dict[str, Any]]) -> None:
        try:
            self.client.transact_write_items(TransactItems=items)
        except Exception as exc:  # noqa: BLE001 - mapped below
            if self._is_condition_failure(exc):
                raise ConditionFailed(str(exc)) from None
            raise

    # ------------------------------------------------------------------ runs
    def get_run(self, run_id: str) -> dict[str, Any] | None:
        resp = self.client.get_item(TableName=self.table, Key={"pk": _s(f"RUN#{run_id}"), "sk": _s("RUN")}, ConsistentRead=True)
        item = resp.get("Item")
        return json.loads(item["doc"]["S"]) if item else None

    def get_forward_context(self, context_id: str) -> dict[str, Any] | None:
        response = self.client.get_item(TableName=self.table,
            Key={"pk": _s("FORWARD_CONTEXT#" + context_id), "sk": _s("HEAD")}, ConsistentRead=True)
        item = response.get("Item")
        if not item:
            return None
        doc = json.loads(item["doc"]["S"])
        if doc.get("context_id") != context_id or doc.get("managed_run_id") != item["managed_run_id"]["S"]:
            raise ConditionFailed("forward context head binding is inconsistent")
        return doc

    def create_run(self, run: Mapping[str, Any], events: Sequence[Mapping[str, Any]], idempotency: Mapping[str, Any] | None = None, *, forward_context: Mapping[str, Any] | None = None) -> None:
        items: list[dict[str, Any]] = [{"Put": {"TableName": self.table, "Item": self._run_item(run), "ConditionExpression": "attribute_not_exists(pk)"}}]
        items += [self._event_put(run["run_id"], ev) for ev in events]
        if idempotency is not None:
            items.append({"Put": {"TableName": self.table, "Item": self._idem_item(idempotency["scope_key"], idempotency, int(idempotency["ttl"])), "ConditionExpression": "attribute_not_exists(pk)"}})
        if forward_context is not None:
            context = _forward_context_doc(run, forward_context)
            context_put = {"TableName": self.table,
                "Item": {"pk": _s("FORWARD_CONTEXT#" + context["context_id"]), "sk": _s("HEAD"),
                    "managed_run_id": _s(run["run_id"]), "doc": _s(json.dumps(context, sort_keys=True, separators=(",", ":")))},
                "ConditionExpression": "attribute_not_exists(pk)"}
            if context["previous_managed_run_id"] is not None:
                context_put.update(ConditionExpression="managed_run_id = :prior",
                    ExpressionAttributeValues={":prior": _s(context["previous_managed_run_id"])})
            items.append({"Put": context_put})
        self._transact(items)

    def update_run(self, run: Mapping[str, Any], expected_revision: int, events: Sequence[Mapping[str, Any]] = (), *, forward_cost_event: Mapping[str, Any] | None = None) -> None:
        put = {
            "Put": {
                "TableName": self.table,
                "Item": self._run_item(run),
                "ConditionExpression": "attribute_exists(pk) AND revision = :rev",
                "ExpressionAttributeValues": {":rev": _n(int(expected_revision))},
            }
        }
        items = [put, *[self._event_put(run["run_id"], ev) for ev in events]]
        if forward_cost_event is not None:
            doc, digest = _cost_event(run, forward_cost_event)
            old = self._get_forward_cost(doc["event_id"])
            if old:
                if old["checksum"]["S"] != digest:
                    raise ConditionFailed("managed cost event is immutable")
                # Preserve original bytes and any delivery acknowledgement; a
                # repeated terminal update never overwrites the outbox item.
                items.append({"ConditionCheck": {"TableName": self.table,
                    "Key": {"pk": old["pk"], "sk": old["sk"]},
                    "ConditionExpression": "#hash = :hash",
                    "ExpressionAttributeNames": {"#hash": "checksum"},
                    "ExpressionAttributeValues": {":hash": _s(digest)}}})
            else:
                items.append({"Put": {"TableName": self.table,
                    "Item": {"pk": _s("MANAGED_COST#" + doc["event_id"]), "sk": _s("EVENT"),
                        "doc": _s(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)),
                        "checksum": _s(digest), "gsi1pk": _s(_cost_scope(doc["experiment_id"])), "gsi1sk": _s(_cost_key(doc))},
                    "ConditionExpression": "attribute_not_exists(pk)"}})
        self._transact(items)

    def _get_forward_cost(self, event_id: str) -> dict[str, Any] | None:
        from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
        response = self.client.get_item(TableName=self.table,
            Key={"pk": _s("MANAGED_COST#" + event_id), "sk": _s("EVENT")}, ConsistentRead=True)
        item = response.get("Item")
        if item:
            doc = json.loads(item["doc"]["S"])
            if doc.get("event_id") != event_id or sha256_checksum(canonical_json_bytes(doc)) != item["checksum"]["S"]:
                raise ConditionFailed("managed cost outbox original bytes are inconsistent")
        return item

    def query_forward_cost_events(self, experiment_id: str, *, limit: int = 100, next_token: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
        cursor = _cost_cursor(experiment_id, next_token, limit)
        if cursor is not None and (not isinstance(cursor, dict) or cursor.get("gsi1pk") != _s(_cost_scope(experiment_id))):
            raise ValueError("invalid managed cost DynamoDB cursor")
        options = {"TableName": self.table, "IndexName": "by_state", "Limit": limit,
            "KeyConditionExpression": "gsi1pk = :scope", "ExpressionAttributeValues": {":scope": _s(_cost_scope(experiment_id))},
            "ScanIndexForward": True}
        if cursor is not None:
            options["ExclusiveStartKey"] = cursor
        response = self.client.query(**options)
        rows = []
        for indexed in response.get("Items", []):
            original = json.loads(indexed["doc"]["S"])
            # The GSI is eventually consistent: a delivered event can remain in
            # its old partition briefly. Strong-read its permanent head before
            # importing; newly lagging events remain pending for the next tick.
            current = self._get_forward_cost(original["event_id"])
            if not current or "ack" in current:
                continue
            doc = json.loads(current["doc"]["S"])
            if doc["experiment_id"] != experiment_id or current.get("gsi1pk") != _s(_cost_scope(experiment_id)):
                raise ConditionFailed("managed cost query scope mismatch")
            rows.append(doc)
        key = response.get("LastEvaluatedKey")
        return rows, _cost_token(experiment_id, key) if key else None

    def ack_forward_cost_event(self, experiment_id: str, event_id: str, observation_ref: Mapping[str, Any], recorded_at: str) -> None:
        _cost_scope(experiment_id)
        item = self._get_forward_cost(event_id)
        if not item:
            raise ConditionFailed("managed cost event missing")
        doc = json.loads(item["doc"]["S"])
        if doc["experiment_id"] != experiment_id:
            raise ConditionFailed("managed cost event outside experiment")
        ack = _cost_ack(doc, observation_ref, recorded_at)
        if "ack" in item:
            if json.loads(item["ack"]["S"])["observation_ref"] != ack["observation_ref"]:
                raise ConditionFailed("managed cost acknowledgement is immutable")
            return
        try:
            self.client.update_item(TableName=self.table, Key={"pk": item["pk"], "sk": item["sk"]},
                UpdateExpression="SET #ack = :ack REMOVE gsi1pk, gsi1sk",
                ConditionExpression="attribute_not_exists(#ack) AND #hash = :hash",
                ExpressionAttributeNames={"#ack": "ack", "#hash": "checksum"},
                ExpressionAttributeValues={":ack": _s(json.dumps(ack, sort_keys=True, separators=(",", ":"))), ":hash": item["checksum"]})
        except Exception as exc:
            if not self._is_condition_failure(exc):
                raise
            current = self._get_forward_cost(event_id)
            if current and json.loads(current.get("ack", _s("{}"))["S"]).get("observation_ref") == ack["observation_ref"]:
                return
            raise ConditionFailed("managed cost acknowledgement changed concurrently") from None

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {
            "TableName": self.table,
            "KeyConditionExpression": "pk = :pk AND begins_with(sk, :evt)",
            "ExpressionAttributeValues": {":pk": _s(f"RUN#{run_id}"), ":evt": _s("EVT#")},
            "ConsistentRead": True,
        }
        while True:
            resp = self.client.query(**kwargs)
            out += [json.loads(i["doc"]["S"]) for i in resp.get("Items", [])]
            if "LastEvaluatedKey" not in resp:
                return out
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    def query_runs(self, state: str | None = None, *, limit: int = 100, after: str | None = None) -> tuple[list[dict[str, Any]], str | None]:
        index, pk_attr, sk_attr, pk_val = ("by_state", "gsi1pk", "gsi1sk", f"STATE#{state}") if state else ("by_submitted", "gsi2pk", "gsi2sk", "RUNS")
        values: dict[str, Any] = {":pk": _s(pk_val)}
        cond = f"{pk_attr} = :pk"
        if after is not None:
            cond += f" AND {sk_attr} > :after"
            values[":after"] = _s(after)
        rows: list[dict[str, Any]] = []
        kwargs: dict[str, Any] = {"TableName": self.table, "IndexName": index, "KeyConditionExpression": cond, "ExpressionAttributeValues": values}
        while len(rows) <= limit:
            resp = self.client.query(**kwargs)
            rows += [json.loads(i["doc"]["S"]) for i in resp.get("Items", [])]
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        # GSIs are eventually consistent: drop rows whose state no longer matches.
        if state:
            rows = [r for r in rows if r["state"] == state]
        rows.sort(key=_sort_key)
        page = rows[:limit]
        nxt = _sort_key(page[-1]) if len(rows) > limit and page else None
        return page, nxt

    # ------------------------------------------------------------------ idempotency
    def _idem_item(self, scope_key: str, record: Mapping[str, Any], ttl_epoch: int) -> dict[str, Any]:
        return {"pk": _s(f"IDEM#{scope_key}"), "sk": _s("IDEM"), "doc": _s(json.dumps({**dict(record), "scope_key": scope_key, "ttl": ttl_epoch}, sort_keys=True, separators=(",", ":"))), "ttl": _n(int(ttl_epoch))}

    def get_idempotency(self, scope_key: str) -> dict[str, Any] | None:
        resp = self.client.get_item(TableName=self.table, Key={"pk": _s(f"IDEM#{scope_key}"), "sk": _s("IDEM")}, ConsistentRead=True)
        item = resp.get("Item")
        return json.loads(item["doc"]["S"]) if item else None

    def append_audit(self, record: Mapping[str, Any]) -> None:
        """Append-only audit item ``AUDIT#production-strategy`` / ``<at>#<audit_id>`` (never overwritten)."""
        item = {"pk": _s("AUDIT#production-strategy"), "sk": _s(f"{record['at']}#{record['audit_id']}"), "doc": _s(json.dumps(dict(record), sort_keys=True, separators=(",", ":")))}
        self.client.put_item(TableName=self.table, Item=item, ConditionExpression="attribute_not_exists(pk)")

    def put_idempotency(self, scope_key: str, record: Mapping[str, Any], ttl_epoch: int) -> None:
        try:
            self.client.put_item(TableName=self.table, Item=self._idem_item(scope_key, record, ttl_epoch), ConditionExpression="attribute_not_exists(pk)")
        except Exception as exc:  # noqa: BLE001
            if self._is_condition_failure(exc):
                raise ConditionFailed("idempotency record exists") from None
            raise

    # ------------------------------------------------------------------ leases
    def lease_slots(self, lease_key: str) -> list[dict[str, Any]]:
        resp = self.client.query(
            TableName=self.table,
            KeyConditionExpression="pk = :pk AND begins_with(sk, :slot)",
            ExpressionAttributeValues={":pk": _s(f"LEASE#{lease_key}"), ":slot": _s("SLOT#")},
            ConsistentRead=True,
        )
        out = []
        for i in resp.get("Items", []):
            out.append({"slot": int(i["sk"]["S"].split("#", 1)[1]), "holder": i["holder"]["S"], "expires_at": i["expires_at"]["S"], "acquired_at": i["acquired_at"]["S"], "heartbeat_at": i["heartbeat_at"]["S"]})
        return sorted(out, key=lambda r: r["slot"])

    def _conditional(self, fn: Any, **kwargs: Any) -> bool:
        try:
            fn(TableName=self.table, **kwargs)
            return True
        except Exception as exc:  # noqa: BLE001
            if self._is_condition_failure(exc):
                return False
            raise

    def acquire_slot(self, lease_key: str, slot: int, holder: str, expires_at: str, acquired_at: str) -> bool:
        item = {"pk": _s(f"LEASE#{lease_key}"), "sk": _s(f"SLOT#{slot}"), "holder": _s(holder), "expires_at": _s(expires_at), "acquired_at": _s(acquired_at), "heartbeat_at": _s(acquired_at)}
        return self._conditional(self.client.put_item, Item=item, ConditionExpression="attribute_not_exists(pk)")

    def renew_slot(self, lease_key: str, slot: int, holder: str, expires_at: str) -> bool:
        return self._conditional(
            self.client.update_item,
            Key={"pk": _s(f"LEASE#{lease_key}"), "sk": _s(f"SLOT#{slot}")},
            UpdateExpression="SET expires_at = :e, heartbeat_at = :e",
            ConditionExpression="holder = :h",
            ExpressionAttributeValues={":e": _s(expires_at), ":h": _s(holder)},
        )

    def release_slot(self, lease_key: str, slot: int, holder: str) -> bool:
        return self._conditional(
            self.client.delete_item,
            Key={"pk": _s(f"LEASE#{lease_key}"), "sk": _s(f"SLOT#{slot}")},
            ConditionExpression="holder = :h",
            ExpressionAttributeValues={":h": _s(holder)},
        )
