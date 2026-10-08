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
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
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


@runtime_checkable
class RunStore(Protocol):
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...

    def create_run(self, run: Mapping[str, Any], events: Sequence[Mapping[str, Any]], idempotency: Mapping[str, Any] | None = None) -> None:
        """Create a run (revision 0) with its first events and, atomically, its idempotency record.

        Raises :class:`ConditionFailed` when the run or the idempotency record already exists."""
        ...

    def update_run(self, run: Mapping[str, Any], expected_revision: int, events: Sequence[Mapping[str, Any]] = ()) -> None:
        """Replace the run if its stored revision equals ``expected_revision`` and append ``events``."""
        ...

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

    def create_run(self, run: Mapping[str, Any], events: Sequence[Mapping[str, Any]], idempotency: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            if run["run_id"] in self.runs:
                raise ConditionFailed("run exists")
            if idempotency is not None and idempotency["scope_key"] in self.idempotency:
                raise ConditionFailed("idempotency record exists")
            self.runs[run["run_id"]] = copy.deepcopy(dict(run))
            self._append(run["run_id"], events)
            if idempotency is not None:
                self.idempotency[idempotency["scope_key"]] = copy.deepcopy(dict(idempotency))

    def update_run(self, run: Mapping[str, Any], expected_revision: int, events: Sequence[Mapping[str, Any]] = ()) -> None:
        with self._lock:
            cur = self.runs.get(run["run_id"])
            if cur is None or int(cur["revision"]) != int(expected_revision):
                raise ConditionFailed("revision changed")
            bucket = self.events.get(run["run_id"], {})
            if any(int(ev["seq"]) in bucket for ev in events):
                raise ConditionFailed("event exists")
            self.runs[run["run_id"]] = copy.deepcopy(dict(run))
            self._append(run["run_id"], events)

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

    def create_run(self, run: Mapping[str, Any], events: Sequence[Mapping[str, Any]], idempotency: Mapping[str, Any] | None = None) -> None:
        items: list[dict[str, Any]] = [{"Put": {"TableName": self.table, "Item": self._run_item(run), "ConditionExpression": "attribute_not_exists(pk)"}}]
        items += [self._event_put(run["run_id"], ev) for ev in events]
        if idempotency is not None:
            items.append({"Put": {"TableName": self.table, "Item": self._idem_item(idempotency["scope_key"], idempotency, int(idempotency["ttl"])), "ConditionExpression": "attribute_not_exists(pk)"}})
        self._transact(items)

    def update_run(self, run: Mapping[str, Any], expected_revision: int, events: Sequence[Mapping[str, Any]] = ()) -> None:
        put = {
            "Put": {
                "TableName": self.table,
                "Item": self._run_item(run),
                "ConditionExpression": "attribute_exists(pk) AND revision = :rev",
                "ExpressionAttributeValues": {":rev": _n(int(expected_revision))},
            }
        }
        self._transact([put, *[self._event_put(run["run_id"], ev) for ev in events]])

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
