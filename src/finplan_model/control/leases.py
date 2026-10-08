"""Per-environment concurrency lease (spec job-execution-controls, "Concurrency lease per
environment"; design D3; CTL-02).

A lease is ``max_holders`` conditional-write slots per ``(environment, instance_class)``; each slot
has a holder ``run_id``, an expiry and a heartbeat. A run must hold a slot before its SageMaker job
starts; without one it stays ``queued``. Heartbeats (state-change handler and dispatcher) push the
expiry forward while the job is alive. An expired slot is reclaimed **only after** the holder's
SageMaker job is confirmed terminal (or never existed and the run is no longer active), and every
reclaim is logged. Because SageMaker quotas are account-wide and shared by beta, gamma and prod, the
lease cannot guarantee capacity: a quota refusal at start re-queues the run (CTL-04).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from finplan_model.core.clock import Clock, parse_utc, utc_iso

from .store import RunStore

__all__ = ["LeaseManager"]


class LeaseManager:
    def __init__(self, store: RunStore, environment: str, *, clock: Clock, ttl_seconds: int, limits: Callable[[], dict[str, int]]) -> None:
        self.store = store
        self.environment = environment
        self.clock = clock
        self.ttl = timedelta(seconds=int(ttl_seconds))
        self._limits = limits

    def key(self, instance_class: str) -> str:
        return f"{self.environment}#{instance_class}"

    def limit(self, instance_class: str) -> int:
        return int(self._limits().get(instance_class, 0))

    def _expiry(self) -> str:
        return utc_iso(self.clock.now() + self.ttl)

    def holders(self, instance_class: str) -> list[dict[str, Any]]:
        return self.store.lease_slots(self.key(instance_class))

    def held_slot(self, instance_class: str, run_id: str) -> int | None:
        return next((s["slot"] for s in self.holders(instance_class) if s["holder"] == run_id), None)

    def acquire(self, instance_class: str, run_id: str) -> int | None:
        """A free slot below the configured limit, or ``None`` (the run stays queued)."""
        existing = self.held_slot(instance_class, run_id)
        if existing is not None:
            return existing
        taken = {s["slot"] for s in self.holders(instance_class)}
        now = utc_iso(self.clock.now())
        for slot in range(self.limit(instance_class)):
            if slot in taken:
                continue
            if self.store.acquire_slot(self.key(instance_class), slot, run_id, self._expiry(), now):
                return slot
        return None

    def renew(self, instance_class: str, run_id: str, slot: int | None = None) -> bool:
        slot = self.held_slot(instance_class, run_id) if slot is None else slot
        if slot is None:
            return False
        return self.store.renew_slot(self.key(instance_class), slot, run_id, self._expiry())

    def release(self, instance_class: str, run_id: str) -> bool:
        slot = self.held_slot(instance_class, run_id)
        if slot is None:
            return False
        return self.store.release_slot(self.key(instance_class), slot, run_id)

    def expired(self, instance_class: str, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or self.clock.now()
        return [s for s in self.holders(instance_class) if parse_utc(s["expires_at"]) < now]

    def reclaim(self, instance_class: str, slot: dict[str, Any]) -> bool:
        """Release an expired slot. The caller must have confirmed the holder's job is terminal."""
        return self.store.release_slot(self.key(instance_class), int(slot["slot"]), str(slot["holder"]))
