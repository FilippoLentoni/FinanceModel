"""Identifier minting (FinanceModel mints ``run_id`` and ``model_version``; contracts identifiers).

* ``run_id`` = ``run_`` + ULID, ``model_version`` = ``mv_`` + ULID (Crockford base32, uppercase).
* ``configuration_id`` = ``cfg_`` + SHA-256 of the RFC 8785 canonical configuration, computed by
  the contract package (:func:`configuration_id`), never re-implemented here.
* Artifact IDs are FinanceModel-issued opaque strings (``core/v1/artifact-ref.json``).

:class:`IdMinter` takes an injected clock and randomness source, so tests mint the same IDs on
every run (:meth:`IdMinter.seeded`).
"""

from __future__ import annotations

import os
import random
import re
from collections.abc import Callable
from typing import Any

from finplan_contracts.canonical import configuration_id as _contract_configuration_id
from finplan_contracts.canonical import request_hash as _contract_request_hash
from ulid import ULID

from .clock import Clock, SystemClock
from .errors import ErrorCode, FinplanError

__all__ = [
    "IdMinter",
    "RUN_ID_RE",
    "MODEL_VERSION_RE",
    "SNAPSHOT_ID_RE",
    "configuration_id",
    "request_hash",
    "require_id",
]

_ULID = r"[0-9A-HJKMNP-TV-Z]{26}"
RUN_ID_RE = re.compile(rf"^run_{_ULID}\Z")
MODEL_VERSION_RE = re.compile(rf"^mv_{_ULID}\Z")
SNAPSHOT_ID_RE = re.compile(rf"^snap_{_ULID}\Z")
CONFIGURATION_ID_RE = re.compile(r"^cfg_[0-9a-f]{64}\Z")
_PATTERNS = {"run_id": RUN_ID_RE, "model_version": MODEL_VERSION_RE, "input_snapshot_id": SNAPSHOT_ID_RE, "configuration_id": CONFIGURATION_ID_RE}


class IdMinter:
    def __init__(self, clock: Clock | None = None, randomness: Callable[[int], bytes] = os.urandom) -> None:
        self.clock = clock or SystemClock()
        self._rand = randomness

    @classmethod
    def seeded(cls, clock: Clock, seed: int = 0) -> "IdMinter":
        rng = random.Random(seed)
        return cls(clock, rng.randbytes)

    def ulid(self) -> str:
        ms = int(self.clock.now().timestamp() * 1000)
        return str(ULID.from_bytes(ms.to_bytes(6, "big") + self._rand(10)))

    def run_id(self) -> str:
        return f"run_{self.ulid()}"

    def model_version(self) -> str:
        return f"mv_{self.ulid()}"

    def artifact_id(self, prefix: str = "art") -> str:
        return f"{prefix}_{self.ulid()}"

    def correlation_id(self) -> str:
        return f"corr_{self.ulid()}"


def configuration_id(document: Any) -> str:
    """``cfg_`` + SHA-256 of the canonical (RFC 8785) form, via the contract package."""
    return _contract_configuration_id(document)


def request_hash(body: Any) -> str:
    """Canonical request hash for idempotency records (contract definition)."""
    return _contract_request_hash(body)


def require_id(field: str, value: Any) -> str:
    """Validate an identifier's prefix and format; raises ``INVALID_IDENTIFIER`` naming the field."""
    pattern = _PATTERNS.get(field)
    if pattern is None:
        raise ValueError(f"no identifier pattern for field {field!r}")
    if not isinstance(value, str) or not pattern.match(value):
        raise FinplanError(ErrorCode.INVALID_IDENTIFIER, f"{field} has the wrong prefix or format", field=field)
    return value
