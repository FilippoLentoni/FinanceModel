"""Contract error codes and the error envelope (contracts ``core/v1/error.json``).

Every FinanceModel failure that crosses a boundary (job API response, job result, log line) is a
:class:`FinplanError` carrying a **registered** contract code. The registry and each code's default
``retryable`` value come from the pinned contract package (``core/v1/error-codes.json``), never
from a copy in this repository.

Rules enforced here:

* the code must be registered; codes whose ``retryable`` is fixed by the contract cannot be
  overridden (``BUDGET_EXCEEDED`` is never retryable, for example);
* ``VALIDATION_FAILED`` always carries ``details.pointer`` (a JSON pointer, ``""`` = whole document),
  ``INVALID_IDENTIFIER`` carries ``details.field`` and ``UNSUPPORTED_CONTRACT_VERSION`` carries
  ``details.served_contract_majors`` (contract conditionals);
* messages and details never echo request values, storage locations or tracebacks:
  :meth:`FinplanError.to_envelope` validates the envelope with the contract validator (which runs
  the ``no_leaks`` check) and degrades to a generic ``INTERNAL`` envelope rather than emit a
  non-conformant one.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from functools import lru_cache
from typing import Any

from finplan_contracts.schemas import load_store

__all__ = [
    "ErrorCode",
    "FinplanError",
    "as_finplan_error",
    "contract_version",
    "registered_codes",
]


class ErrorCode(StrEnum):
    VALIDATION_FAILED = "VALIDATION_FAILED"
    INVALID_IDENTIFIER = "INVALID_IDENTIFIER"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    IDEMPOTENCY_KEY_REUSED = "IDEMPOTENCY_KEY_REUSED"
    IMMUTABLE_RECORD = "IMMUTABLE_RECORD"
    PRECONDITION_FAILED = "PRECONDITION_FAILED"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    OPERATION_NOT_PERMITTED = "OPERATION_NOT_PERMITTED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    RATE_LIMITED = "RATE_LIMITED"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    UNSUPPORTED_CONTRACT_VERSION = "UNSUPPORTED_CONTRACT_VERSION"
    INTERNAL = "INTERNAL"


@lru_cache(maxsize=1)
def registered_codes() -> dict[str, dict[str, Any]]:
    """The contract's registered codes: ``{code: {"retryable": bool, "retryable_fixed": bool, ...}}``."""
    return dict(load_store().get("error-codes").schema["x-finplan-error-codes"])


@lru_cache(maxsize=1)
def contract_version() -> str:
    """Version of the pinned contract package (stamped into every envelope and record)."""
    return load_store().version


_FALLBACK_CORRELATION_ID = "corr-unavailable"


class FinplanError(Exception):
    """A failure with a registered contract error code.

    ``details`` must hold only safe, structured values (identifiers, names, numbers, reasons).
    ``pointer`` / ``field`` are shortcuts for the contract-required detail keys.
    """

    def __init__(
        self,
        code: ErrorCode | str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        retryable: bool | None = None,
        pointer: str | None = None,
        field: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        code = str(code)
        reg = registered_codes().get(code)
        if reg is None:
            raise ValueError(f"unregistered contract error code {code!r}")
        if retryable is None:
            retryable = bool(reg["retryable"])
        elif reg.get("retryable_fixed") and retryable != bool(reg["retryable"]):
            raise ValueError(f"{code} has a fixed retryable={reg['retryable']}")
        d = dict(details or {})
        if pointer is not None:
            d["pointer"] = pointer
        if field is not None:
            d["field"] = field
        if code == ErrorCode.VALIDATION_FAILED:
            d.setdefault("pointer", "")
        if code == ErrorCode.INVALID_IDENTIFIER and "field" not in d:
            raise ValueError("INVALID_IDENTIFIER requires details.field")
        if code == ErrorCode.UNSUPPORTED_CONTRACT_VERSION and not d.get("served_contract_majors"):
            raise ValueError("UNSUPPORTED_CONTRACT_VERSION requires details.served_contract_majors")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = d
        self.retryable = retryable
        self.correlation_id = correlation_id

    # ------------------------------------------------------------------ factories
    @classmethod
    def validation(cls, message: str, *, pointer: str = "", **details: Any) -> "FinplanError":
        return cls(ErrorCode.VALIDATION_FAILED, message, pointer=pointer, details=details)

    @classmethod
    def not_permitted(cls, message: str, **details: Any) -> "FinplanError":
        return cls(ErrorCode.OPERATION_NOT_PERMITTED, message, details=details)

    @classmethod
    def precondition(cls, message: str, *, reason: str, **details: Any) -> "FinplanError":
        return cls(ErrorCode.PRECONDITION_FAILED, message, details={"reason": reason, **details})

    @classmethod
    def internal(cls, message: str, **details: Any) -> "FinplanError":
        return cls(ErrorCode.INTERNAL, message, details=details)

    @classmethod
    def dependency_unavailable(cls, message: str, *, retryable: bool | None = None, **details: Any) -> "FinplanError":
        return cls(ErrorCode.DEPENDENCY_UNAVAILABLE, message, details=details, retryable=retryable)

    # ------------------------------------------------------------------ envelope
    def to_envelope(self, correlation_id: str | None = None, *, version: str | None = None, synthetic: bool | None = None, validate: bool = True) -> dict[str, Any]:
        """The contract error envelope; validated against ``core/v1/error.json`` when ``validate``."""
        env: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
            "correlation_id": correlation_id or self.correlation_id or _FALLBACK_CORRELATION_ID,
            "contract_version": version or contract_version(),
        }
        if synthetic is not None:
            env["synthetic"] = synthetic
        if validate:
            from finplan_contracts.validate import validate as contract_validate

            result = contract_validate(env, "error")
            if not result.valid:
                # Never emit a non-conformant (possibly leaking) envelope.
                safe = {
                    "code": ErrorCode.INTERNAL.value,
                    "message": "error envelope failed contract validation",
                    "retryable": False,
                    "details": {"original_code": self.code, "issues": [i.message for i in result.issues][:5]},
                    "correlation_id": env["correlation_id"],
                    "contract_version": env["contract_version"],
                }
                if synthetic is not None:
                    safe["synthetic"] = synthetic
                return safe
        return env


def as_finplan_error(exc: BaseException) -> FinplanError:
    """Map any exception to a :class:`FinplanError`; unexpected ones become ``INTERNAL`` with no traceback."""
    if isinstance(exc, FinplanError):
        return exc
    return FinplanError.internal("unexpected failure", exception_type=type(exc).__name__)
