"""Run context: everything a job, the evaluator or a control-plane operation needs from outside.

A :class:`RunContext` is built once per job (container entry point) or per API request and passed
down explicitly. It carries the environment, the run identity and purpose, the injected clock and
ID minter, the pinned contract version, the evaluator version and the image digest, plus a
structured logger whose every line carries the ``correlation_id`` (spec job-execution-controls,
"Failure semantics": the correlation ID appears in the job logs).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any

from .clock import Clock, SystemClock, utc_iso
from .errors import FinplanError, as_finplan_error, contract_version
from .ids import IdMinter

__all__ = ["ENVIRONMENTS", "PURPOSES", "REPO", "RunContext", "EVALUATOR_VERSION"]

REPO = "financemodel"
#: Deployed environments, plus ``local`` for offline container and unit-test runs.
ENVIRONMENTS = ("beta", "gamma", "prod", "local")
#: Run purposes (spec experiment-job-interface, "Run purpose").
PURPOSES = ("research", "tuning", "holdout_evaluation", "production_candidate")
#: Semver of the common evaluator and simulator (recorded in every result; bump on any change that
#: can alter trades or metrics). The image digest is recorded next to it.
EVALUATOR_VERSION = "1.0.0"


@dataclass(frozen=True)
class RunContext:
    environment: str = "local"
    purpose: str = "research"
    run_id: str | None = None
    correlation_id: str = ""
    clock: Clock = field(default_factory=SystemClock)
    ids: IdMinter = field(default_factory=IdMinter)
    contract_version: str = field(default_factory=contract_version)
    evaluator_version: str = EVALUATOR_VERSION
    image_digest: str | None = None
    synthetic: bool = True
    model_version: str | None = None
    logger_name: str = "finplan_model"

    def __post_init__(self) -> None:
        if self.environment not in ENVIRONMENTS:
            raise ValueError(f"unknown environment {self.environment!r}")
        if self.purpose not in PURPOSES:
            raise ValueError(f"unknown purpose {self.purpose!r}")
        if not self.correlation_id:
            object.__setattr__(self, "correlation_id", self.ids.correlation_id())

    # ------------------------------------------------------------------ helpers
    @classmethod
    def for_tests(cls, *, seed: int = 0, start: str = "2026-01-05T00:00:00Z", **kwargs: Any) -> "RunContext":
        from .clock import FrozenClock

        clock = kwargs.pop("clock", None) or FrozenClock(start)
        ids = kwargs.pop("ids", None) or IdMinter.seeded(clock, seed)
        return cls(clock=clock, ids=ids, **kwargs)

    def with_run(self, run_id: str, **changes: Any) -> "RunContext":
        return replace(self, run_id=run_id, **changes)

    def now_ts(self) -> str:
        return utc_iso(self.clock.now())

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(self.logger_name)

    def log(self, event: str, level: int = logging.INFO, **fields: Any) -> dict[str, Any]:
        """One structured JSON log line with ``correlation_id``, ``run_id`` and ``environment``."""
        record = {"event": event, "correlation_id": self.correlation_id, "environment": self.environment, **({"run_id": self.run_id} if self.run_id else {}), **fields}
        self.logger.log(level, json.dumps(record, sort_keys=True, default=str))
        return record

    def error_envelope(self, exc: BaseException) -> dict[str, Any]:
        """The contract error envelope for ``exc`` stamped with this context's correlation ID."""
        err: FinplanError = as_finplan_error(exc)
        return err.to_envelope(self.correlation_id, version=self.contract_version, synthetic=True if self.synthetic else None)
