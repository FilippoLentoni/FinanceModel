"""Forward modeling competitions run by the Lambda forward runner (beta only).

``beta-competition-v1``: ten synthetic USD 100 000 paper books on the fixed eight-asset
cohort (AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA, VOO; the frozen PPO export's universe),
two decisions per weekday at 09:45 and 15:45 America/New_York, each filled at the quote
FinancialPlanning collected for the same run (``same_run_regular_session_quote``, quote at
most 900 s old at the fill). Schedules live in FinancialPlanning's Step Functions state
machine; FinanceModel has no EventBridge schedule for it.

Missed slots: nothing is backfilled. A slot the state machine never invokes leaves no
record; a slot processed more than 3600 s after its scheduled time records every book as
``skipped`` (no model invoked, no fill), exactly as the hourly experiments do. Per-slot
idempotency (permanent run IDs) and 600 s invocation leases are unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .hourly import POLICY as HOURLY_POLICY
from .schedule import TWICE_DAILY


@dataclass(frozen=True)
class Competition:
    experiment_id: str
    approach_ids: tuple[str, ...]
    cohort: str
    comparison_policy: dict[str, Any] = field(default_factory=dict)


BETA_COMPETITION_V1 = Competition(
    experiment_id="beta-competition-v1",
    approach_ids=("cash", "buy_and_hold", "equal_weight", "classical_min_variance", "classical_cvar", "ppo",
                  "agent_min_variance", "agent_cvar", "agent_ppo", "direct_claude_news"),
    cohort="eight_assets",
    comparison_policy={**copy.deepcopy(HOURLY_POLICY), "version": "beta-competition/1",
                       "decision_opportunity": TWICE_DAILY,
                       "execution_timing": "same_run_regular_session_quote",
                       # The runner may take several minutes after the quote collection.
                       "execution_price_freshness_seconds": 900,
                       "research_schedule": "weekday 09:00 and 15:00 America/New_York (direct_claude_news prompts)"},
)

COMPETITIONS: dict[str, Competition] = {c.experiment_id: c for c in (BETA_COMPETITION_V1,)}


def competition(experiment: str) -> Competition:
    try:
        return COMPETITIONS[experiment]
    except KeyError:
        raise ValueError(f"unknown experiment {experiment!r}; known: {', '.join(sorted(COMPETITIONS))}") from None
