"""Daily bounded research windows, checked before any model or provider inference."""

from datetime import date

from finplan_model.core.errors import FinplanError

PILOT_DECISIONS = 21
MAX_DECISIONS = 32
MIN_HISTORY_PRICES = 21


def daily_scope(market, universe, start, end, *, frequency):
    if frequency != "daily":
        raise FinplanError.validation("all benchmark families must decide daily", pointer="/configuration/payload/rebalance_frequency")
    if start is None or end is None:
        raise FinplanError.validation("daily LLM benchmarks require an explicit bounded window", pointer="/evaluation_window")
    sessions = [s for s in market.sessions if start <= s <= end]
    if len(sessions) < 2 or sessions[0] != start or sessions[-1] != end:
        raise FinplanError.validation("benchmark bounds must name at least two actual market sessions", pointer="/evaluation_window")
    decisions = len(sessions) - 1  # The final session fills the preceding decision.
    if decisions > MAX_DECISIONS:
        raise FinplanError.validation("daily LLM benchmark exceeds the pre-inference decision cap", pointer="/evaluation_window", max_decisions=MAX_DECISIONS, requested_decisions=decisions)
    instruments = tuple(universe or market.instruments)
    if not instruments or set(instruments) - set(market.instruments):
        raise FinplanError.validation("benchmark universe is not covered", pointer="/configuration/payload/universe")
    for session in sessions:
        for instrument in instruments:
            bar = market.bar(instrument, session)
            if bar is None or bar.available_at > market.decision_time(session):
                raise FinplanError.validation("benchmark requires aligned completed bars available at each decision", pointer="/evaluation_window", instrument=instrument, session=session.isoformat())
    history, _ = market.view(start, instruments).price_matrix("close")
    if len(history) < MIN_HISTORY_PRICES:
        raise FinplanError.validation("benchmark lacks point-in-time feature warmup", pointer="/evaluation_window", required_history_prices=MIN_HISTORY_PRICES)
    return {
        "kind": "limited_daily_pilot", "rebalance_frequency": "daily",
        "evaluation_window": {"start": start.isoformat(), "end": end.isoformat()},
        "market_sessions": len(sessions), "decision_count": decisions,
        "max_decisions": MAX_DECISIONS, "history_prices_at_first_decision": len(history),
        "history_warmup_retained": True, "full_2026_benchmark": False,
        "untouched_holdout_comparison": False,
        "comparison_scope": "same-window daily controls only; not the full PPO/optimizer holdout",
        "prospective_promotion_evidence": False,
    }


def pilot_window(market, universe, *, as_of: date):
    sessions = [s for s in market.sessions if s <= as_of]
    if len(sessions) < PILOT_DECISIONS + 1:
        raise FinplanError.validation("snapshot lacks 22 sessions for the daily pilot", pointer="/input_snapshot_id")
    start, end = sessions[-(PILOT_DECISIONS + 1)], sessions[-1]
    scope = daily_scope(market, universe, start, end, frequency="daily")
    return scope["evaluation_window"], scope
