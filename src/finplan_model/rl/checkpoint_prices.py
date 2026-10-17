"""Freeze aligned prices through the approved, checksum-verifying snapshot reader.

A historical price snapshot supplies many transitions, rather than one PPO sample.
It is a retrospective reconstruction; its retrieval date is never presented as a
historical first-seen vintage. News availability is deliberately unaffected.
"""
from __future__ import annotations

from datetime import UTC, datetime

from finplan_model.core.errors import FinplanError
from finplan_model.core.platform import SnapshotReader
from finplan_model.jobs.market_loader import market_from_content
from finplan_model.rl.arrays import PriceArrays
from finplan_model.sim.config import SimulationConfig


def freeze_prices(client, snapshot_id, instruments, simulation, *, news_start, checkpoint_at, now=None):
    """Return the checkpoint price bundle from verified approved real source bytes.

    Keep the exact source session calendar, adjusted price basis, simulation
    configuration, identity and revision disclosures. Missing bars remain
    missing and cannot be traded. No provider calls or invented news rows.
    """
    now = now or datetime.now(UTC)
    content = SnapshotReader(client).load(snapshot_id)
    if content.snapshot.synthetic or content.payload.get("synthetic"):
        raise FinplanError.precondition("checkpoint prices require a real approved snapshot", reason="real_price_provenance_required")
    names = tuple(sorted(instruments))
    if not names or len(names) != len(set(names)):
        raise FinplanError.validation("price universe must be nonempty and unique", pointer="/instruments")
    market = market_from_content(content)
    if set(names) - set(market.instruments):
        raise FinplanError.precondition("approved snapshot does not cover the requested universe", reason="universe_not_covered")
    # Manifest calendars can extend beyond completed source observations. Never
    # freeze future/empty sessions as if they had actually been observed.
    last = min(max(b.session_date for b in market.bars_of(i)) for i in names)
    sessions = [d for d in market.sessions if d <= last and market.decision_time(d) <= now]
    first = max(min(b.session_date for b in market.bars_of(i)) for i in names)
    sessions = [d for d in sessions if d >= first]
    if len(sessions) < 2:
        raise FinplanError.precondition("aligned completed price history is missing", reason="insufficient_history")
    from finplan_model.sim.market import MarketData
    kept = set(sessions)
    market = MarketData(sessions, [b for i in names for b in market.bars_of(i) if b.session_date in kept],
                        decision_times={d: market.decision_time(d) for d in sessions}, synthetic=False,
                        dataset_id=market.dataset_id, dataset_checksum=market.dataset_checksum)
    config = SimulationConfig.from_dict(simulation)
    arrays = PriceArrays.from_market(market, names, execution_timing=config.execution_timing)
    record = content.snapshot.record
    block = content.payload.get("universe") or {}
    return {
        "instruments": list(arrays.instruments), "decision_times": [t.isoformat() for t in arrays.decision_times],
        "close": arrays.close.tolist(), "exec_price": arrays.exec_price.tolist(),
        "volume": arrays.volume.tolist(), "has_bar": arrays.has_bar.tolist(),
        "input_snapshot_id": snapshot_id, "snapshot_checksum": content.snapshot.manifest_checksum,
        "snapshot_status": record["status"], "synthetic": False, "simulation": config.to_dict(),
        "news_start": news_start, "checkpoint_at": checkpoint_at,
        "price_provenance": {
            "verified_artifacts": dict(content.verified), "lineage": content.snapshot.lineage,
            "source_created_at": record.get("created_at"), "calendar": content.manifest.get("calendar"),
            "return_basis": block.get("return_basis", "close"), "quality_flags": record.get("quality_flags", []),
            "quality_details": record.get("quality_details", {}), "bias_disclosures": record.get("bias_disclosures", []),
            "availability_mode": "retrospective_session_close_assumption",
            "vintage_note": "Historical price reconstruction; source retrieval and revisions are not backdated. Forward news retains actual availability.",
        },
    }
