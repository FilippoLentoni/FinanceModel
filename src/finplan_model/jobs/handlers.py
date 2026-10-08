"""Job-type handlers of the ``financemodel-cpu`` image (design D5; task 6.1).

==================  ===========================================================================
entry point         what it does (all through the common evaluator; nothing else produces results)
==================  ===========================================================================
``prepare_dataset`` resolve and verify the snapshot, build the point-in-time market data and store
                    a dataset descriptor (delegates to ``finplan_model.datasets.prepare_dataset_job``
                    when task group 3 provides it); ``solution_status`` ``not_applicable``
``run_backtest``    one strategy over the evaluation window
``run_benchmark``   the configured strategy plus the three controls on the same dataset and
                    simulation configuration, sequentially in one job (one processing slot); the
                    comparison guard refuses mixed settings
``run_daily_recommendation``  the production strategy frozen on the run (M2) on the approved
                    research-universe snapshot; proposes weights for every instrument plus cash and
                    carries the snapshot's bias disclosures (staged by the entry point)
``report``          the benchmark report (delegates to ``finplan_model.reporting.report_job`` when
                    task group 5 provides it; otherwise ``DEPENDENCY_UNAVAILABLE``)
``model_selection`` decision 27 offline model selection (controls, traditional optimizers, PPO and
                    SAC): tuning and RL training on train/validation, one test evaluation
                    (:mod:`finplan_model.selection.job`; runs as a SageMaker Training job; the RL
                    learners are imported only when this entry point runs)
==================  ===========================================================================

Every handler receives a :class:`JobInputs` (run context, run spec, verified market data, artifact
store) and returns a contract job-result document. Integrity failures stop the job **before**
strategy code runs (the snapshot is verified in :func:`finplan_model.jobs.market_loader.load_market`).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from finplan_model.core.artifacts import ArtifactStore
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.core.platform import SnapshotContent
from finplan_model.evaluate import assert_comparable, evaluate
from finplan_model.sim.config import SimulationConfig
from finplan_model.sim.market import MarketData

from .comparison import benchmark_section
from .results import succeeded_result
from .strategy_resolver import CONTROLS, resolve_strategy

__all__ = ["HANDLERS", "JobInputs", "register_handler"]


@dataclass
class JobInputs:
    ctx: RunContext
    spec: Mapping[str, Any]
    market: MarketData
    snapshot: SnapshotContent
    artifacts: ArtifactStore

    @property
    def sim_config(self) -> SimulationConfig:
        return SimulationConfig.from_dict(self.spec["simulation"])

    @property
    def window(self) -> tuple[date | None, date | None]:
        w = self.spec.get("evaluation_window") or {}
        return (date.fromisoformat(w["start"]) if w.get("start") else None, date.fromisoformat(w["end"]) if w.get("end") else None)

    @property
    def universe(self) -> list[str] | None:
        u = [i for i in self.spec.get("universe") or [] if i in self.market.instruments]
        missing = sorted(set(self.spec.get("universe") or []) - set(self.market.instruments))
        if missing:
            raise FinplanError.precondition("the snapshot does not cover every instrument of the configuration", reason="universe_not_covered", instruments=missing[:10])
        return u or None


Handler = Callable[[JobInputs], dict[str, Any]]


def _hook(module: str, name: str) -> Callable[..., Any] | None:
    try:
        fn = getattr(importlib.import_module(module), name, None)
    except ImportError:  # pragma: no cover
        return None
    return fn if callable(fn) else None


def _evaluate(inp: JobInputs, strategy_name: str) -> Any:
    params = (inp.spec.get("strategy_params") or {}) if strategy_name == inp.spec.get("strategy") else {}
    strategy = resolve_strategy(strategy_name, params, constraints=inp.sim_config.constraints)
    start, end = inp.window
    return evaluate(strategy, inp.market, inp.sim_config, ctx=inp.ctx, start=start, end=end, universe=inp.universe)


def _proposal(inp: JobInputs, res: Any) -> dict[str, Any] | None:
    """Production candidates propose an allocation (staged by task 9.1); other runs do not."""
    if inp.spec.get("purpose") != "production_candidate":
        return None
    from finplan_model.staging import proposed_allocation

    return proposed_allocation(res)


def _bias(inp: JobInputs) -> dict[str, Any] | None:
    """The mandatory bias section of a universe run (fails closed without disclosures), else None."""
    from .universe import bias_section, is_universe_content

    return bias_section(inp.snapshot) if is_universe_content(inp.snapshot) else None


def _with_bias(doc: dict[str, Any], section: Mapping[str, Any] | None) -> dict[str, Any]:
    if section is not None:
        doc["payload"]["bias_section"] = dict(section)
    return doc


def run_backtest(inp: JobInputs) -> dict[str, Any]:
    section = _bias(inp)
    name = str(inp.spec["strategy"])
    res = _evaluate(inp, name)
    body = res.to_dict()
    if section is not None:
        body = {**body, "bias_section": section}
    ref = inp.artifacts.put_json(body, kind="run_artifact", synthetic=True if inp.ctx.synthetic else None, domain="finance")
    doc = succeeded_result(inp.ctx, inp.spec, solution_status=res.solution_status, artifacts=[ref], performance=res.metrics, dataset_checksum=inp.market.dataset_checksum, proposed_allocation=_proposal(inp, res), benchmark=benchmark_section([res], inp.market, primary=name))
    return _with_bias(doc, section)


def _last_targets(res: Any) -> tuple[dict[str, float], float] | None:
    for rec in reversed(list(getattr(res.simulation, "decisions", []) or [])):
        cons = rec.get("constraints") or {}
        if rec.get("action") in ("executed", "projected") and cons.get("final_weights") is not None:
            return {k: float(v) for k, v in cons["final_weights"].items()}, float(cons.get("final_cash") or 0.0)
    return None


def run_daily_recommendation(inp: JobInputs) -> dict[str, Any]:
    """The production strategy frozen at submission, on the universe snapshot (daily-recommendation-job)."""
    from .universe import full_allocation, universe_block

    frozen = inp.spec.get("production_strategy") or {}
    name = str(inp.spec["strategy"])
    if not frozen.get("strategy_id") or frozen["strategy_id"] != name:
        raise FinplanError.validation("the run does not carry the production strategy frozen at submission", pointer="/production_strategy", field="strategy_id")
    if inp.spec.get("purpose") != "production_candidate":
        raise FinplanError.validation("daily_recommendation runs are production candidates", pointer="/purpose")
    block = universe_block(inp.snapshot.payload)
    if block is None:
        raise FinplanError.validation("daily_recommendation needs an equity-etf-daily research-universe snapshot", pointer="/input_snapshot_id")
    section = _bias(inp)
    res = _evaluate(inp, name)
    instruments = [str(i["instrument_id"]) for i in block.get("instruments", []) if i.get("kind") != "cash"]
    targets = _last_targets(res)
    allocation = full_allocation(targets[0], targets[1], instruments) if targets else None
    body = {"strategy": name, "production_strategy": dict(frozen), "evaluation": res.to_dict(), "proposed_allocation": allocation, "bias_section": section}
    ref = inp.artifacts.put_json(body, kind="run_artifact", synthetic=True if inp.ctx.synthetic else None, domain="finance")
    # Controls (cash, buy_and_hold, equal_weight) solve nothing (``not_applicable``); their rule-based
    # allocation, projected through the constraint set, is a feasible recommendation to stage.
    solution = "feasible" if res.solution_status == "not_applicable" and allocation else res.solution_status
    doc = succeeded_result(inp.ctx, inp.spec, solution_status=solution, artifacts=[ref], performance=res.metrics, dataset_checksum=inp.market.dataset_checksum, proposed_allocation=allocation)
    return _with_bias(doc, section)


def run_benchmark(inp: JobInputs) -> dict[str, Any]:
    section = _bias(inp)
    main = str(inp.spec["strategy"])
    names = [main, *[c for c in CONTROLS if c != main]]
    results = [_evaluate(inp, n) for n in names]
    assert_comparable(results)
    doc = {"strategies": names, "results": [r.to_dict() for r in results]}
    if section is not None:
        doc["bias_section"] = section
    ref = inp.artifacts.put_json(doc, kind="run_artifact", synthetic=True if inp.ctx.synthetic else None, domain="finance")
    reporter = _hook("finplan_model.reporting", "benchmark_report")
    refs = [ref]
    if reporter is not None:
        report = reporter(results, ctx=inp.ctx)
        report_doc = dict(report) if isinstance(report, Mapping) else report.to_dict()
        if section is not None:
            report_doc["bias_section"] = section  # mandatory "Hindsight and survivorship bias" section
        refs.append(inp.artifacts.put_json(report_doc, kind="run_artifact", synthetic=True if inp.ctx.synthetic else None, domain="finance"))
    out = succeeded_result(inp.ctx, inp.spec, solution_status=results[0].solution_status, artifacts=refs, performance=results[0].metrics, dataset_checksum=inp.market.dataset_checksum, proposed_allocation=_proposal(inp, results[0]), benchmark=benchmark_section(results, inp.market, primary=main))
    return _with_bias(out, section)


def prepare_dataset(inp: JobInputs) -> dict[str, Any]:
    hook = _hook("finplan_model.datasets", "prepare_dataset_job")
    if hook is not None:
        return hook(inp)
    descriptor = {
        "dataset_id": inp.market.dataset_id,
        "dataset_checksum": inp.market.dataset_checksum,
        "input_snapshot_id": inp.spec["input_snapshot_id"],
        "configuration_id": inp.spec["configuration_id"],
        "sessions": [inp.market.sessions[0].isoformat(), inp.market.sessions[-1].isoformat()],
        "n_sessions": len(inp.market.sessions),
        "instruments": list(inp.market.instruments),
        "lineage": inp.snapshot.snapshot.lineage,
        "synthetic": bool(inp.market.synthetic),
    }
    ref = inp.artifacts.put_json(descriptor, kind="research_dataset", synthetic=True if inp.ctx.synthetic else None, domain="finance")
    return succeeded_result(inp.ctx, inp.spec, solution_status="not_applicable", artifacts=[ref], performance={}, dataset_checksum=inp.market.dataset_checksum)


def report(inp: JobInputs) -> dict[str, Any]:
    hook = _hook("finplan_model.reporting", "report_job")
    if hook is None:
        raise FinplanError.dependency_unavailable("the report job is not available in this image", retryable=False, job_type="report")
    return hook(inp)


def model_selection(inp: JobInputs) -> dict[str, Any]:
    from finplan_model.selection.job import run_model_selection

    return run_model_selection(inp)


HANDLERS: dict[str, Handler] = {
    "prepare_dataset": prepare_dataset,
    "run_backtest": run_backtest,
    "run_benchmark": run_benchmark,
    "daily_recommendation": run_daily_recommendation,
    "report": report,
    "model_selection": model_selection,
}


def register_handler(job_type: str, handler: Handler) -> None:
    """Later task groups or changes register additional job types here."""
    HANDLERS[job_type] = handler
