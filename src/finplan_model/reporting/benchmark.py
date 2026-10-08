"""Benchmark runs: controls plus selected strategies over one dataset (BASE-01, REP-02, REP-05, REP-07).

:func:`run_benchmark` evaluates every requested strategy - with the three controls added
automatically when the request omits them - through the common evaluator, on the same dataset and
the same simulation configuration, for each requested period:

* ``walk_forward`` - every walk-forward fold's test window (period type ``walk_forward_fold``);
* ``validation`` - the validation range (``validation``);
* ``holdout`` - only through :class:`~finplan_model.datasets.holdout.HoldoutAccessor` (purpose
  ``holdout_evaluation``, frozen candidate, logged); each holdout evaluation stores a holdout
  metrics record (``holdout``);
* a prospective dataset is evaluated over its prospective window (``prospective_paper``).

Strategies with randomness (scenario-CVaR) run once per configured seed. Every result is a stored
``EvaluationResult.to_dict()`` document, so reports are regenerated from stored results only.
All results of one run are checked with the comparison guard (SIM-01).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from finplan_model.core.artifacts import ArtifactStore
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.datasets.dataset import Dataset
from finplan_model.datasets.holdout import FrozenCandidate, HoldoutAccessor
from finplan_model.evaluate import EvaluatorIdentity, assert_comparable, evaluate
from finplan_model.sim.config import SimulationConfig
from finplan_model.strategies.registry import BASELINES, build_strategy, descriptor, with_controls

from .holdout_record import build_holdout_record, store_holdout_record
from .storage import RUN_RESULT_KIND, store_research_artifact

__all__ = ["BenchmarkRequest", "BenchmarkRun", "ResultEntry", "StrategySpec", "run_benchmark"]

PERIODS = ("walk_forward", "validation", "holdout")
_LABEL_RE = r"^[a-z][a-z0-9_]{0,63}$"


@dataclass(frozen=True)
class StrategySpec:
    name: str
    params: Mapping[str, Any]
    label: str
    seeds: tuple[int, ...] | None = None
    auto_added: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "label": self.label, "params": dict(self.params), "seeds": None if self.seeds is None else list(self.seeds), "auto_added": self.auto_added}


@dataclass(frozen=True)
class BenchmarkRequest:
    strategies: tuple[StrategySpec, ...]
    periods: tuple[str, ...]
    auto_added_controls: tuple[str, ...]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BenchmarkRequest":
        import re

        for k in d:
            if k not in ("strategies", "periods", "$comment"):
                raise FinplanError.validation(f"unknown benchmark setting {k}", pointer=f"/{k}")
        raw = d.get("strategies") or []
        if not isinstance(raw, list):
            raise FinplanError.validation("strategies must be a list", pointer="/strategies")
        specs_raw, added = with_controls([s if isinstance(s, Mapping) else {"name": s} for s in raw])
        specs: list[StrategySpec] = []
        for n, s in enumerate(specs_raw):
            for k in s:
                if k not in ("name", "params", "label", "seeds", "auto_added"):
                    raise FinplanError.validation(f"unknown strategy setting {k}", pointer=f"/strategies/{n}/{k}")
            name = str(s.get("name"))
            if name not in BASELINES:
                raise FinplanError.validation("unknown strategy", pointer=f"/strategies/{n}/name", known=sorted(BASELINES))
            label = str(s.get("label") or name)
            if not re.match(_LABEL_RE, label):
                raise FinplanError.validation("strategy label must be snake case", pointer=f"/strategies/{n}/label")
            params = dict(s.get("params") or {})
            seeds = s.get("seeds")
            if seeds is not None:
                if not BASELINES[name].has_randomness:
                    raise FinplanError.validation("seeds apply only to strategies with randomness", pointer=f"/strategies/{n}/seeds")
                if not isinstance(seeds, list) or not seeds or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
                    raise FinplanError.validation("seeds must be a list of distinct non-negative integers", pointer=f"/strategies/{n}/seeds")
                seeds = tuple(seeds)
            elif BASELINES[name].has_randomness:
                seeds = (int(params.get("seed", 0)),)
            specs.append(StrategySpec(name, params, label, seeds, bool(s.get("auto_added"))))
        labels = [s.label for s in specs]
        if len(set(labels)) != len(labels):
            raise FinplanError.validation("strategy labels must be unique (give repeated strategies a label)", pointer="/strategies")
        periods = d.get("periods", ["walk_forward", "validation"])
        if not isinstance(periods, list) or not periods or any(p not in PERIODS for p in periods):
            raise FinplanError.validation(f"periods must be a non-empty subset of {', '.join(PERIODS)}", pointer="/periods")
        return cls(tuple(specs), tuple(p for p in PERIODS if p in periods), tuple(added))

    def to_dict(self) -> dict[str, Any]:
        return {"strategies": [s.to_dict() for s in self.strategies], "periods": list(self.periods), "auto_added_controls": list(self.auto_added_controls)}


@dataclass(frozen=True)
class ResultEntry:
    strategy: str  # label
    strategy_name: str
    seed: int | None
    period_type: str  # walk_forward_fold | validation | holdout | prospective_paper
    period_id: str
    result: dict[str, Any]
    result_ref: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "strategy_name": self.strategy_name, "seed": self.seed, "period_type": self.period_type, "period_id": self.period_id, "result": self.result, "result_ref": self.result_ref}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ResultEntry":
        return cls(d["strategy"], d["strategy_name"], d.get("seed"), d["period_type"], d["period_id"], dict(d["result"]), d.get("result_ref"))


@dataclass
class BenchmarkRun:
    request: BenchmarkRequest
    dataset: dict[str, Any]
    simulation_configuration: dict[str, Any]
    identity: dict[str, Any]
    descriptors: dict[str, dict[str, Any]]
    entries: list[ResultEntry]
    folds: list[dict[str, Any]]
    holdout_bounds: dict[str, str] | None = None
    holdout_records: list[dict[str, Any]] = field(default_factory=list)
    holdout_record_refs: list[dict[str, Any]] = field(default_factory=list)
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Everything a report is generated from (stored with the run's artifacts)."""
        return {
            "request": self.request.to_dict(),
            "dataset": self.dataset,
            "simulation_configuration": self.simulation_configuration,
            "identity": self.identity,
            "descriptors": self.descriptors,
            "entries": [e.to_dict() for e in self.entries],
            "folds": self.folds,
            "holdout_bounds": self.holdout_bounds,
            "holdout_records": self.holdout_records,
            "holdout_record_refs": self.holdout_record_refs,
            "run_id": self.run_id,
        }


def run_benchmark(
    request: BenchmarkRequest | Mapping[str, Any],
    dataset: Dataset,
    simulation: SimulationConfig | Mapping[str, Any],
    *,
    ctx: RunContext,
    holdout_accessor: HoldoutAccessor | None = None,
    candidate: FrozenCandidate | None = None,
    model_versions: Mapping[str, str] | None = None,
    store: ArtifactStore | None = None,
) -> BenchmarkRun:
    req = request if isinstance(request, BenchmarkRequest) else BenchmarkRequest.from_dict(request)
    sim = simulation if isinstance(simulation, SimulationConfig) else SimulationConfig.from_dict(simulation)
    mv = dict(model_versions or {})
    entries: list[ResultEntry] = []
    windows: list[tuple[str, str, date, date]] = []
    prospective = dataset.period == "prospective_paper"
    if prospective:
        w = dataset.prospective_window
        assert w is not None
        windows.append(("prospective_paper", "prospective", w.start, w.end))
    else:
        if "walk_forward" in req.periods:
            if not dataset.folds:
                raise FinplanError.validation("the dataset has no walk-forward folds", pointer="/periods")
            windows += [("walk_forward_fold", f.fold_id, f.test.start, f.test.end) for f in dataset.folds]
        if "validation" in req.periods:
            v = dataset.validation
            assert v is not None
            windows.append(("validation", "validation", v.start, v.end))
    market = dataset.market_data()
    strategies = []
    for spec in req.strategies:
        base = build_strategy(spec.name, spec.params, constraints=sim.constraints)
        for seed in spec.seeds or (None,):
            strategies.append((spec, seed, base.with_seed(seed) if seed is not None else base))  # type: ignore[attr-defined]

    def record(spec: Any, seed: Any, ptype: str, pid: str, res: Any) -> ResultEntry:
        doc = res.to_dict()
        ref = store_research_artifact(doc, store, kind=RUN_RESULT_KIND, synthetic=dataset.synthetic).to_dict() if store is not None else None
        return ResultEntry(spec.label, spec.name, seed, ptype, pid, doc, ref)

    evaluated = []
    for spec, seed, strat in strategies:
        sctx = ctx.with_run(ctx.run_id, model_version=mv.get(spec.label)) if ctx.run_id else ctx
        for ptype, pid, start, end in windows:
            res = evaluate(strat, market, sim, ctx=sctx, start=start, end=end)
            evaluated.append(res)
            entries.append(record(spec, seed, ptype, pid, res))

    holdout_bounds = None
    records: list[dict[str, Any]] = []
    record_refs: list[dict[str, Any]] = []
    if "holdout" in req.periods and not prospective:
        if holdout_accessor is None:
            raise FinplanError.precondition("a holdout evaluation needs the holdout accessor", reason="holdout_accessor_required")
        data = holdout_accessor.read(ctx, candidate)  # purpose, frozen candidate and access log
        holdout_bounds = {"start": data.start, "end": data.end}
        for spec, seed, strat in strategies:
            sctx = ctx.with_run(ctx.run_id, model_version=mv.get(spec.label)) if ctx.run_id else ctx
            res = evaluate(strat, data.market, sim, ctx=sctx, start=date.fromisoformat(data.start), end=date.fromisoformat(data.end))
            evaluated.append(res)
            entry = record(spec, seed, "holdout", "holdout", res)
            entries.append(entry)
            rec = build_holdout_record(entry.result, dataset=dataset.describe(), holdout=holdout_bounds, simulation_configuration=sim.to_dict(), run_id=ctx.run_id, model_version=mv.get(spec.label), seed=seed)
            rec["strategy_label"] = spec.label
            records.append(rec)
            if store is not None:
                record_refs.append(store_holdout_record(rec, store).to_dict())
    assert_comparable(evaluated)
    ident = EvaluatorIdentity.of(sim, market, evaluator_version=ctx.evaluator_version, image_digest=ctx.image_digest).to_dict()
    run = BenchmarkRun(
        request=req,
        dataset=dataset.describe(),
        simulation_configuration=sim.to_dict(),
        identity=ident,
        descriptors={s.label: {**descriptor(s.name), "label": s.label, "auto_added_control": s.auto_added, "configuration_id": build_strategy(s.name, s.params, constraints=sim.constraints).configuration_id} for s in req.strategies},
        entries=entries,
        folds=[f.to_dict() for f in dataset.folds] if not prospective else [],
        holdout_bounds=holdout_bounds,
        holdout_records=records,
        holdout_record_refs=record_refs,
        run_id=ctx.run_id,
    )
    ctx.log("benchmark_completed", strategies=len(req.strategies), entries=len(entries), auto_added_controls=list(req.auto_added_controls), synthetic=dataset.synthetic)
    return run

