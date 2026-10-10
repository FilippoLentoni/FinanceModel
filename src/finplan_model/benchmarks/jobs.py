"""Real benchmark handlers; all portfolio results use the common simulator/evaluator."""

from __future__ import annotations

import json
import os
import time

from finplan_model.core.errors import FinplanError
from finplan_model.evaluate import assert_comparable, evaluate
from finplan_model.jobs.comparison import benchmark_section
from finplan_model.jobs.results import succeeded_result
from finplan_model.jobs.strategy_resolver import CONTROLS, resolve_strategy

from .jev import JevClient, JevStrategy, SECRET, calibration_report
from .qwen import QwenSwarm, VllmBackend


def _finish(inp, strategy, *, provider_evidence, started):
    from finplan_model.jobs.handlers import _bias, _with_bias
    start, end = inp.window
    results = [evaluate(strategy, inp.market, inp.sim_config, ctx=inp.ctx, start=start, end=end, universe=inp.universe)]
    for name in CONTROLS:
        results.append(evaluate(resolve_strategy(name, constraints=inp.sim_config.constraints), inp.market, inp.sim_config, ctx=inp.ctx, start=start, end=end, universe=inp.universe))
    assert_comparable(results)
    evidence = provider_evidence()
    evidence.update({"protocol": "finplan-llm-benchmark/1", "evaluation": results[0].to_dict(), "controls": [r.to_dict() for r in results[1:]],
        "historical_pretraining_leakage_risk": True, "model_pretraining_cutoff": "not_verified", "prospective_paper_validation_required": True,
        "selection": "no strategy activation; request fixes horizon and parameters before evaluation", "bias_section": _bias(inp)})
    ref = inp.artifacts.put_json(evidence, kind="run_artifact", synthetic=True if inp.ctx.synthetic else None, domain="finance")
    result = succeeded_result(inp.ctx, inp.spec, solution_status=results[0].solution_status, artifacts=[ref], performance=results[0].metrics,
        dataset_checksum=inp.market.dataset_checksum, instance_seconds=time.monotonic() - started,
        benchmark=benchmark_section(results, inp.market, primary=strategy.name), accuracy=evidence.get("calibration", {}))
    # Compact evidence stays in the API; all messages/decisions live in the sealed artifact.
    result["payload"]["llm_benchmark"] = {k: v for k, v in evidence.items() if k not in ("evaluation", "controls", "messages", "decisions", "responses")}
    return _with_bias(result, evidence["bias_section"])


def jev_backtest(inp, *, client=None):
    if inp.spec["purpose"] not in ("research", "tuning", "holdout_evaluation"):
        raise FinplanError.not_permitted("Jev benchmark accepts research purposes only", reason="outbound_purpose_not_permitted")
    started = time.monotonic()
    if client is None:
        import boto3
        session = boto3.session.Session(region_name=os.environ.get("AWS_REGION", "us-east-2"))
        secret = session.client("secretsmanager").get_secret_value(SecretId=SECRET)["SecretString"]
        try:
            raw = json.loads(secret)
            secret = raw.get("api_key", raw.get("key"))
        except ValueError:
            pass
        client = JevClient(secret)
    raw_values = [v for i in inp.market.instruments for b in inp.market.bars_of(i) for v in (b.open, b.high, b.low, b.close, b.volume) if v is not None]
    strategy = JevStrategy(client, ctx=inp.ctx, raw_values=raw_values)
    def evidence():
        return {"provider": "typesafe_systemone", "usage": client.usage, "responses": strategy.records,
            "calibration": calibration_report(strategy.records, inp.market, observed_end=inp.window[1]), "decision_horizon_sessions": 21,
            "sizing": {"threshold": strategy.threshold, "max_delta": strategy.max_delta, "confidence_minimum": .5}, "raw_series_egress": False}
    return _finish(inp, strategy, provider_evidence=evidence, started=started)


def swarm_mode_a(inp, *, backend=None, weights_root="/opt/ml/input/data/weights"):
    started = time.monotonic()
    engine = backend or VllmBackend(weights_root)
    try:
        strategy = QwenSwarm(engine, run_id=inp.ctx.run_id)
        return _finish(inp, strategy, provider_evidence=lambda: strategy.evidence, started=started)
    finally:
        close = getattr(engine, "close", None)
        if close:
            close()


def rl_weight_staging(inp):
    import boto3
    from finplan_model.core.aws_clients import s3_client
    from .weights import stage_weights
    session = boto3.session.Session(region_name=os.environ.get("AWS_REGION", "us-east-2"))
    bucket = session.client("ssm").get_parameter(Name=f"/finplan/{inp.ctx.environment}/financemodel/config/research-storage-ref")["Parameter"]["Value"]
    record = stage_weights(s3_client(session.region_name, session=session), bucket)
    ref = inp.artifacts.put_json(record, kind="run_artifact", synthetic=True if inp.ctx.synthetic else None, domain="finance")
    result = succeeded_result(inp.ctx, inp.spec, solution_status="not_applicable", artifacts=[ref], performance={}, dataset_checksum=inp.market.dataset_checksum)
    result["payload"]["weight_staging"] = record
    return result
