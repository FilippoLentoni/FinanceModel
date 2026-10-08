"""Container entry point: ``python -m finplan_model.jobs <job_type>`` (task 6.1).

Deployed mode (SageMaker Processing): the control plane starts the job with
``ContainerArguments = [<job_type>]`` and the environment ``FINPLAN_ENVIRONMENT``,
``FINPLAN_RUN_ID``, ``FINPLAN_JOB_TYPE``, ``FINPLAN_CORRELATION_ID``, ``FINPLAN_RUN_SPEC_SHA256`` and
``FINPLAN_IMAGE_DIGEST``. The job reads its run spec from research storage (bucket from
``/finplan/<env>/financemodel/config/research-storage-ref``), refuses a spec whose SHA-256 differs,
resolves the approved snapshot through the platform API (endpoint from
``/finplan/<env>/financialplanning/api/plan-endpoint``, SigV4 with the job role), verifies every
checksum before strategy code runs, runs the job type and writes the job result **last**.

Local mode (no AWS at all; tests and ``docker run`` on fixtures)::

    python -m finplan_model.jobs run_backtest --local <dir> --run-id <run_id> \\
        --fixture-snapshot <synthetic snapshot payload.json>

reads ``<dir>/runs/<run_id>/spec.json``, serves the synthetic payload through the in-process mock
platform (as an approved snapshot with the spec's ``input_snapshot_id``) and writes
``<dir>/runs/<run_id>/job-result.json`` plus artifacts under ``<dir>/artifacts/``.

Exit codes: ``0`` the job succeeded (any ``solution_status``, including ``infeasible``); ``1`` the job
failed and wrote a failed result (contract error envelope with the run's ``correlation_id``);
``2`` usage error. A non-zero exit makes SageMaker report the job ``Failed``; the control plane then
reads the failed result (or reports ``INTERNAL`` when none was written).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from finplan_model.core.artifacts import ArtifactStore, LocalArtifactStore
from finplan_model.core.clock import Clock, SystemClock
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.core.ids import IdMinter, require_id
from finplan_model.core.outcome import failed_job_result
from finplan_model.core.platform import FixturePlatformClient, PlatformClient, build_synthetic_snapshot

from .handlers import HANDLERS, JobInputs
from .market_loader import load_market
from .runio import LocalRunIO, RunIO
from .spec import validate_run_spec

__all__ = ["main", "run_job"]

log = logging.getLogger("finplan_model.jobs")


class _Stopped(BaseException):
    """SIGTERM from SageMaker (stop request or stopping condition)."""


def _on_sigterm(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
    raise _Stopped()


def run_job(
    job_type: str,
    run_id: str,
    *,
    run_io: RunIO,
    platform: PlatformClient,
    artifacts: ArtifactStore,
    environment: str = "local",
    spec_checksum: str | None = None,
    correlation_id: str = "",
    image_digest: str | None = None,
    clock: Clock | None = None,
    ids: IdMinter | None = None,
    staging_store: Any = None,
    registry: Any = None,
) -> tuple[int, dict[str, Any]]:
    """Run one job end to end; always writes a job result for a well-formed ``run_id``.

    ``staging_store`` (a :class:`finplan_model.staging.StagingStore`) and ``registry`` (a
    :class:`finplan_model.registry.ModelRegistry`) enable the production-candidate hand-off (task
    9.1): a ``production_candidate`` run's bundle is staged, manifest last, **before** the result is
    written, and the result carries the ``staging`` block. A bundle that does not validate fails the
    run (``VALIDATION_FAILED``) with nothing staged.
    """
    clock = clock or SystemClock()
    ids = ids or IdMinter(clock)
    ctx = RunContext(environment=environment, run_id=run_id, correlation_id=correlation_id, clock=clock, ids=ids, image_digest=image_digest, logger_name="finplan_model.jobs")
    try:
        require_id("run_id", run_id)
        spec = validate_run_spec(run_io.get_spec(run_id, spec_checksum))
        if spec["run_id"] != run_id or spec["job_type"] != job_type:
            raise FinplanError.validation("the run spec does not match the job arguments", pointer="/job_type")
        ctx = replace(ctx, purpose=spec["purpose"], correlation_id=spec["correlation_id"], synthetic=bool(spec.get("synthetic", True)), model_version=spec.get("model_version"), image_digest=image_digest or spec.get("image_digest"))
        handler = HANDLERS.get(job_type)
        if handler is None:
            raise FinplanError.validation("unknown job type", pointer="/job_type")
        ctx.log("job_started", job_type=job_type)
        # Integrity first: approved snapshot only, every checksum verified before strategy code runs.
        market, content = load_market(platform, spec["input_snapshot_id"])
        from .universe import bias_disclosures_of, cash_assumption, is_universe_content

        if is_universe_content(content):
            # the snapshot's declared cash assumption replaces the configured cash rate
            spec = {**spec, "simulation": {**dict(spec["simulation"]), "cash_rate_annual": cash_assumption(content)["annual_rate"]}}
        ctx = replace(ctx, synthetic=bool(market.synthetic or spec.get("synthetic")))
        doc = handler(JobInputs(ctx=ctx, spec=spec, market=market, snapshot=content, artifacts=artifacts))
        if spec["purpose"] == "production_candidate":
            from finplan_model.staging import StagingTarget, stage_run_output

            if staging_store is None:
                doc["staging"] = {"status": "not_staged", "reason": "staging_unavailable"}
            else:
                doc["staging"] = stage_run_output(staging_store, doc, purpose=spec["purpose"], target=StagingTarget.from_spec(spec), clock=clock, registry=registry, spec=spec, actor=f"finplan-{environment}-financemodel-job-execution-role", bias_disclosures=bias_disclosures_of(content) or None)
                ctx.log("run_output_staged" if doc["staging"]["status"] == "staged" else "run_output_not_staged", reason=doc["staging"].get("reason"))
        run_io.put_result(run_id, doc)  # written last: the completion record of the job
        ctx.log("job_succeeded", solution_status=doc.get("solution_status"), artifacts=len(doc.get("artifacts", [])))
        return 0, doc
    except (Exception, _Stopped) as exc:
        err: BaseException = FinplanError.internal("the job was stopped before it finished", reason="job_stopped") if isinstance(exc, _Stopped) else exc
        if not isinstance(err, FinplanError):
            log.error(json.dumps({"event": "job_exception", "correlation_id": ctx.correlation_id, "exception_type": type(err).__name__}))
        try:
            require_id("run_id", run_id)
        except FinplanError:
            return 1, ctx.error_envelope(err)
        doc = failed_job_result(ctx, err, run_id=run_id)
        try:
            run_io.put_result(run_id, doc)
        except Exception:  # noqa: BLE001 - the control plane reports INTERNAL without a result
            ctx.log("result_write_failed", level=logging.ERROR)
        return 1, doc


def _deployed(job_type: str) -> int:  # pragma: no cover - needs AWS; covered by integration-beta
    import boto3

    from finplan_model.core.artifacts import S3ArtifactStore
    from finplan_model.core.config import load_config
    from finplan_model.core.platform import HttpPlatformClient

    from .runio import S3RunIO

    env = os.environ["FINPLAN_ENVIRONMENT"]
    run_id = os.environ["FINPLAN_RUN_ID"]
    config_dir = os.environ.get("FINPLAN_CONFIG_DIR")
    cfg = load_config(env, Path(config_dir) if config_dir else None)
    session = boto3.session.Session(region_name=cfg.region)
    ssm = session.client("ssm")
    bucket_raw = ssm.get_parameter(Name=cfg.ssm["research_storage_ref"])["Parameter"]["Value"].strip()
    bucket = json.loads(bucket_raw)["bucket"] if bucket_raw.startswith("{") else bucket_raw
    endpoint = ssm.get_parameter(Name=cfg.ssm["plan_endpoint"])["Parameter"]["Value"]
    from finplan_model.core.aws_clients import s3_client

    s3 = s3_client(cfg.region, session=session)
    staging_store = registry = None
    try:
        from finplan_model.registry import ModelRegistry, S3RegistryStore
        from finplan_model.staging import S3StagingStore

        staging_ref = ssm.get_parameter(Name=cfg.ssm["run_staging_ref"])["Parameter"]["Value"]
        staging_store = S3StagingStore(s3, staging_ref)
        registry_bucket = ssm.get_parameter(Name=cfg.ssm_name("config", "registry-storage-ref"))["Parameter"]["Value"]
        registry = ModelRegistry(S3RegistryStore(s3, registry_bucket))
    except Exception:  # noqa: BLE001 - a production candidate then reports staging_unavailable
        log.warning(json.dumps({"event": "staging_or_registry_reference_unavailable"}))
    code, _ = run_job(
        job_type,
        run_id,
        run_io=S3RunIO(s3, bucket),
        platform=HttpPlatformClient(endpoint, region=cfg.region, credentials=session.get_credentials()),
        artifacts=S3ArtifactStore(s3, bucket),
        environment=env,
        spec_checksum=os.environ.get("FINPLAN_RUN_SPEC_SHA256"),
        correlation_id=os.environ.get("FINPLAN_CORRELATION_ID", ""),
        image_digest=os.environ.get("FINPLAN_IMAGE_DIGEST"),
        staging_store=staging_store,
        registry=registry,
    )
    return code


def _local(job_type: str, args: argparse.Namespace) -> int:
    root = Path(args.local)
    run_io = LocalRunIO(root)
    platform = FixturePlatformClient()
    if args.fixture_snapshot:
        # Only the snapshot id is taken here; run_job re-reads and verifies the spec.
        try:
            snapshot_id = str(run_io.get_spec(args.run_id, None)["input_snapshot_id"])
        except (FinplanError, KeyError):
            snapshot_id = None
        payload = json.loads(Path(args.fixture_snapshot).read_text(encoding="utf-8"))
        if payload.get("synthetic") is not True:
            print("local runs accept synthetic fixtures only (synthetic: true)", file=sys.stderr)
            return 2
        if snapshot_id:
            record, blobs = build_synthetic_snapshot(payload, input_snapshot_id=snapshot_id)
            platform.add_snapshot(record, blobs)
    code, doc = run_job(job_type, args.run_id, run_io=run_io, platform=platform, artifacts=LocalArtifactStore(root / "artifacts"), environment="local", spec_checksum=args.spec_checksum, image_digest=args.image_digest)
    print(json.dumps({"run_id": args.run_id, "completion_status": doc.get("completion_status"), "solution_status": doc.get("solution_status")}, sort_keys=True))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m finplan_model.jobs", description="FinanceModel CPU job entry point (prepare_dataset, run_backtest, run_benchmark, report, daily_recommendation, model_selection).")
    ap.add_argument("job_type", choices=sorted(HANDLERS))
    ap.add_argument("--local", metavar="DIR", help="offline mode: run spec, result and artifacts under DIR (no AWS)")
    ap.add_argument("--run-id", help="local mode: the run_id whose spec to run")
    ap.add_argument("--fixture-snapshot", metavar="FILE", help="local mode: synthetic snapshot payload served as an approved snapshot")
    ap.add_argument("--spec-checksum", default=None, help="local mode: expected sha256 of the run spec")
    ap.add_argument("--image-digest", default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
    if args.local:
        if not args.run_id:
            ap.error("--local needs --run-id")
        return _local(args.job_type, args)
    if os.environ.get("FINPLAN_JOB_TYPE") not in (None, args.job_type):
        print("job type argument does not match FINPLAN_JOB_TYPE", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, _on_sigterm)
    return _deployed(args.job_type)
