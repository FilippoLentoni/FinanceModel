#!/usr/bin/env python3
"""Release identity, artifact digest, release manifest and published references (tasks 10.4, 10.5;
DEP-03, DEP-04; contracts D4, D6).

Build stage (once per commit):

* :func:`mint_release_id` - ``rel_`` + ULID, shared by every environment of the build;
* :func:`assembly_digest` - ``sha256:`` over the sorted (path, SHA-256) list of every assembly file
  **plus** the ``financemodel-cpu`` image digest, so beta, gamma and prod manifests of one release
  record the same ``artifact_digest`` (DEP-03);
* :class:`ReleaseInfo` (``release-info.json`` in BuildOutput): release ID, source commit, artifact
  digest, pinned contract version and wheel digest, served majors, the image repository and digest;
* :func:`store_build_output` / :func:`fetch_build_output` - the release ledger in the pipeline store
  (``releases/<release_id>/``), used by rollback (no rebuild; digest re-verified).

Each deploy (``PublishRelease`` action, :func:`publish_release`), as the ``pipeline`` writer bound to
the environment (:func:`finplan_contracts.ssm.check_write` on every write):

1. reads the stack outputs of ``finplan-<env>-financemodel-{storage,control}``;
2. seeds the model registry with the baseline strategies for this release's image digest (design D9);
3. publishes the references that exist (:func:`planned_parameters`): ``config/research-storage-ref``,
   ``config/registry-storage-ref``, ``job/job-api-role-ref``, ``job/<job-type>`` per deployed job
   type (the image **by digest**, never a tag), ``job/job-role-ref``, ``api/job-endpoint``,
   ``model/registry-ref`` and ``config/approver-role-ref``;
4. publishes ``config/budget-enforced-role-names`` (job-submission, dispatcher, state-handler and
   job-execution roles and this environment's pipeline roles) for the FinancialPlanning budget
   action (contracts D4); the account-level pipeline and build roles are in the ``shared`` list the
   bootstrap writes (D16). FinanceModel creates no budget;
5. validates and writes ``release/manifest`` (contract ``release-manifest``; ``outputs`` lists every
   published reference, so it names the job endpoint, the registry reference and the deployed job
   types, plus the pinned contract version; prod adds ``approved_by``/``approved_at``) and
   ``release/current-release-id``, and copies the manifest into the ledger.

All AWS access goes through injected clients; the unit suite uses moto and fakes.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from finplan_contracts import ssm as contract_ssm  # noqa: E402
from finplan_contracts.validate import validate  # noqa: E402
from ulid import ULID  # noqa: E402

from infra.stacks import naming as n  # noqa: E402

__all__ = [
    "ManifestError",
    "ReleaseInfo",
    "approval_record",
    "assembly_digest",
    "build_manifest",
    "contract_pin",
    "enforced_role_names",
    "fetch_build_output",
    "image_uri",
    "mint_release_id",
    "planned_parameters",
    "publish_release",
    "stack_outputs",
    "store_build_output",
]

REPO = n.REPO
RELEASES_PREFIX = "releases/"
ADVANCED_TIER_BYTES = 4096
APPROVAL_ACTION = "ApproveProd"
SEED_ACTOR = "financemodel-release"


class ManifestError(ValueError):
    pass


# ===================================================================== build stage
def mint_release_id(now: datetime | None = None) -> str:
    return "rel_" + str(ULID.from_datetime(now) if now else ULID())


def assembly_digest(assembly: str | Path, image_digest: str | None = None) -> str:
    root = Path(assembly)
    if not (root / "manifest.json").is_file():
        raise ManifestError(f"{root} is not a cloud assembly (manifest.json missing)")
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8") + b"\0" + hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii") + b"\n")
    if image_digest:
        h.update(b"image\0" + image_digest.encode("ascii") + b"\n")
    return "sha256:" + h.hexdigest()


def contract_pin(root: str | Path) -> tuple[str, str]:
    pin = json.loads((Path(root) / "contracts-pin.json").read_text(encoding="utf-8"))
    return str(pin["version"]), "sha256:" + str(pin["sha256"])


@dataclass
class ReleaseInfo:
    release_id: str
    source_commit: str
    artifact_digest: str
    contract_version: str
    contract_digest: str
    served_contract_majors: list[int]
    region: str
    built_at: str
    image_repository: str | None = None
    image_digest: str | None = None
    base_images: dict[str, str] = field(default_factory=dict)
    rollback: bool = False
    synthetic: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def load(cls, path: str | Path) -> ReleaseInfo:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("contract_gaps_enabled", None)  # recorded by builds before contracts 1.0.0
        return cls(**data)


# ===================================================================== references
def _name(env: str, category: str, name: str) -> str:
    return contract_ssm.build(env, REPO, category, name)


def stack_outputs(cfn: Any, env: str) -> dict[str, str]:
    """Outputs of the environment's FinanceModel stacks (``{OutputKey: OutputValue}``)."""
    out: dict[str, str] = {}
    for part in ("storage", "control"):
        name = f"finplan-{env}-{REPO}-{part}"
        try:
            stacks = cfn.describe_stacks(StackName=name)["Stacks"]
        except Exception as exc:  # noqa: BLE001
            raise ManifestError(f"stack {name} is not deployed ({type(exc).__name__})") from None
        for o in stacks[0].get("Outputs") or []:
            out[str(o["OutputKey"])] = str(o["OutputValue"])
    return out


#: Stack outputs every deploy must produce (contracts 1.0.0 carries every FinanceModel matrix row,
#: so the job API, the job-execution role and the approver role always exist).
REQUIRED_OUTPUTS = ("ResearchStorageRef", "RegistryStorageRef", "JobApiRoleRef", "JobRoleRef", "JobEndpoint", "RegistryRef", "ApproverRoleRef")


def image_uri(account: str, region: str, repository: str, digest: str) -> str:
    if not digest.startswith("sha256:"):
        raise ManifestError("the job image must be referenced by digest")
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{repository}@{digest}"


def enforced_role_names(env: str, outputs: Mapping[str, str]) -> list[str]:
    """Role names the platform's budget action denies at 100% (contracts D4)."""
    names = [n.role_name(env, n.JOB_API_HANDLER), n.role_name(env, n.DISPATCHER), n.role_name(env, n.STATE_HANDLER)]
    if outputs.get("JobRoleRef"):
        names.append(n.role_name(env, n.JOB_EXECUTION))
    names += [n.deploy_role_name(env), n.exec_role_name(env), n.stage_role_name(env)]
    # The account-level pipeline and build roles are published once, by the bootstrap, at
    # /finplan/shared/financemodel/config/budget-enforced-role-names (contracts 1.0.0, D16;
    # infra.stacks.naming.tooling_role_names); a pipeline (environment writer) never writes shared.
    return names


def planned_parameters(env: str, info: ReleaseInfo, outputs: Mapping[str, str], *, account: str, deployed_job_types: list[str]) -> dict[str, tuple[str, str]]:
    """``{logical output key: (SSM name, value)}`` for every reference this deploy publishes."""
    plan: dict[str, tuple[str, str]] = {}

    def add(key: str, category: str, name: str, value: str | None) -> None:
        if value:
            plan[key] = (_name(env, category, name), value)

    add("research-storage-ref", "config", "research-storage-ref", outputs.get("ResearchStorageRef"))
    add("registry-storage-ref", "config", "registry-storage-ref", outputs.get("RegistryStorageRef"))
    add("job-api-role-ref", "job", "job-api-role-ref", outputs.get("JobApiRoleRef"))
    add("job-role-ref", "job", "job-role-ref", outputs.get("JobRoleRef"))
    add("job-endpoint", "api", "job-endpoint", outputs.get("JobEndpoint"))
    add("registry-ref", "model", "registry-ref", outputs.get("RegistryRef"))
    add("approver-role-ref", "config", "approver-role-ref", outputs.get("ApproverRoleRef"))
    if info.image_repository and info.image_digest:
        uri = image_uri(account, info.region, info.image_repository, info.image_digest)
        for jt in deployed_job_types:
            doc = {"job_type": jt, "image_uri": uri, "image_digest": info.image_digest, "release_id": info.release_id, "deployed": True}
            add(f"job-{jt.replace('_', '-')}", "job", jt.replace("_", "-"), json.dumps(doc, sort_keys=True, separators=(",", ":")))
    add("budget-enforced-role-names", "config", "budget-enforced-role-names", ",".join(enforced_role_names(env, outputs)))
    return plan


# ===================================================================== manifest
def build_manifest(info: ReleaseInfo, env: str, *, deployed_at: str, previous_release_id: str | None, outputs: Mapping[str, str], approval: Mapping[str, str] | None = None, rolled_back_from: str | None = None, deployed_job_types: list[str] | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "repo": REPO,
        "environment": env,
        "region": info.region,
        "release_id": info.release_id,
        "source_commit": info.source_commit,
        "artifact_digest": info.artifact_digest,
        "contract_version": info.contract_version,
        "contract_digest": info.contract_digest,
        "deployed_at": deployed_at,
        "previous_release_id": previous_release_id,
        "outputs": dict(outputs),
        "served_contract_majors": sorted(set(info.served_contract_majors)),
        "synthetic": info.synthetic,
    }
    if approval:
        doc["approved_by"] = approval["approved_by"]
        doc["approved_at"] = approval["approved_at"]
    if rolled_back_from is not None:
        doc["rolled_back_from"] = rolled_back_from
    res = validate(doc, "release-manifest")
    problems = [f"{i.pointer or '/'}: {i.message}" for i in res.issues]
    problems += contract_ssm.validate_value(_name(env, "release", "manifest"), json.dumps(doc))
    if problems:
        raise ManifestError("release manifest is invalid: " + "; ".join(dict.fromkeys(problems)))
    return doc


def _ts(value: Any) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(value)


def approval_record(codepipeline: Any, pipeline_name: str, execution_id: str, action_name: str = APPROVAL_ACTION) -> dict[str, str]:
    token: str | None = None
    while True:
        kw: dict[str, Any] = {"pipelineName": pipeline_name, "filter": {"pipelineExecutionId": execution_id}}
        if token:
            kw["nextToken"] = token
        resp = codepipeline.list_action_executions(**kw)
        for d in resp.get("actionExecutionDetails") or []:
            if d.get("actionName") == action_name and d.get("status") == "Succeeded":
                who = d.get("updatedBy") or ((d.get("output") or {}).get("executionResult") or {}).get("externalExecutionSummary")
                when = d.get("lastUpdateTime")
                if who and when:
                    return {"approved_by": str(who), "approved_at": _ts(when)}
        token = resp.get("nextToken")
        if not token:
            break
    raise ManifestError(f"no succeeded manual approval '{action_name}' in pipeline execution {execution_id}; prod manifests require approved_by and approved_at")


# ===================================================================== publish
def _get(ssm: Any, name: str) -> str | None:
    try:
        return ssm.get_parameter(Name=name)["Parameter"]["Value"]
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
        if code == "ParameterNotFound":
            return None
        raise


def _put(ssm: Any, env: str, name: str, value: str) -> None:
    decision = contract_ssm.check_write(name, contract_ssm.Writer(REPO, "pipeline", env))
    if not decision:
        raise ManifestError("; ".join(decision.reasons))
    problems = contract_ssm.validate_value(name, value)
    if problems:
        raise ManifestError(f"{name}: " + "; ".join(problems))
    tier = "Advanced" if len(value.encode("utf-8")) > ADVANCED_TIER_BYTES else "Standard"
    ssm.put_parameter(Name=name, Value=value, Type="String", Overwrite=True, Tier=tier)


def deployed_job_types(env: str) -> list[str]:
    from finplan_model.core.config import load_config

    cfg = load_config(env, ROOT / "config")
    return sorted(name for name, jt in cfg.job_types.items() if jt.deployed)


def publish_release(
    info: ReleaseInfo,
    env: str,
    *,
    ssm: Any,
    cfn: Any,
    account: str,
    s3: Any | None = None,
    store_bucket: str | None = None,
    now: datetime | None = None,
    approval: Mapping[str, str] | None = None,
    seed: Callable[[Mapping[str, str]], Any] | None = None,
    job_types: list[str] | None = None,
) -> dict[str, Any]:
    """Publish the references, the manifest and the current-release pointer (DEP-04)."""
    if env == "prod" and not approval:
        raise ManifestError("prod manifests require the approval record (approved_by, approved_at)")
    outputs = stack_outputs(cfn, env)
    for required in REQUIRED_OUTPUTS:
        if not outputs.get(required):
            raise ManifestError(f"the {env} deploy did not produce the output {required}")
    if seed is not None and info.image_digest:
        seed(outputs)
    types = job_types if job_types is not None else deployed_job_types(env)
    plan = planned_parameters(env, info, outputs, account=account, deployed_job_types=types)
    for _key, (name, value) in sorted(plan.items()):
        _put(ssm, env, name, value)
    pointer = _name(env, "release", "current-release-id")
    manifest_name = _name(env, "release", "manifest")
    current = _get(ssm, pointer)
    previous: str | None = current
    rolled_back_from: str | None = None
    if current == info.release_id:
        existing = json.loads(_get(ssm, manifest_name) or "{}")
        previous = existing.get("previous_release_id")
        rolled_back_from = existing.get("rolled_back_from")
    elif info.rollback:
        rolled_back_from = current
    manifest = build_manifest(info, env, deployed_at=_ts(now or datetime.now(UTC)), previous_release_id=previous, approval=approval, rolled_back_from=rolled_back_from, outputs={k: name for k, (name, _v) in plan.items()})
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    _put(ssm, env, manifest_name, body)
    _put(ssm, env, pointer, info.release_id)
    if s3 is not None and store_bucket:
        s3.put_object(Bucket=store_bucket, Key=f"{RELEASES_PREFIX}{info.release_id}/manifests/{env}.json", Body=body.encode("utf-8"), ContentType="application/json")
    return manifest


def registry_seeder(s3: Any, env: str, image_digest: str, *, now: Callable[[], datetime] | None = None) -> Callable[[Mapping[str, str]], dict[str, Any]]:
    """Seed the environment's registry with the baselines for ``image_digest`` (design D9)."""

    def seed(outputs: Mapping[str, str]) -> dict[str, Any]:
        from finplan_model.registry import ModelRegistry, S3RegistryStore, seed_baselines

        registry = ModelRegistry(S3RegistryStore(s3, outputs["RegistryStorageRef"]))
        return seed_baselines(registry, image_digest, actor=SEED_ACTOR)

    return seed


# ===================================================================== release ledger
def zip_dir(directory: Path) -> bytes:
    from scripts.publish_assets import deterministic_zip

    return deterministic_zip(directory)


def store_build_output(s3: Any, bucket: str, info: ReleaseInfo, out_dir: str | Path) -> str:
    key = f"{RELEASES_PREFIX}{info.release_id}/build-output.zip"
    s3.put_object(Bucket=bucket, Key=key, Body=zip_dir(Path(out_dir)), IfNoneMatch="*")
    s3.put_object(Bucket=bucket, Key=f"{RELEASES_PREFIX}{info.release_id}/release-info.json", Body=info.to_json().encode("utf-8"), IfNoneMatch="*")
    return key


def fetch_build_output(s3: Any, bucket: str, release_id: str, out_dir: str | Path) -> ReleaseInfo:
    if not release_id.startswith("rel_"):
        raise ManifestError(f"{release_id!r} is not a release_id")
    try:
        body = s3.get_object(Bucket=bucket, Key=f"{RELEASES_PREFIX}{release_id}/build-output.zip")["Body"].read()
    except Exception as exc:
        raise ManifestError(f"release {release_id} has no stored build output in the pipeline store") from exc
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        for member in zf.namelist():
            if member.startswith("/") or ".." in Path(member).parts:
                raise ManifestError(f"unsafe path in stored build output: {member}")
        zf.extractall(out)
    info = ReleaseInfo.load(out / "release-info.json")
    if info.release_id != release_id:
        raise ManifestError("stored build output belongs to another release")
    if assembly_digest(out / "cdk.out", info.image_digest) != info.artifact_digest:
        raise ManifestError("stored build output digest does not match its release record")
    info.rollback = True
    (out / "release-info.json").write_text(info.to_json(), encoding="utf-8")
    return info
