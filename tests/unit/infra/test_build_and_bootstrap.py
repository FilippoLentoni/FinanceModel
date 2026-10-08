"""Build stage, Lambda bundle, container image, asset publishing, stage runner and bootstrap
(task 10.5; DEP-01..DEP-03, PIPE-style "no artifact on a failing gate"; ENV-12, ENV-13) with fakes:
no docker, no network, no AWS."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import boto3
import pytest
from finplan_contracts.bootstrap import BootstrapConfig, BootstrapStop, Clients
from moto import mock_aws

from infra.stacks import lambda_code
from scripts import bootstrap as fm_bootstrap
from scripts import build_stage, container_image, lambda_bundle, stage_runner
from scripts.release import ReleaseInfo

ROOT = Path(__file__).resolve().parents[3]
COMMIT = "0123456789abcdef0123456789abcdef01234567"
DIGEST = "sha256:" + "e" * 64


# ----------------------------------------------------------------- Lambda code (source-only refusal)
def test_release_synth_refuses_source_only_code(tmp_path):
    assert lambda_code.release_mode({"FINPLAN_RELEASE_BUILD": "1"})
    assert lambda_code.release_mode({"CODEBUILD_BUILD_ID": "b"})
    assert not lambda_code.release_mode({"CODEBUILD_BUILD_ID": "b", "FINPLAN_RELEASE_BUILD": "0"})
    with pytest.raises(lambda_code.SourceOnlyCodeError):
        lambda_code.function_code({"FINPLAN_RELEASE_BUILD": "1"})
    with pytest.raises(lambda_code.SourceOnlyCodeError, match="incomplete"):
        lambda_code.function_code({"FINPLAN_RELEASE_BUILD": "1", "FINPLAN_LAMBDA_BUNDLE_DIR": str(tmp_path)})
    assert "finplan_contracts missing" in lambda_code.bundle_problems(tmp_path)


def test_bundle_commands_use_the_lock_hashes_and_lambda_platform(tmp_path):
    calls: list[list[str]] = []

    def runner(cmd, cwd):
        calls.append(cmd)
        if cmd[1] == "pip":  # emulate the install: the closure the gate requires
            target = Path(cmd[cmd.index("--target") + 1])
            for name in ("finplan_contracts", "jsonschema", "rfc8785", "numpy", "boto3"):
                (target / name).mkdir(parents=True, exist_ok=True)
        return ""

    manifest = lambda_bundle.build_bundle(ROOT, tmp_path / "b", runner=runner)
    export, install = calls
    assert export[1:3] == ["export", "--frozen"] and "--no-dev" in export
    assert "--require-hashes" in install and "--only-binary" in install
    assert install[install.index("--python-platform") + 1] == "aarch64-manylinux_2_28"  # arm64, as the platform
    assert manifest["functions"]["job-api-handler"] == "finplan_model.control.handlers.api_handler"
    assert lambda_bundle.verify_bundle(tmp_path / "b") == []
    assert (tmp_path / "b" / "finplan_model" / "registry" / "handlers.py").is_file()


# ----------------------------------------------------------------- container image (built once, by digest)
class FakeEcr:
    def get_authorization_token(self):
        return {"authorizationData": [{"authorizationToken": base64.b64encode(b"AWS:token").decode()}]}

    def describe_images(self, repositoryName, imageIds):  # noqa: N803
        return {"imageDetails": [{"imageDigest": DIGEST, "imageTags": [imageIds[0]["imageTag"]]}]}


def test_image_built_once_pinned_and_pushed_by_digest():
    calls: list[list[str]] = []

    def run(cmd, cwd, input=None):  # noqa: A002
        calls.append(cmd)
        if cmd[:2] == ["docker", "inspect"]:
            ref = cmd[-1]
            name = ref.rsplit(":", 1)[0]
            return f"{name}@sha256:{'f' * 64}\n"
        return ""

    img = container_image.build_and_push(release_id="rel_01KDVDNAZ83BAMMYCEGWF33DPM", source_commit=COMMIT, account="<account-id>", region="us-east-2", repository="finplan-shared-financemodel-cpu-images", ecr=FakeEcr(), root=ROOT, run=run)
    assert img.digest == DIGEST
    builds = [c for c in calls if c[:2] == ["docker", "build"]]
    assert len(builds) == 1 and "linux/amd64" in builds[0]
    assert all("@sha256:" in a for a in builds[0] if a.startswith(("PYTHON_IMAGE=", "UV_IMAGE=")))
    assert set(img.base_images) == {"PYTHON_IMAGE", "UV_IMAGE"}
    with pytest.raises(container_image.ImageError):
        container_image.build_and_push(release_id="latest", source_commit=COMMIT, account="a", region="r", repository="x", ecr=FakeEcr(), root=ROOT, run=run)
    # the image import check runs after the build and before the push (platform lesson L3)
    order = [c[:2] for c in calls[: len(calls)]]
    checks = [c for c in calls if c[:2] == ["docker", "run"]]
    assert len(checks) == 1 and "--network" in checks[0] and set(container_image.IMAGE_IMPORTS) <= set(checks[0])
    assert order.index(["docker", "build"]) < order.index(["docker", "run"]) < order.index(["docker", "push"])


def test_image_without_its_dependencies_is_never_pushed():
    calls: list[list[str]] = []

    def run(cmd, cwd, input=None):  # noqa: A002
        calls.append(cmd)
        if cmd[:2] == ["docker", "inspect"]:
            return f"{cmd[-1].rsplit(':', 1)[0]}@sha256:{'f' * 64}\n"
        if cmd[:2] == ["docker", "run"]:
            raise container_image.ImageError("ModuleNotFoundError: No module named 'finplan_contracts'")
        return ""

    with pytest.raises(container_image.ImageError, match="nothing pushed"):
        container_image.build_and_push(release_id="rel_01KDVDNAZ83BAMMYCEGWF33DPM", source_commit=COMMIT, account="<account-id>", region="us-east-2", repository="finplan-shared-financemodel-cpu-images", ecr=FakeEcr(), root=ROOT, run=run)
    assert not [c for c in calls if c[:2] in (["docker", "push"], ["docker", "login"])]


# ----------------------------------------------------------------- build stage
def _fake_synth(out: Path) -> Path:
    out.mkdir(parents=True)
    (out / "manifest.json").write_text("{}")
    (out / "x.template.json").write_text('{"Resources": {}}')
    return out


def _fake_bundle(root: Path, out: Path) -> dict:
    return {"unzipped_bytes": 1, "files": 1, "python_platform": "aarch64-manylinux_2_28"}


def test_build_stage_produces_build_output(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for rel in ("config", "vendor/finplan-contracts"):
        (root / rel).mkdir(parents=True)
    for f in ("beta", "gamma", "prod", "shared"):
        (root / "config" / f"{f}.json").write_text((ROOT / "config" / f"{f}.json").read_text())
    (root / "contracts-pin.json").write_text((ROOT / "contracts-pin.json").read_text())
    image = SimpleNamespace(repository="finplan-shared-financemodel-cpu-images", digest=DIGEST, base_images={"PYTHON_IMAGE": "x@sha256:" + "1" * 64})
    info = build_stage.run_build(root, tmp_path / "out", source_commit=COMMIT, synth_fn=_fake_synth, bundle_fn=_fake_bundle, image_fn=lambda rid, c: image, gates=(), log=lambda s: None)
    saved = ReleaseInfo.load(tmp_path / "out" / "release-info.json")
    assert saved.release_id == info.release_id and saved.release_id.startswith("rel_")
    assert saved.image_digest == DIGEST and saved.contract_version == "1.1.0" and saved.served_contract_majors == [1]
    assert (tmp_path / "out" / "cdk.out" / "manifest.json").is_file() and (tmp_path / "out" / "config" / "beta.json").is_file()


def test_failing_gate_produces_no_artifact(tmp_path, monkeypatch):
    from scripts import build_gates

    monkeypatch.setattr(build_gates, "GATES", [("always-fails", "pre", lambda ctx: ["planted failure"])])
    with pytest.raises(build_stage.BuildFailed, match="pre-synth gates failed"):
        build_stage.run_build(ROOT, tmp_path / "out", source_commit=COMMIT, synth_fn=_fake_synth, bundle_fn=_fake_bundle, log=lambda s: None)
    assert not (tmp_path / "out").exists()
    with pytest.raises(build_stage.BuildFailed, match="40-character"):
        build_stage.run_build(ROOT, tmp_path / "out", source_commit="main", log=lambda s: None)


# ----------------------------------------------------------------- asset publishing
class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.puts: list[dict] = []

    def head_object(self, Bucket, Key):  # noqa: N803
        if (Bucket, Key) not in self.objects:
            err = Exception("404")
            err.response = {"Error": {"Code": "404"}}  # type: ignore[attr-defined]
            raise err
        return {}

    def put_object(self, **kw):
        self.puts.append(kw)
        self.objects[(kw["Bucket"], kw["Key"])] = kw["Body"]


def test_assets_go_to_the_financemodel_store_write_once(assembly):
    from scripts.publish_assets import planned_uploads, publish

    store = "finplan-shared-financemodel-pipeline-store-<account-id>"
    uploads = planned_uploads(assembly.directory, account="<account-id>", region="us-east-2")
    assert uploads and all(b == store and k.startswith("assets/") for _a, _p, _pk, b, k in uploads)
    s3 = FakeS3()
    first = publish(assembly.directory, s3, account="<account-id>", region="us-east-2")
    assert all(p.uploaded for p in first) and all(put["IfNoneMatch"] == "*" for put in s3.puts)
    second = publish(assembly.directory, s3, account="<account-id>", region="us-east-2")
    assert not any(p.uploaded for p in second)  # content-addressed: published once


# ----------------------------------------------------------------- stage runner
def test_promotion_check_passes_the_1x_pin_in_every_environment(tmp_path):
    """Contracts 1.0.0 removed the beta-only stop: PromotionCheck promotes to gamma and prod."""
    info = ReleaseInfo(release_id="rel_01KDVDNAZ83BAMMYCEGWF33DPM", source_commit=COMMIT, artifact_digest=DIGEST, contract_version="1.0.0", contract_digest=DIGEST, served_contract_majors=[1], region="us-east-2", built_at="2026-10-08T00:00:00Z", image_digest=DIGEST)
    for env in ("beta", "gamma", "prod"):
        assert stage_runner.precheck_action(env, info) == [], env
    info.image_digest = None
    assert any("digest" in p for p in stage_runner.precheck_action("beta", info))


def test_zero_executed_tests_fail_the_stage(tmp_path):
    def run(cmd, cwd, env):
        junit = Path(next(a for a in cmd if a.startswith("--junitxml=")).split("=", 1)[1])
        junit.write_text('<testsuites><testsuite tests="2" skipped="2" failures="0" errors="0"/></testsuites>')
        assert env["FINPLAN_TARGET_ENV"] == "beta"
        return SimpleNamespace(returncode=0)

    assert stage_runner.tests_action("beta", run=run, environ={}, out=lambda s: None) == 1

    def run_ok(cmd, cwd, env):
        junit = Path(next(a for a in cmd if a.startswith("--junitxml=")).split("=", 1)[1])
        junit.write_text('<testsuites><testsuite tests="2" skipped="0" failures="0" errors="0"/></testsuites>')
        return SimpleNamespace(returncode=0)

    assert stage_runner.tests_action("prod", run=run_ok, environ={}, out=lambda s: None) == 0


# ----------------------------------------------------------------- bootstrap (ENV-12, ENV-13), all mocked
class _Sts:
    def __init__(self, account):
        self.account = account

    def get_caller_identity(self):
        return {"Account": self.account, "Arn": "arn:aws:iam::" + self.account + ":root"}


class _Conn:
    def __init__(self, status="AVAILABLE"):
        self.status = status
        self.asked: list[str] = []

    def get_connection(self, ConnectionArn):  # noqa: N803
        self.asked.append(ConnectionArn)
        return {"Connection": {"ConnectionStatus": self.status}}


class _Pipeline:
    def __init__(self, status="Succeeded"):
        self.status = status
        self.calls: list[str] = []

    def disable_stage_transition(self, **kw):
        self.calls.append("disable")

    def enable_stage_transition(self, **kw):
        self.calls.append("enable")

    def start_pipeline_execution(self, name):
        self.calls.append("start")
        return {"pipelineExecutionId": "exec-1"}

    def get_pipeline_state(self, name):
        return {"stageStates": [{"stageName": "Source", "latestExecution": {"pipelineExecutionId": "exec-1", "status": self.status}}]}


SYNTHETIC_PRICE = 0.5  # placeholder, never a real price


def _price_item(component: str, usd: float) -> str:
    return json.dumps({"product": {"attributes": {"component": component, "instanceName": "ml.m5.xlarge"}}, "terms": {"OnDemand": {"t": {"priceDimensions": {"d": {"unit": "Hrs", "pricePerUnit": {"USD": str(usd)}}}}}}})


class _Pricing:
    def __init__(self, sagemaker_price: float | None = SYNTHETIC_PRICE):
        self.sagemaker_price = sagemaker_price
        self.calls: list[dict] = []

    def get_products(self, **kw):
        self.calls.append(kw)
        if kw.get("ServiceCode") != "AmazonSageMaker" or self.sagemaker_price is None:
            return {"PriceList": []}
        return {"PriceList": [_price_item("Hosting", self.sagemaker_price / 10), _price_item("Processing", self.sagemaker_price)]}


@pytest.fixture
def boot(assembly, tmp_path):
    with mock_aws():
        ssm = boto3.client("ssm", region_name="us-east-2")
        ssm.put_parameter(Name="/finplan/shared/financialplanning/config/codeconnection-ref", Value="example-connection-ref", Type="String")
        yield SimpleNamespace(ssm=ssm, assembly=assembly.directory, tmp=tmp_path)


def _clients(boot, conn=None, pipeline=None, account="<account-id>", pricing=None):
    return Clients(sts=_Sts(account), codeconnections=conn or _Conn(), ssm=boot.ssm, codepipeline=pipeline or _Pipeline(), pricing=pricing or _Pricing())


def test_bootstrap_reuses_the_platform_connection_and_deploys_only_tooling(boot):
    config = fm_bootstrap.load_bootstrap_config({"account_id": "<account-id>", "primary_region": "us-east-2"}, boot.ssm)
    assert config.repo == "financemodel" and config.codeconnection_arn == "example-connection-ref" and config.pipeline_name == "finplan-shared-financemodel-pipeline"
    runs: list[list[str]] = []
    out: list[str] = []
    conn, pipe = _Conn(), _Pipeline()
    report = fm_bootstrap.run(config, _clients(boot, conn, pipe), session_region="us-east-2", assembly=boot.assembly, bootstrap_dir=boot.tmp / "boot", approve=lambda plan: True, runner=lambda cmd, cwd, env: (runs.append(cmd), SimpleNamespace(returncode=0))[1], record_path=boot.tmp / "dry-run.json", sleep=lambda s: None, out=out.append)
    assert report.completed and report.plan.stack_names == sorted(fm_bootstrap.BOOTSTRAP_STACKS) or set(report.plan.stack_names) == set(fm_bootstrap.BOOTSTRAP_STACKS)
    (cmd,) = runs
    assert "--all" in cmd and any(a.endswith("SourceDryRunPassed=false") for a in cmd)
    assert conn.asked == ["example-connection-ref"] and pipe.calls == ["disable", "start", "enable"]
    assert boot.ssm.get_parameter(Name="/finplan/shared/financemodel/config/codeconnection-ref")["Parameter"]["Value"] == "example-connection-ref"
    names = [p["Name"] for p in boot.ssm.describe_parameters()["Parameters"]]
    # FinanceModel writes no budget parameter; it only publishes its own tooling role names (D16)
    assert [n for n in names if "budget" in n or "cost-ceiling" in n] == ["/finplan/shared/financemodel/config/budget-enforced-role-names"]
    assert json.loads((boot.tmp / "dry-run.json").read_text())["deploy_stages_enabled"] is True
    assert any("contract defaults" in line for line in out)  # allocation absent: reported, never written
    # shared tooling role names (contracts 1.0.0, D16), valid for the platform's budget action
    roles = boot.ssm.get_parameter(Name="/finplan/shared/financemodel/config/budget-enforced-role-names")["Parameter"]["Value"]
    assert roles.split(",") == ["finplan-shared-financemodel-pipeline-role", "finplan-shared-financemodel-pipeline-build-project-role"]
    from finplan_contracts import ssm as contract_ssm

    assert "/finplan/shared/financemodel/config/budget-enforced-role-names" in contract_ssm.budget_enforced_role_name_keys()
    assert contract_ssm.validate_value("/finplan/shared/financemodel/config/budget-enforced-role-names", roles) == []
    # instance prices fetched from the Price List API (Processing component), one per environment
    for env in ("beta", "gamma", "prod"):
        doc = json.loads(boot.ssm.get_parameter(Name=f"/finplan/{env}/financemodel/config/instance-prices")["Parameter"]["Value"])
        assert doc["usd_per_hour"] == {"ml.m5.xlarge": SYNTHETIC_PRICE} and doc["source"] == "AWS Price List API" and doc["component"] == "Processing"
    # integration-snapshot-id is the operator's: reported, never written
    assert not [n for n in names if n.endswith("/integration-snapshot-id")]
    assert any("[ACTION] /finplan/beta/financemodel/config/integration-snapshot-id" in line for line in out)


def test_bootstrap_stops_before_deploying(boot):
    config = BootstrapConfig.from_mapping({"account_id": "<account-id>", "primary_region": "us-east-2", "codeconnection_arn": "example-connection-ref", "repo": "financemodel"})
    runs: list = []
    runner = lambda cmd, cwd, env: runs.append(cmd)  # noqa: E731
    kw = dict(session_region="us-east-2", assembly=boot.assembly, bootstrap_dir=boot.tmp / "boot", runner=runner, sleep=lambda s: None, out=lambda s: None)
    with pytest.raises(BootstrapStop) as declined:
        fm_bootstrap.run(config, _clients(boot), approve=lambda plan: False, **kw)
    assert declined.value.step == "approval"
    with pytest.raises(BootstrapStop) as wrong_account:
        fm_bootstrap.run(config, _clients(boot, account="<other-account>"), approve=lambda plan: True, **kw)
    assert wrong_account.value.step == "caller"
    with pytest.raises(BootstrapStop) as unavailable:
        fm_bootstrap.run(config, _clients(boot, conn=_Conn("PENDING")), approve=lambda plan: True, **kw)
    assert unavailable.value.step == "connection"
    assert runs == []
    with pytest.raises(BootstrapStop) as dry:
        fm_bootstrap.run(config, _clients(boot, pipeline=_Pipeline("Failed")), approve=lambda plan: True, runner=lambda cmd, cwd, env: SimpleNamespace(returncode=0), **{k: v for k, v in kw.items() if k != "runner"})
    assert dry.value.step == "source-dry-run" and "GitHub App installation" in dry.value.message
    with pytest.raises(BootstrapStop, match="not synthesized"):
        fm_bootstrap.bootstrap_assembly(boot.tmp / "missing", boot.tmp / "b2")


def _offline_build(tmp_path, name, **kw):
    import shutil

    if not shutil.which("uv"):
        pytest.skip("uv is not on PATH")
    try:
        return lambda_bundle.build_bundle(ROOT, tmp_path / name, offline=True, **kw)
    except lambda_bundle.BundleError as exc:
        if "offline" in str(exc).lower() or "cache" in str(exc).lower() or "network" in str(exc).lower():
            pytest.skip(f"wheels not in the local uv cache: {str(exc)[:200]}")
        raise


def test_real_bundle_imports_every_handler_from_the_bundle_alone(tmp_path):
    """Builds the bundle for real from uv.lock (offline, from the local uv cache) for THIS host's
    platform, then imports every handler with only the bundle on ``sys.path`` (the source-only
    package incident of the platform). The arm64 release bundle cannot be imported on the build
    host; the next test checks its binaries."""
    manifest = _offline_build(tmp_path, "bundle", python_platform=None)
    assert manifest["unzipped_bytes"] < lambda_bundle.LAMBDA_UNZIPPED_LIMIT
    lambda_bundle.import_check(tmp_path / "bundle")


def test_release_bundle_is_arm64_and_complete(tmp_path):
    """Regression (platform lesson L3): the release bundle carries the locked dependency closure
    built for the functions' architecture (arm64, ``aarch64-manylinux_2_28``), never x86 wheels
    and never a source-only package."""
    from infra.stacks.control import LAMBDA_ARCHITECTURE

    assert LAMBDA_ARCHITECTURE.name == "arm64"
    assert lambda_bundle.LAMBDA_PLATFORM == "aarch64-manylinux_2_28"
    manifest = _offline_build(tmp_path, "arm")
    assert manifest["python_platform"] == "aarch64-manylinux_2_28"
    assert lambda_bundle.verify_bundle(tmp_path / "arm") == []
    assert lambda_bundle.foreign_binaries(tmp_path / "arm") == []
    assert any((tmp_path / "arm" / "numpy").rglob("*aarch64*.so")), "numpy extension modules must be the arm64 build"


def test_foreign_binaries_detects_an_x86_shared_object(tmp_path):
    so = tmp_path / "lib" / "x.so"
    so.parent.mkdir()
    so.write_bytes(b"\x7fELF\x02\x01\x01" + b"\0" * 11 + (0x3E).to_bytes(2, "little") + b"\0" * 40)  # EM_X86_64
    assert lambda_bundle.foreign_binaries(tmp_path) == ["lib/x.so"]
    so.write_bytes(b"\x7fELF\x02\x01\x01" + b"\0" * 11 + (0xB7).to_bytes(2, "little") + b"\0" * 40)  # EM_AARCH64
    assert lambda_bundle.foreign_binaries(tmp_path) == []
