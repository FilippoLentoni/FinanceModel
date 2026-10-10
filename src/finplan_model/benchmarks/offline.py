"""Network-isolated Qwen Training Job hand-off and verified output import.

SageMaker downloads channels and uploads /opt/ml/model outside the isolated container.
The container has no AWS credentials and performs no network calls. No endpoint lifecycle
exists: the existing dispatcher handles Training Job readiness, cancellation, retries and leases.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from finplan_model.core.artifacts import ArtifactRef, LocalArtifactStore, S3ArtifactStore, canonical_json_bytes, sha256_checksum, verify_checksum
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.core.outcome import failed_job_result, require_valid
from finplan_model.core.platform import ResolvedSnapshot, SnapshotContent, SnapshotReader
from finplan_model.jobs.market_loader import market_from_content
from finplan_model.jobs.spec import validate_run_spec

from .weights import PREFIX
from .qwen import MODEL_ID, REVISION


class OfflineBatchIO:
    def __init__(self, s3, bucket, platform, run_io, artifacts):
        self.s3, self.bucket, self.platform, self.run_io, self.artifacts = s3, bucket, platform, run_io, artifacts

    def prepare(self, run, spec, definition):
        if not definition.get("code_prefix") or not definition.get("code_checksum"):
            raise FinplanError.precondition("offline Qwen code channel has not been released", reason="qwen_code_not_released")
        try:
            staged = json.loads(self.s3.get_object(Bucket=self.bucket, Key=PREFIX + "STAGED.json")["Body"].read())
            manifest = self.s3.get_object(Bucket=self.bucket, Key=PREFIX + "MANIFEST.sha256")["Body"].read()
            if staged["manifest_checksum"] != sha256_checksum(manifest) or staged.get("model_id") != MODEL_ID or staged.get("revision") != REVISION:
                raise ValueError("manifest")
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NotFound") or isinstance(exc, (ValueError, KeyError)):
                raise FinplanError.precondition("stage and verify the pinned Qwen weights before GPU approval", reason="qwen_weights_not_staged") from None
            raise
        content = SnapshotReader(self.platform).load(run["input_snapshot_id"])
        bundle = {"spec": spec, "record": content.snapshot.record, "manifest": content.manifest, "payload": content.payload, "verified": content.verified}
        data = canonical_json_bytes(bundle)
        key = "scratch/offline/" + run["run_id"] + "/input.json"
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType="application/json")
        channels = {"bundle": "s3://" + self.bucket + "/scratch/offline/" + run["run_id"] + "/", "weights": "s3://" + self.bucket + "/" + PREFIX,
            "code": "s3://" + self.bucket + "/" + definition["code_prefix"]}
        return {"channels": channels, "bundle_checksum": sha256_checksum(data), "code_checksum": definition["code_checksum"]}

    def import_result(self, run, description):
        uri = (description.get("ModelArtifacts") or {}).get("S3ModelArtifacts", "")
        parsed = urlsplit(uri)
        prefix = "scratch/training-output/" + run["job_name"] + "/output/"
        if parsed.scheme != "s3" or parsed.netloc != self.bucket or not parsed.path.lstrip("/").startswith(prefix) or not parsed.path.endswith("/model.tar.gz"):
            raise FinplanError.precondition("isolated result path is outside this run's trusted output prefix", reason="offline_output_path_invalid")
        body = self.s3.get_object(Bucket=self.bucket, Key=parsed.path.lstrip("/"))["Body"].read(32 * 2**20 + 1)
        if len(body) > 32 * 2**20:
            raise FinplanError.precondition("isolated result archive exceeds its bound", reason="offline_output_bound")
        files, total = {}, 0
        try:
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
                for member in archive:
                    path = PurePosixPath(member.name)
                    name = path.as_posix()
                    if path.is_absolute() or ".." in path.parts:
                        raise FinplanError.precondition("isolated archive contains an unsafe member", reason="offline_output_invalid")
                    if member.isdir():
                        continue
                    if not member.isfile() or name in files:
                        raise FinplanError.precondition("isolated archive contains an unsafe member", reason="offline_output_invalid")
                    total += member.size
                    if total > 128 * 2**20 or len(files) >= 1000:
                        raise FinplanError.precondition("isolated expanded archive exceeds its bound", reason="offline_output_bound")
                    files[name] = archive.extractfile(member).read()
            manifest = json.loads(files["MANIFEST.json"])
        except (tarfile.TarError, KeyError, ValueError, OSError, EOFError):
            raise FinplanError.precondition("isolated output archive is invalid", reason="offline_output_invalid") from None
        if not isinstance(manifest, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in manifest.items()) or set(manifest) != set(files) - {"MANIFEST.json"}:
            raise FinplanError.precondition("isolated output manifest is incomplete", reason="offline_output_invalid")
        for name, checksum in manifest.items():
            verify_checksum(files[name], checksum, what="offline result")
        try:
            result = require_valid(json.loads(files["result.json"]), "job-result")
        except (KeyError, ValueError, TypeError):
            raise FinplanError.precondition("isolated job result is invalid", reason="offline_output_invalid") from None
        if result["run_id"] != run["run_id"] or result.get("configuration_id") != run["configuration_id"] or result.get("input_snapshot_id") != run["input_snapshot_id"]:
            raise FinplanError.precondition("isolated result identity differs from the run", reason="offline_output_invalid")
        for ref in result.get("artifacts", []):
            name = "artifacts/" + ref["kind"] + "/" + ref["artifact_id"]
            if name not in files:
                raise FinplanError.precondition("isolated output omits a declared artifact", reason="offline_output_invalid")
            data = files[name]
            verify_checksum(data, ref["checksum"], what="offline artifact")
            stored = self.artifacts.put(data, kind=ref["kind"], content_type=ref["content_type"], artifact_id=ref["artifact_id"], synthetic=ref.get("synthetic"), domain=ref.get("domain"))
            if stored.checksum != ref["checksum"]:
                raise FinplanError.precondition("isolated artifact import checksum differs", reason="offline_output_invalid")
        self.run_io.put_result(run["run_id"], result)
        return result


def main():
    from finplan_model.jobs.handlers import JobInputs
    from .jobs import swarm_mode_a
    bundle_path = Path("/opt/ml/input/data/bundle/input.json")
    data = bundle_path.read_bytes()
    verify_checksum(data, os.environ["FINPLAN_OFFLINE_BUNDLE_SHA256"], what="offline input")
    bundle = json.loads(data)
    spec = validate_run_spec(bundle["spec"])
    ctx = RunContext(environment=spec["environment"], purpose=spec["purpose"], run_id=spec["run_id"], correlation_id=spec["correlation_id"], image_digest=spec.get("image_digest"), synthetic=spec["synthetic"])
    content = SnapshotContent(ResolvedSnapshot(spec["input_snapshot_id"], bundle["record"]), bundle["manifest"], bundle["payload"], bundle["verified"])
    root = Path("/opt/ml/model")
    artifacts = LocalArtifactStore(root / "artifacts")
    try:
        inp = JobInputs(ctx=ctx, spec=spec, market=market_from_content(content), snapshot=content, artifacts=artifacts)
        result = swarm_mode_a(inp)
    except Exception as exc:
        result = failed_job_result(ctx, exc, run_id=spec["run_id"])
        result.update(configuration_id=spec["configuration_id"], input_snapshot_id=spec["input_snapshot_id"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "result.json").write_bytes(canonical_json_bytes(result))
    manifest = {p.relative_to(root).as_posix(): sha256_checksum(p.read_bytes()) for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
    (root / "MANIFEST.json").write_bytes(canonical_json_bytes(manifest))
    return 0 if result["completion_status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
