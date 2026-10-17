"""FinanceModel-owned staging of the pinned public checkpoint; never reuses another project bucket."""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError

from .qwen import MODEL_ID, REVISION

PREFIX = "scratch/qwen-weights/" + REVISION + "/"
MAX_TOTAL_BYTES = 60 * 2**30


def stage_weights(s3, bucket, *, open_url=urllib.request.urlopen):
    try:
        staged = json.loads(s3.get_object(Bucket=bucket, Key=PREFIX + "STAGED.json")["Body"].read())
        manifest = s3.get_object(Bucket=bucket, Key=PREFIX + "MANIFEST.sha256")["Body"].read()
        if staged["model_id"] == MODEL_ID and staged["revision"] == REVISION and staged["manifest_checksum"] == sha256_checksum(manifest):
            return {**staged, "already_staged": True}
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code not in ("NoSuchKey", "404", "NotFound") and not isinstance(exc, (KeyError, ValueError)):
            raise
    with open_url("https://huggingface.co/api/models/" + MODEL_ID + "/revision/" + REVISION + "?blobs=true", timeout=20) as response:
        metadata = json.loads(response.read(2_000_000))
    if metadata.get("sha") != REVISION:
        raise FinplanError.precondition("Hugging Face revision identity does not match the pin", reason="qwen_revision_mismatch")
    rows, total = [], 0
    for item in metadata.get("siblings", []):
        name = item.get("rfilename", "")
        if not name or Path(name).name != name or name in ("STAGED.json", "MANIFEST.sha256"):
            continue
        expected = (item.get("lfs") or {}).get("sha256")
        with tempfile.TemporaryDirectory(prefix="qwen-stage-") as directory:
            path = Path(directory) / name
            digest = hashlib.sha256()
            with open_url("https://huggingface.co/" + MODEL_ID + "/resolve/" + REVISION + "/" + name, timeout=30) as response, path.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    total += len(chunk)
                    if total > MAX_TOTAL_BYTES:
                        raise FinplanError.precondition("checkpoint download exceeds the approved storage bound", reason="qwen_weight_size_bound")
                    digest.update(chunk)
                    output.write(chunk)
            checksum = digest.hexdigest()
            if expected and checksum != expected:
                raise FinplanError.precondition("downloaded checkpoint file failed its source checksum", reason="qwen_weight_checksum_mismatch")
            if name == "LICENSE" and "Apache" not in path.read_text()[:2000]:
                raise FinplanError.precondition("checkpoint license is not the discovered Apache license", reason="qwen_license_mismatch")
            s3.upload_file(str(path), bucket, PREFIX + name, ExtraArgs={"Metadata": {"sha256": checksum}})
            rows.append((name, checksum))
    names = {n for n, _ in rows}
    if not {"config.json", "tokenizer_config.json", "LICENSE"}.issubset(names) or not any(n.endswith(".safetensors") for n in names):
        raise FinplanError.precondition("checkpoint download is incomplete", reason="qwen_weight_manifest_incomplete")
    manifest = "".join(checksum + "  " + name + "\n" for name, checksum in sorted(rows)).encode()
    record = {"model_id": MODEL_ID, "revision": REVISION, "license": "Apache-2.0", "manifest_checksum": sha256_checksum(manifest), "file_count": len(rows), "total_bytes": total,
        "storage_lifecycle": "scratch prefix expires after configured retention; restage if expired"}
    s3.put_object(Bucket=bucket, Key=PREFIX + "MANIFEST.sha256", Body=manifest, ContentType="text/plain")
    s3.put_object(Bucket=bucket, Key=PREFIX + "STAGED.json", Body=canonical_json_bytes(record), ContentType="application/json")
    return record
