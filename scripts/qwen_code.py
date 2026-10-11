"""Small immutable code channel for the existing AWS vLLM DLC; no GPU image build.

The pipeline's CPython 3.12 x86 environment supplies the locked validation closure, including
rpds' native wheel. NumPy/SciPy/Torch stay those of the verified vLLM image, avoiding GPU/CPU
torch replacement. Import readiness fails closed inside the run if the DLC lacks prerequisites.
"""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path

PACKAGES = ("finplan_contracts", "rfc8785", "ulid", "jsonschema", "jsonschema_specifications", "referencing", "rpds", "attrs", "attr", "typing_extensions")
RELEASE_PREFIX = "releases/qwen-code/"
BOOTSTRAP = '''import hashlib, os, pathlib, runpy, sys, zipfile
p = pathlib.Path('/opt/ml/input/data/code/code.zip')
expected = os.environ['FINPLAN_OFFLINE_CODE_SHA256']
assert 'sha256:' + hashlib.sha256(p.read_bytes()).hexdigest() == expected, 'offline code checksum mismatch'
target = pathlib.Path('/opt/ml/code')
with zipfile.ZipFile(p) as archive:
    for entry in archive.infolist():
        name = pathlib.PurePosixPath(entry.filename)
        assert not name.is_absolute() and '..' not in name.parts, 'unsafe code member'
    archive.extractall(target)
sys.path.insert(0, str(target))
runpy.run_module('finplan_model.benchmarks.offline', run_name='__main__')
'''


def build(root: Path, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((root / "src/finplan_model").rglob("*.py")):
            archive.write(path, path.relative_to(root / "src").as_posix())
        for package in PACKAGES:
            spec = importlib.util.find_spec(package)
            if spec is None or spec.origin is None:
                raise RuntimeError("Qwen code channel dependency missing: " + package)
            source = Path(spec.origin)
            if source.name == "__init__.py":
                for path in sorted(source.parent.rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                        archive.write(path, (Path(package) / path.relative_to(source.parent)).as_posix())
            else:
                archive.write(source, source.name)
    return output


def publish(s3, bucket, release_id, data):
    from finplan_model.core.artifacts import sha256_checksum
    # Referenced release code must outlive the temporary scratch retention window.
    prefix = RELEASE_PREFIX + release_id + "/"
    s3.put_object(Bucket=bucket, Key=prefix + "code.zip", Body=data, ContentType="application/zip")
    s3.put_object(Bucket=bucket, Key=prefix + "bootstrap.py", Body=BOOTSTRAP.encode(), ContentType="text/x-python")
    return {"code_prefix": prefix, "code_checksum": sha256_checksum(data)}
