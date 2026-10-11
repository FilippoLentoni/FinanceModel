import io
import tarfile
from types import SimpleNamespace

import pytest

from finplan_model.benchmarks.offline import OfflineBatchIO
from finplan_model.benchmarks.jobs import swarm_mode_a
from finplan_model.core.artifacts import InMemoryArtifactStore, canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.jobs.runio import InMemoryRunIO
from tests.unit.control.support import Harness, job_result


def archive(files, *, manifest=None):
    files = dict(files)
    files["MANIFEST.json"] = canonical_json_bytes(manifest if manifest is not None else {k: sha256_checksum(v) for k, v in files.items()})
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as tar:
        for name in ("./", "./artifacts/", "./artifacts/run_artifact/"):
            directory = tarfile.TarInfo(name)
            directory.type = tarfile.DIRTYPE
            tar.addfile(directory)
        for name, value in files.items():
            member = tarfile.TarInfo("./" + name)
            member.size = len(value)
            tar.addfile(member, io.BytesIO(value))
    return data.getvalue()


def setup_output(files, manifest=None):
    h = Harness()
    _, response = h.submit()
    run = {**h.run(response["run_id"]), "job_name": "fm-beta-fixture"}
    result = job_result(run)
    result["artifacts"] = []
    files = {"result.json": canonical_json_bytes(result), **files}
    data = archive(files, manifest=manifest)
    s3 = SimpleNamespace(get_object=lambda **kwargs: {"Body": io.BytesIO(data)})
    run_io = InMemoryRunIO()
    io_backend = OfflineBatchIO(s3, "<research-bucket>", None, run_io, InMemoryArtifactStore())
    description = {"ModelArtifacts": {"S3ModelArtifacts": "s3://<research-bucket>/scratch/training-output/fm-beta-fixture/output/model.tar.gz"}}
    return run, result, io_backend, description


def test_verified_offline_archive_accepts_normal_tar_directories_and_writes_result_last():
    run, expected, backend, description = setup_output({})
    data = canonical_json_bytes({"synthetic": True, "messages": []})
    ref = InMemoryArtifactStore().put(data, kind="run_artifact", synthetic=True, domain="finance")
    expected["artifacts"] = [ref.to_dict()]
    output = archive({"result.json": canonical_json_bytes(expected), "artifacts/run_artifact/" + ref.artifact_id: data})
    backend.s3 = SimpleNamespace(get_object=lambda **kwargs: {"Body": io.BytesIO(output)})
    result = backend.import_result(run, description)
    assert result == expected and backend.run_io.get_result(run["run_id"]) == result
    assert backend.artifacts.get(ref) == data


@pytest.mark.parametrize("member_type", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE])
def test_offline_archive_rejects_links_and_devices(member_type):
    run, _, backend, description = setup_output({})
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as tar:
        member = tarfile.TarInfo("./artifacts/unsafe")
        member.type = member_type
        member.linkname = "../../outside"
        tar.addfile(member)
    backend.s3 = SimpleNamespace(get_object=lambda **kwargs: {"Body": io.BytesIO(output.getvalue())})
    with pytest.raises(FinplanError) as caught:
        backend.import_result(run, description)
    assert caught.value.details["reason"] == "offline_output_invalid"
    assert backend.run_io.get_result(run["run_id"]) is None


@pytest.mark.parametrize("case", ["manifest_list", "missing_artifact", "unsafe_path", "wrong_identity"])
def test_malformed_offline_output_is_a_safe_contract_error(case):
    run, result, backend, description = setup_output({})
    files = {}
    manifest = None
    if case == "manifest_list":
        manifest = []
    elif case == "missing_artifact":
        result["artifacts"] = job_result(run)["artifacts"]
    elif case == "unsafe_path":
        files["../outside"] = b"refused"
    else:
        result["configuration_id"] = "cfg_" + "d" * 32
    data = archive({"result.json": canonical_json_bytes(result), **files}, manifest=manifest)
    backend.s3 = SimpleNamespace(get_object=lambda **kwargs: {"Body": io.BytesIO(data)})
    with pytest.raises(FinplanError):
        backend.import_result(run, description)
    assert backend.run_io.get_result(run["run_id"]) is None


def test_swarm_backend_teardown_on_evaluation_failure():
    closed = []
    backend = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(AttributeError):
        swarm_mode_a(SimpleNamespace(), backend=backend)
    assert closed == [True]
