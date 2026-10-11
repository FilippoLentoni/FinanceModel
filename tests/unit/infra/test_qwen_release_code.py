"""Published code remains usable after scratch expiry and matches least-privilege IAM."""

from finplan_contracts import boundaries
from finplan_contracts.iam import Request, evaluate

from infra.stacks import naming, policies
from scripts.qwen_code import BOOTSTRAP, publish

ENV = "beta"
ACCOUNT = "<account-id>"
CONTEXT = {"partition": "aws", "region": "us-east-2", "account": ACCOUNT}
RELEASE = "rel_01KDVDNAZ83BAMMYCEGWF33DPM"


class Storage:
    def __init__(self):
        self.objects = []

    def put_object(self, **request):
        self.objects.append(request)


def test_published_code_is_retained_and_the_stage_and_job_can_access_it(assembly):
    storage = Storage()
    bucket = naming.bucket_name(ENV, naming.RESEARCH_BUCKET, ACCOUNT)
    result = publish(storage, bucket, RELEASE, b"synthetic-code-fixture")
    assert result["code_prefix"] == "releases/qwen-code/" + RELEASE + "/"
    assert {o["Key"] for o in storage.objects} == {result["code_prefix"] + "code.zip", result["code_prefix"] + "bootstrap.py"}
    assert next(o["Body"] for o in storage.objects if o["Key"].endswith("bootstrap.py")) == BOOTSTRAP.encode()
    research = next(b for b in assembly.resources("finplan-beta-financemodel-storage", "AWS::S3::Bucket").values() if any(t.get("Value") == "research-workspace-bucket" for t in b["Properties"]["Tags"]))
    expirations = [r for r in research["Properties"]["LifecycleConfiguration"]["Rules"] if "ExpirationInDays" in r]
    assert expirations and all("Prefix" in r for r in expirations)
    stage = {"Version": "2012-10-17", "Statement": policies.stage_role_statements(ENV, "arn:aws:s3:::example-pipeline-store", **CONTEXT)}
    job = policies.job_execution_policy(ENV, **CONTEXT)
    for obj in storage.objects:
        assert all(not obj["Key"].startswith(rule["Prefix"]) for rule in expirations)
        resource = "arn:aws:s3:::" + bucket + "/" + obj["Key"]
        assert evaluate(Request("s3:PutObject", resource, {}), [stage], boundaries.env_permission_boundary(ENV, **CONTEXT)).allowed
        assert evaluate(Request("s3:GetObject", resource, {}), [job], boundaries.research_permission_boundary(ENV, **CONTEXT)).allowed
        assert not evaluate(Request("s3:PutObject", resource, {}), [job], boundaries.research_permission_boundary(ENV, **CONTEXT)).allowed
    assert evaluate(Request("s3:ListBucket", "arn:aws:s3:::" + bucket, {"s3:prefix": result["code_prefix"]}), [job], boundaries.research_permission_boundary(ENV, **CONTEXT)).allowed
