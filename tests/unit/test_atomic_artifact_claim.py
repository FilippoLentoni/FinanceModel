"""Paid invocation guards require genuinely conditional creates, not HEAD then PUT."""
import concurrent.futures
import json
import pytest
from finplan_model.core.artifacts import InMemoryArtifactStore, LocalArtifactStore, S3ArtifactStore


@pytest.mark.parametrize('kind',['memory','local','s3'])
def test_single_winner_claim_immutable_resolution(kind,tmp_path):
    if kind=='memory':store=InMemoryArtifactStore()
    elif kind=='local':store=LocalArtifactStore(tmp_path)
    else:
        import boto3
        from moto import mock_aws
        with mock_aws():
            s3=boto3.client('s3',region_name='us-east-2')
            s3.create_bucket(Bucket='example-artifact-claims',CreateBucketConfiguration={'LocationConstraint':'us-east-2'})
            store=S3ArtifactStore(s3,'example-artifact-claims')
            return _prove(store)
    _prove(store)


def _prove(store):
    aid='controlled_permanent_claim'
    def claim(n):
        body=json.dumps({'attempt':n}).encode()
        return n,store.create_once(body,kind='run_artifact',artifact_id=aid)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(claim,range(16)))
    winners=[n for n,won in results if won]
    assert len(winners)==1
    ref=store.resolve(kind='run_artifact',artifact_id=aid)
    assert store.get_json(ref)=={'attempt':winners[0]}
    assert not store.create_once(b'{"attempt":999}',kind='run_artifact',artifact_id=aid)
    assert store.get_json(ref)=={'attempt':winners[0]}
