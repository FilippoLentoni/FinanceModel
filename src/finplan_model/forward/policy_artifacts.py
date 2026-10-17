"""Frozen model identity references, separate from captured inference input content.

Model weights already live in immutable FinanceModel artifacts. Hourly source
bundles retain their original references; recovery loads those exact bytes and
never reselects a newer model or regenerates the captured observation vector.
"""
from __future__ import annotations
import copy
import json
from finplan_model.core.artifacts import ArtifactRef, S3ArtifactStore, canonical_json_bytes, sha256_checksum


def freeze_policy_references(policies, references):
    result = {}
    for approach_id, bundle in policies.items():
        checksum = sha256_checksum(canonical_json_bytes(bundle))
        ref = references.get(checksum)
        if ref is None:
            raise ValueError('original immutable policy reference missing: ' + approach_id)
        ref = ArtifactRef.from_dict(ref)
        if ref.owner != 'financemodel' or ref.kind != 'policy_inference':
            raise ValueError('policy reference owner/kind mismatch')
        result[approach_id] = {'artifact': ref.to_dict(), 'model_checksum': checksum,
                              'configuration_id': bundle['configuration_id'],
                              'strategy_id': bundle['strategy_id'],
                              'instruments': copy.deepcopy(bundle['instruments']),
                              'features': copy.deepcopy(bundle.get('environment', {}).get('features', []))}
    return result


def restore_frozen_policies(source, *, artifacts=None, strict=True):
    if 'frozen_policy_refs' not in source:
        return copy.deepcopy(source.get('frozen_policies', {})), {}
    refs = source['frozen_policy_refs']
    if refs and artifacts is None:
        import boto3
        from finplan_model.core.aws_clients import s3_client
        session = boto3.Session(region_name='us-east-2')
        bucket = session.client('ssm').get_parameter(Name='/finplan/beta/financemodel/config/research-storage-ref')['Parameter']['Value']
        artifacts = S3ArtifactStore(s3_client('us-east-2', session=session), bucket)
    restored, failures, cache = {}, {}, {}
    for approach_id, frozen in refs.items():
        try:
            ref = ArtifactRef.from_dict(frozen['artifact'])
            if ref.owner != 'financemodel' or ref.kind != 'policy_inference':
                raise ValueError('policy reference owner/kind mismatch')
            key = (ref.artifact_id, ref.checksum)
            if key not in cache:
                cache[key] = json.loads(artifacts.get(ref))
            bundle = cache[key]
            if sha256_checksum(canonical_json_bytes(bundle)) != frozen['model_checksum']:
                raise ValueError('frozen model content checksum mismatch')
            for name in ('configuration_id', 'strategy_id', 'instruments'):
                if bundle[name] != frozen[name]:
                    raise ValueError('frozen model identity mismatch: ' + name)
            if bundle.get('environment', {}).get('features', []) != frozen['features']:
                raise ValueError('frozen policy feature specification mismatch')
            restored[approach_id] = copy.deepcopy(bundle)
        except Exception as exc:
            if strict:
                raise
            failures[approach_id] = {'reason': 'original_policy_artifact_unavailable',
                                     'error': type(exc).__name__ + ': ' + str(exc),
                                     'expected': copy.deepcopy(frozen)}
    return restored, failures
