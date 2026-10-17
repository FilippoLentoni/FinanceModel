"""Original policy weights are referenced once, exact prepared inputs remain separate."""
import copy
import pytest
from finplan_model.core.artifacts import InMemoryArtifactStore, canonical_json_bytes, sha256_checksum
from finplan_model.forward.policy_artifacts import freeze_policy_references, restore_frozen_policies


def example():
    return {'configuration_id':'cfg_original','strategy_id':'ppo','instruments':['A','B'],
            'environment':{'features':['current_weights']},'actor':{'weights':[float(n) for n in range(3000)]}}


def setup():
    store=InMemoryArtifactStore();bundle=example();ref=store.put(canonical_json_bytes(bundle),kind='policy_inference').to_dict()
    refs=freeze_policy_references({'ppo':bundle,'hosted_langgraph':bundle},{sha256_checksum(canonical_json_bytes(bundle)):ref})
    return store,bundle,refs


def test_references_exclude_weights_and_restore_original_not_latest_selection():
    store,bundle,refs=setup();source={'frozen_policy_refs':refs};before=copy.deepcopy(source)
    newer={**bundle,'configuration_id':'cfg_new'};store.put(canonical_json_bytes(newer),kind='policy_inference')
    restored,failures=restore_frozen_policies(source,artifacts=store)
    assert not failures and restored=={'ppo':bundle,'hosted_langgraph':bundle}
    assert source==before and len(canonical_json_bytes(source))<2000
    assert 'actor' not in refs['ppo']
    restored['ppo']['actor']['weights'][0]=999
    assert restored['hosted_langgraph']==bundle


@pytest.mark.parametrize('field,value',[('model_checksum','sha256:'+'0'*64),('configuration_id','different'),('features',[])])
def test_model_identity_or_prepared_feature_specification_cannot_drift(field,value):
    store,bundle,refs=setup();refs['ppo'][field]=value
    with pytest.raises(ValueError):restore_frozen_policies({'frozen_policy_refs':refs},artifacts=store)


def test_one_missing_artifact_blocks_only_affected_approaches_and_keeps_original_reference():
    store,bundle,refs=setup();refs['ppo']['artifact']['artifact_id']='policy_inference_missing'
    restored,failures=restore_frozen_policies({'frozen_policy_refs':refs},artifacts=store,strict=False)
    assert restored=={'hosted_langgraph':bundle}
    assert failures['ppo']['expected']==refs['ppo']
    assert failures['ppo']['reason']=='original_policy_artifact_unavailable'


def test_corrupted_bytes_do_not_satisfy_original_checksum():
    store,bundle,refs=setup();aid=refs['ppo']['artifact']['artifact_id'];ref,_=store.objects[aid]
    store.objects[aid]=(ref,b'{}')
    with pytest.raises(Exception,match='checksum'):restore_frozen_policies({'frozen_policy_refs':refs},artifacts=store)


def test_old_archive_compatibility_does_not_load_new_aws_selection():
    bundle=example();source={'frozen_policies':{'ppo':bundle}}
    result,errors=restore_frozen_policies(source)
    assert result==source['frozen_policies'] and not errors
    result['ppo']['actor']['weights'][0]=999
    assert source['frozen_policies']['ppo']==bundle
    assert restore_frozen_policies({})==({}, {})


def test_foreign_owner_reference_rejected_before_artifact_fetch():
    store,bundle,refs=setup();refs['ppo']['artifact']['owner']='financialplanning'
    with pytest.raises(ValueError,match='owner'):restore_frozen_policies({'frozen_policy_refs':refs},artifacts=store)


@pytest.mark.parametrize('slot',['2026-10-18T05:30:00+05:45','2026-10-18T00:00:00.000001Z'])
def test_interim_alignment_uses_utc_including_microseconds_before_any_backend_call(slot,tmp_path):
    from finplan_model.forward.hourly import run_interim
    with pytest.raises(ValueError,match='half-hour aligned'):run_interim(slot,'alignment-test',tmp_path,backend=object())
