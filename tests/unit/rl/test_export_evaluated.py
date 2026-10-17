"""Controlled tensor fixtures; no training, network calls or research performance."""
import copy
import io
import zipfile

import numpy as np
import pytest
import torch

from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.rl.export_evaluated import export_evaluated_members
from finplan_model.rl.spec import EnvSpec, build_observation


def fixture():
    store = InMemoryArtifactStore()
    env = EnvSpec.from_dict({'window':20,'features':['log_return_window','current_weights']}).document()
    env.pop('reward_formula')
    n = len(build_observation(EnvSpec.from_dict(env), np.ones((21,2)), np.zeros(2), 1.))
    tensors = {'actor.latent_pi.0.weight':torch.zeros((64,n)), 'actor.latent_pi.0.bias':torch.zeros(64),
               'actor.latent_pi.2.weight':torch.zeros((64,64)), 'actor.latent_pi.2.bias':torch.zeros(64),
               'actor.mu.weight':torch.zeros((3,64)), 'actor.mu.bias':torch.zeros(3)}
    buffer=io.BytesIO();torch.save(tensors,buffer)
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('policy.pth',buffer.getvalue())
        z.writestr('data','This untrusted SB3 cloudpickle JSON is never deserialized')
    ref=store.put(archive.getvalue(),kind='rl_policy',content_type='application/zip').to_dict()
    protocol={'splits':{'train':{'start':'2024-01-01','end':'2024-12-31'},'validation':{'start':'2025-01-01','end':'2025-06-30'},'test':{'start':'2025-07-01','end':'2025-12-31'}},'rl':{'seeds':[0,1],'sac':{'net_arch':[64,64],'total_timesteps':8000}}}
    rows=[{'algorithm':'sac','artifact_id':ref['artifact_id'],'checksum':ref['checksum'],'configuration_id':'cfg_exampleconfiguration','reward':{'risk_penalty':.5},'seed':seed} for seed in (0,1)]
    selection={'chosen_hyperparameters':{'sac':{'configuration_id':rows[0]['configuration_id'],'seed':0}}}
    diagnostics=store.put_json({'policies':rows,'protocol':protocol,'selection':selection},kind='run_artifact').to_dict()
    result={'run_id':'run_examplecontrolled','completion_status':'succeeded','input_snapshot_id':'snap_examplecontrolled','dataset_checksum':'sha256:'+'c'*64,
            'artifacts':[ref,diagnostics], 'payload':{'model_selection':{'policies':rows,'selection':selection,'rl':{'sac':{'environment':env,'grid':[{'configuration_id':rows[0]['configuration_id'],'reward':{'risk_penalty':.5},'checkpoints':[{'seed':0,'best_step':5000},{'seed':1,'best_step':0}]}]}}}}}
    spec={'run_id':result['run_id'],'input_snapshot_id':result['input_snapshot_id'],'purpose':'research','environment':'beta','synthetic':False,'selection_protocol':protocol,'universe':['BBB','AAA'],'simulation':{'constraints':{'max_weight':1},'constraint_policy':'project'}}
    return result,spec,store


def test_legacy_single_seed_and_other_evaluated_members_export_independently():
    result,spec,store=fixture()
    bundles=export_evaluated_members(result,spec,store)
    assert len(bundles)==2
    assert [b['members'][0]['seed'] for b in bundles]==[0,1]
    assert [b['evaluated_variant']['selected_in_original_run'] for b in bundles]==[True,False]
    assert bundles[0]['instruments']==['AAA','BBB']
    assert all(len(b['members'])==1 and b['strategy_id']=='sac' for b in bundles)
    assert bundles[1]['evaluated_variant']['evaluated_checkpoint']['best_step']==0
    assert all(b['evaluated_variant']['training_performed'] is False for b in bundles)
    spec['simulation']['constraints']['max_weight']=.1
    assert bundles[0]['constraints']['max_weight']==1


def test_new_explicit_diagnostic_id_supported():
    result,spec,store=fixture()
    result['payload']['model_selection']['full_diagnostics_artifact_id']=result['artifacts'][1]['artifact_id']
    assert len(export_evaluated_members(result,spec,store))==2


@pytest.mark.parametrize('defect',['run','gamma','synthetic','failed','source_row','policy_checksum','bad_second_member','chronology','universe','validation_grid','news'])
def test_invalid_provenance_rejects_before_any_tensor_export(defect,monkeypatch):
    result,spec,store=fixture()
    summary=result['payload']['model_selection']
    if defect=='run':spec['run_id']='run_othercontrolled'
    if defect=='gamma':spec['environment']='gamma'
    if defect=='synthetic':spec['synthetic']=True
    if defect=='failed':result['completion_status']='failed'
    if defect=='source_row':summary['policies'][0]['seed']=99
    if defect=='policy_checksum':result['artifacts'][0]['checksum']='sha256:'+'d'*64
    if defect=='bad_second_member':summary['policies'][1]['seed']=99
    if defect=='chronology':spec['selection_protocol']['splits']['train']['end']='2025-02-01'
    if defect=='universe':spec['universe']=['AAA','AAA']
    if defect=='validation_grid':summary['rl']['sac']['grid'][0]['checkpoints'].pop()
    if defect=='news':summary['rl']['sac']['environment']['features'].append('etf_news_metadata_v1')
    monkeypatch.setattr('finplan_model.rl.export_evaluated.export_actor',lambda *a:pytest.fail('must reject before tensors'))
    with pytest.raises(Exception):export_evaluated_members(result,spec,store)


def test_corrupt_original_artifact_cannot_be_exported(monkeypatch):
    result,spec,store=fixture()
    aid=result['artifacts'][0]['artifact_id'];ref,data=store.objects[aid]
    store.objects[aid]=(ref,data+b'corrupt')
    monkeypatch.setattr('finplan_model.rl.export_evaluated.export_actor',lambda *a:pytest.fail('checksum must reject before tensors'))
    with pytest.raises(Exception,match='checksum'):export_evaluated_members(result,spec,store)
