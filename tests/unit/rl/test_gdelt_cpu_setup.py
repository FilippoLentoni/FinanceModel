"""Deferred setup preserves acceptance identity and fails before learner execution."""
import copy
from datetime import date
from types import SimpleNamespace
import pytest
from finplan_model.core.artifacts import InMemoryArtifactStore,canonical_json_bytes,sha256_checksum
from finplan_model.core.clock import FrozenClock
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError
from finplan_model.rl import gdelt_setup
from finplan_model.rl.gdelt_intraday import corpus,FEATURE_VERSION
from finplan_model.selection.protocol import DEFAULT_PROTOCOL
from finplan_model.selection.recursive_profiles import profile_protocol,GDELT_INTRADAY_PROFILE
from tests.unit.rl.test_gdelt_intraday import archive
from tests.unit.rl.test_policy_export import source

def inputs(monkeypatch):
    protocol=profile_protocol(DEFAULT_PROTOCOL,GDELT_INTRADAY_PROFILE)
    checksum='sha256:'+'a'*64
    snapshot=SimpleNamespace(manifest_checksum=checksum,record={'coverage':{'end':'2026-10-09'}})
    request=gdelt_setup.source_request(snapshot,protocol,'snap_synthetic',['AAPL','VOO'],'2026-10-12T13:00:00Z')
    protocol['rl']['gdelt_corpus_request']=request
    sessions=[date(2024,1,2),date(2025,5,30),date(2025,6,2),date(2025,7,2),date(2026,10,9)]
    inp=SimpleNamespace(spec={'input_snapshot_id':'snap_synthetic','universe':['AAPL','VOO'],'selection_protocol':copy.deepcopy(protocol)},
                        snapshot=SimpleNamespace(snapshot=snapshot),universe=['AAPL','VOO'],market=SimpleNamespace(sessions=sessions),
                        artifacts=InMemoryArtifactStore(),platform=object(),ctx=RunContext(environment='local',clock=FrozenClock('2026-10-17T10:00:00Z')))
    def freeze(client,symbols,start,end,cutoff,**kwargs):
        assert kwargs['max_workers']==2 and kwargs['required_days']==['2024-01-02','2025-05-30','2025-07-02','2026-10-09']
        dataset=corpus([archive(day,captured=3 if day=='2025-05-30' else 4) for day in kwargs['required_days']],symbols)
        return dataset,{'checksum':sha256_checksum(canonical_json_bytes(dataset))}
    monkeypatch.setattr(gdelt_setup,'freeze_from_api',freeze)
    return inp,protocol

def completed(monkeypatch):
    inp,protocol=inputs(monkeypatch);accepted=copy.deepcopy(protocol)
    setup=gdelt_setup.prepare(inp,protocol,deadline=100,clock=lambda:0)
    result,spec=source();spec.update(inp.spec);spec['selection_protocol']=accepted
    spec['simulation']={'constraints':{'max_weight':.6}}
    result.update(artifacts_complete=True,dataset_checksum=accepted['rl']['gdelt_corpus_request']['query']['snapshot_manifest_checksum'])
    result['artifacts'].extend([setup['corpus']['artifact'],setup['source_manifest']])
    result['payload']['model_selection']['source_setup']=setup
    result['payload']['model_selection']['protocol_id']=setup['realized_protocol_id']
    from finplan_model.rl.spec import EnvSpec
    model=result['payload']['model_selection']['rl']['ppo']
    model['environment']=EnvSpec.from_dict(protocol['rl']['env']).document()
    model['hyperparameters']=protocol['rl']['ppo'];model['seeds']=protocol['rl']['seeds']
    return inp,protocol,result,spec

def test_setup_persists_realized_identity_without_mutating_accepted_spec(monkeypatch):
    inp,protocol,result,spec=completed(monkeypatch)
    assert 'gdelt_corpus' not in spec['selection_protocol']['rl']
    metadata,identity=gdelt_setup.verified_realization(result,spec,inp.artifacts)
    assert metadata['checksum']==protocol['rl']['gdelt_corpus']['checksum']
    from finplan_model.jobs.policy_export import freeze_policy_source
    frozen=freeze_policy_source(result,spec,'run_synthetic',artifacts=inp.artifacts)
    assert frozen['training_gdelt_corpus_checksum']==metadata['checksum']
    assert frozen['training_source_query_checksum']==identity['training_source_query_checksum']
    assert frozen['gdelt_feature_protocol']==FEATURE_VERSION

@pytest.mark.parametrize('mutation',['query','manifest_reference','corpus_reference','realized_protocol','snapshot','evaluation_protocol','environment'])
def test_export_rejects_forged_or_mismatched_realization(monkeypatch,mutation):
    inp,protocol,result,spec=completed(monkeypatch);setup=result['payload']['model_selection']['source_setup']
    if mutation=='query':spec['selection_protocol']['rl']['gdelt_corpus_request']['query']['decision_at']='2026-10-13T13:00:00Z'
    elif mutation=='manifest_reference':setup['source_manifest']={**setup['source_manifest'],'checksum':'sha256:'+'b'*64}
    elif mutation=='corpus_reference':setup['corpus']['artifact']={**setup['corpus']['artifact'],'checksum':'sha256:'+'b'*64}
    elif mutation=='realized_protocol':setup['realized_protocol_checksum']='sha256:'+'b'*64
    elif mutation=='snapshot':result['dataset_checksum']='sha256:'+'b'*64
    elif mutation=='evaluation_protocol':result['payload']['model_selection']['protocol_id']='cfg_wrong'
    else:result['payload']['model_selection']['rl']['ppo']['environment']['features']=['current_weights']
    from finplan_model.jobs.policy_export import freeze_policy_source
    with pytest.raises(FinplanError):freeze_policy_source(result,spec,'run_synthetic',artifacts=inp.artifacts)

@pytest.mark.parametrize('failure',['source','late_central','late_retrieval','deadline','query'])
def test_setup_failure_occurs_before_any_learner(monkeypatch,failure):
    inp,protocol=inputs(monkeypatch);real_freeze=gdelt_setup.freeze_from_api
    if failure=='query':protocol['rl']['gdelt_corpus_request']['query']['symbols']=['VOO']
    def freeze(*a,**kw):
        if failure=='source':raise FinplanError.precondition('source coverage missing',reason='gdelt_coverage_missing')
        ds,evidence=real_freeze(*a,**kw)
        if failure.startswith('late_'):
            key='central_available_at' if failure=='late_central' else 'retrieved_at'
            ds['documents'][0]['document']['lineage'][key]='2026-10-13T13:00:00Z'
        return ds,evidence
    monkeypatch.setattr(gdelt_setup,'freeze_from_api',freeze)
    called=[]
    import finplan_model.selection.job as job
    monkeypatch.setattr(job,'select_models',lambda *a,**kw:called.append(True))
    inp.spec.update(purpose='research',max_runtime_seconds=180 if failure=='deadline' else 3000,selection_protocol=protocol)
    monkeypatch.setattr(job,'_bias',lambda inp:None,raising=False)
    # _bias is imported from handlers; the source fixture has no universe payload.
    import finplan_model.jobs.handlers as handlers
    monkeypatch.setattr(handlers,'_bias',lambda inp:None)
    with pytest.raises(FinplanError):job.run_model_selection(inp)
    assert called==[] and not inp.artifacts.objects
