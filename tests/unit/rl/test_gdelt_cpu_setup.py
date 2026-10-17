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
    model['hyperparameters']=protocol['rl']['ppo'];model['seeds']=[{'seed':seed} for seed in protocol['rl']['seeds']]
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


def test_actual_sb3_training_result_exports_against_the_realized_source_manifest(monkeypatch):
    """Five32-step learners on synthetic sources exercise real serialization and export."""
    import copy,json,time
    from pathlib import Path
    from datetime import date,timedelta
    from finplan_model.core.artifacts import InMemoryArtifactStore
    from finplan_model.jobs.entrypoint import run_job
    from finplan_model.jobs.policy_export import freeze_policy_source,run_prepare_policy
    from finplan_model.jobs.handlers import JobInputs
    from finplan_model.jobs.market_loader import load_market
    from finplan_model.core.context import RunContext
    from finplan_model.rl import gdelt_setup
    from finplan_model.rl.gdelt_intraday import freeze_from_api,FEATURE_VERSION
    from tests.unit.rl.test_gdelt_intraday import archive,wrap
    from tests.unit.rl.support import tiny_protocol
    from tests.unit.selection.test_model_selection_job import _harness,selection_request

    started=time.perf_counter();protocol=tiny_protocol();protocol['rl'].update(algorithms=['ppo'],seeds=list(range(5)),policy_selection='ensemble',decision_timing='next_session_morning',development_evaluation=True)
    protocol['rl']['env'].update(features=['market_summary','current_weights','portfolio_drawdown','gdelt_intraday_organization_tone_v1'],step_sessions=1,decision_frequency='daily',version='finplan-rl-env/2')
    h,platform=_harness(protocol=protocol);store=InMemoryArtifactStore();h.service.d.artifacts=store
    real_freeze=freeze_from_api
    class Client:
     def __init__(self,required):self.required=set(required)
     def get_gdelt_intraday_metadata(self,symbols,first,last,cutoff,*,availability_mode):
      a,b=date.fromisoformat(first),date.fromisoformat(last);days=[str(a+timedelta(days=k)) for k in range((b-a).days+1)];docs=[]
      for day in days:
       if day not in self.required:continue
       d=archive(day,captured=3 if day=='2025-05-30' else 4)['document'];base=d['features'][0]
       d['features']=[{**base,'instrument_id':iid,'article_count':3,'cluster_count':2,'tone':float((date.fromisoformat(day).toordinal()%7)-3)} for iid in symbols]
       d['aliases']={iid:['synthetic '+iid] for iid in symbols};docs.append(wrap(d))
      missing=sorted(set(days)-self.required)
      return {'contract_version':'1.11.0','feature_version':FEATURE_VERSION,'symbols':symbols,'decision_at':cutoff,'availability_mode':availability_mode,'coverage':{'start_date':first,'end_date':last,'expected_days':len(days),'captured_days':len(docs),'complete':not missing,'missing_days':missing},'documents':docs}
    def freeze(client,symbols,first,last,cutoff,**kwargs):
     return real_freeze(Client(kwargs['required_days']),symbols,first,last,cutoff,**kwargs)
    monkeypatch.setattr(gdelt_setup,'freeze_from_api',freeze)
    rid=h.start(**selection_request());spec=h.run_io.get_spec(rid,None)
    code,result=run_job('model_selection',rid,run_io=h.run_io,platform=platform,artifacts=store,environment='beta',clock=h.clock)
    assert code==0,result
    summary=result['payload']['model_selection'];assert len(summary['policies'])==5
    assert summary['source_setup']['status']=='complete' and summary['protocol_id']==summary['source_setup']['realized_protocol_id']
    frozen=freeze_policy_source(result,spec,rid,artifacts=store)
    market,snapshot=load_market(platform,spec['input_snapshot_id']);export_spec={**spec,'job_type':'prepare_policy','policy_source':frozen}
    export=run_prepare_policy(JobInputs(ctx=RunContext(environment='local',clock=h.clock,synthetic=True),spec=export_spec,market=market,snapshot=snapshot,artifacts=store,platform=platform))
    assert export['completion_status']=='succeeded' and export['payload']['policy_export']['member_count']==5
    bundle=store.get_json(export['artifacts'][0]);assert bundle['training_gdelt_corpus_checksum']==summary['source_setup']['corpus']['checksum']
