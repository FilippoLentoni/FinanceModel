"""A succeeded recursive PPO experiment uses the same verified export boundary."""
from types import SimpleNamespace
import pytest
from tests.unit.control.support import Harness,request
from tests.unit.rl.test_policy_export import source


@pytest.mark.parametrize('kind',['model_selection','recursive_evaluate'])
def test_dry_run_exports_complete_evaluated_recursive_or_selection_ensemble(kind):
    h=Harness(auto_approve=1.);result,spec=source();rid=h.ids.run_id()
    h.service._get=lambda run_id:{'state':'succeeded','job_type':kind}
    h.deps.run_io=SimpleNamespace(get_result=lambda run_id:result,get_spec=lambda run_id,checksum:spec)
    cfg=request()['configuration'];cfg['payload'].update(strategy='prepare_policy',policy_source_run_id=rid,policy_strategy_id='ppo')
    status,doc=h.submit(job_type='prepare_policy',configuration=cfg,dry_run=True)
    assert status==200 and doc['dry_run'] is True and doc['run_id'] is None
    assert not h.sagemaker.calls


def test_manual_daily_profile_uses_existing_long_job_cap_and_server_frozen_history():
    from finplan_model.selection.recursive_profiles import DAILY_HISTORY_PROFILE
    h=Harness(auto_approve=1.);cfg=request()['configuration']
    cfg['payload'].update(strategy='model_selection',objective=DAILY_HISTORY_PROFILE,rebalance_frequency='daily')
    _,submitted=h.submit(job_type='model_selection',configuration=cfg,max_runtime_seconds=3000)
    stored=h.run(submitted['run_id']);protocol=stored['selection_protocol']
    assert stored['max_runtime_seconds']==3000 and protocol['splits']['train']['start']=='2013-01-01'
    assert protocol['rl']['seeds']==[0,1,2,3,4] and protocol['rl']['ppo']['total_timesteps']==30000
    assert protocol['rl']['development_evaluation'] is True


def test_weekly_controller_cannot_use_the_manual_long_profile_job():
    from finplan_model.control.auth import Principal
    from finplan_model.core.errors import FinplanError
    from finplan_model.selection.recursive_profiles import DAILY_HISTORY_PROFILE
    from tests.unit.control.support import arn
    h=Harness(auto_approve=1.);body=request(job_type='model_selection',max_runtime_seconds=3000,idempotency_key='classical-weekly-long-attempt')
    body['configuration']['payload'].update(strategy='model_selection',objective=DAILY_HISTORY_PROFILE,rebalance_frequency='daily')
    with pytest.raises(FinplanError) as error:
        h.service.submit_job(Principal.from_arn(arn('finplan-beta-financemodel-job-api-handler-research-role')),body,correlation_id='corr-manual-profile')
    assert error.value.code=='FORBIDDEN' and not h.store.runs and not h.sagemaker.calls

def test_intraday_readiness_uses_verified_sessions_inside_frozen_splits(monkeypatch):
    from datetime import date
    import finplan_model.control.service as service_module
    import finplan_model.jobs.market_loader as loader
    import finplan_model.rl.gdelt_intraday as gdelt
    from finplan_model.selection.protocol import DEFAULT_PROTOCOL
    from finplan_model.selection.recursive_profiles import profile_protocol,GDELT_INTRADAY_PROFILE
    h=Harness(auto_approve=1.);seen={};h.service.d.platform=SimpleNamespace()
    monkeypatch.setattr(service_module,'SnapshotReader',lambda client:SimpleNamespace(resolve=lambda sid:SimpleNamespace(record={'coverage':{'end':'2026-10-16'}})))
    actual=[date(2024,1,2),date(2025,5,30),date(2025,6,2),date(2025,7,2),date(2025,12,31),date(2026,1,2),date(2026,10,16)]
    monkeypatch.setattr(loader,'load_market',lambda client,sid:(SimpleNamespace(sessions=actual),None))
    def freeze(client,symbols,start,end,cutoff,**kwargs):
        seen.update(kwargs)
        return {'feature_version':gdelt.FEATURE_VERSION,'availability_mode':'historical_source_proxy'},{'checksum':'sha256:'+'a'*64,'source_days':len(kwargs['required_days']),'required_session_days':kwargs['required_days']}
    monkeypatch.setattr(gdelt,'freeze_from_api',freeze)
    h.service.d.artifacts=SimpleNamespace(put_json=lambda *a,**k:SimpleNamespace(to_dict=lambda:{'checksum':'sha256:'+'a'*64}))
    protocol=profile_protocol(DEFAULT_PROTOCOL,GDELT_INTRADAY_PROFILE)
    body=request();body['configuration']['payload']['universe']=['AAPL','VOO']
    h.service._freeze_gdelt_corpus(protocol,SimpleNamespace(body=body))
    assert seen['required_days']==['2024-01-02','2025-05-30','2025-07-02','2025-12-31','2026-01-02','2026-10-16']
    assert protocol['rl']['gdelt_corpus']['required_session_days']==seen['required_days']


def test_matched_price_control_changes_only_news_state_not_periods_seeds_or_objectives():
    import copy
    from finplan_model.selection.recursive_profiles import profile_protocol,GDELT_INTRADAY_PROFILE,GDELT_PRICE_CONTROL_PROFILE
    from finplan_model.selection.protocol import DEFAULT_PROTOCOL,validate_protocol
    from finplan_model.rl.spec import EnvSpec
    news=profile_protocol(DEFAULT_PROTOCOL,GDELT_INTRADAY_PROFILE)
    price=profile_protocol(DEFAULT_PROTOCOL,GDELT_PRICE_CONTROL_PROFILE)
    expected=copy.deepcopy(news);expected['rl']['env']['features'].remove('gdelt_intraday_organization_tone_v1')
    assert price==expected
    assert EnvSpec.from_dict(news['rl']['env']).observation_size(8)==98
    assert EnvSpec.from_dict(price['rl']['env']).observation_size(8)==58
    h=Harness(auto_approve=1.);cfg=request()['configuration']
    cfg['payload'].update(strategy='model_selection',objective=GDELT_PRICE_CONTROL_PROFILE,rebalance_frequency='daily')
    _,submitted=h.submit(job_type='model_selection',configuration=cfg,max_runtime_seconds=3000)
    frozen=h.run(submitted['run_id'])['selection_protocol']
    assert frozen==validate_protocol(price) and 'gdelt_corpus' not in frozen['rl']
