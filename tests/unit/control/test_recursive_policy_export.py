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
