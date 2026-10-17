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
