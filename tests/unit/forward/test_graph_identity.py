"""Hosted graph dependencies are frozen without requiring the Agent package in Model builds."""
import copy
import sys
from types import ModuleType

import pytest

from finplan_model.forward.hourly import _graph_adapter

CHECKSUM='sha256:'+'a'*64


def source(strategy='ppo'):
    return {'input_snapshot_id':'controlled_snapshot','investable_universe':['A','B'],
        'observations':[{'instrument_id':'A','session_date':'2026-07-10','close':100.}],
        'served_advisory_policy':{'strategy_id':strategy,'instruments':['A','B'],'configuration_id':'controlled_configuration'},
        'served_advisory_policy_ref':{'artifact_id':'controlled_policy','kind':'policy_inference','checksum':CHECKSUM}}


def holdings():
    return {'weights':[{'instrument_id':'A','weight':0.},{'instrument_id':'B','weight':0.}],
        'cash_weight':1.,'portfolio_value':100000.,'high_watermark':100000.}


def response(strategy='ppo',checksum=CHECKSUM,with_capture=True):
    prepared={'schema_version':'controlled_served_input_fixture','prepared':{'observation_vector':[0.,1.],
              'feature_names':['controlled_price_feature','controlled_cash_feature']}}
    rec={'strategy':strategy,'policy_artifact_checksum':checksum,'diagnostics':{'actual_model_inputs':
        [{'payload':prepared,'artifact':{'artifact_id':'controlled_input','checksum':'sha256:'+'b'*64}}] if with_capture else []}}
    return {'status':'recommended','target_weights':[{'instrument_id':'A','weight':.5},{'instrument_id':'B','weight':.5}],
        'cash_weight':0.,'answer':{'recommendation':rec},'captured_inputs':[{'boundary':'tool_result','result':rec}],
        'capture_receipts':[{'activity_id':'controlled_activity'}]}


def stub_agent(monkeypatch,invoke):
    agent=ModuleType('finplan_agent');agent.__path__=[]
    modeling=ModuleType('finplan_agent.modeling');modeling.__path__=[]
    hosted=ModuleType('finplan_agent.modeling.hosted');hosted.invoke_graph=invoke
    agent.modeling=modeling;modeling.hosted=hosted
    for name,module in [('finplan_agent',agent),('finplan_agent.modeling',modeling),('finplan_agent.modeling.hosted',hosted)]:
        monkeypatch.setitem(sys.modules,name,module)


def call(monkeypatch,returned,*,shared=None):
    captures=[];requests=[]
    def invoke(arguments,*,capture_before):
        requests.append(copy.deepcopy(arguments))
        capture_before({'boundary':'controlled_actual_graph_request','arguments':arguments})
        return copy.deepcopy(returned)
    stub_agent(monkeypatch,invoke)
    result=_graph_adapter({},shared or source(),holdings(),captures.append)
    return result,captures,requests


@pytest.mark.parametrize('change',[{'checksum':'sha256:'+'c'*64},{'strategy':'sac'}])
def test_changed_hosted_allocator_is_unavailable_and_original_capture_survives(monkeypatch,change):
    original=response(**change)
    result,captures,requests=call(monkeypatch,original)
    assert result['status']=='unavailable' and result['reason']=='hosted_model_identity_changed'
    assert result['answer']==original['answer'] and result['captured_inputs']==original['captured_inputs']
    assert result['capture_receipts']==original['capture_receipts']
    assert captures[0]['arguments']==requests[0]
    assert result['observed_policy_checksum']==original['answer']['recommendation']['policy_artifact_checksum']


def test_missing_actual_model_input_is_not_claimed_as_covered(monkeypatch):
    original=response(with_capture=False)
    result,_,_=call(monkeypatch,original)
    assert result['status']=='unavailable' and result['reason']=='hosted_actual_policy_inputs_or_valid_recommendation_unavailable'
    assert result['answer']==original['answer'] and result['captured_inputs']==original['captured_inputs']


def test_incompatible_cohort_records_not_invoked_without_calling_hosted_agent(monkeypatch):
    stub_agent(monkeypatch,lambda *args,**kw:pytest.fail('incompatible frozen universe must not invoke'))
    shared=source();shared['investable_universe']=['C','D'];captures=[]
    result=_graph_adapter({},shared,holdings(),captures.append)
    assert result['status']=='unavailable'
    assert captures[0]['invocation_state']=='not_invoked'
    assert captures[0]['supported_universe']==['A','B'] and captures[0]['supplied_universe']==['C','D']


def test_compatible_actual_sac_label_and_public_inputs_remain_honest(monkeypatch):
    original=response(strategy='sac')
    result,captures,requests=call(monkeypatch,original,shared=source(strategy='sac'))
    assert result['status']=='recommended' and result['framework']=='LangGraph'
    assert 'served sac' in result['allocation_origin'] and 'PPO' not in result['allocation_origin']
    assert result['answer']['recommendation']['diagnostics']['actual_model_inputs']==original['answer']['recommendation']['diagnostics']['actual_model_inputs']
    assert requests==[{'input_snapshot_id':'controlled_snapshot','as_of':'2026-07-10',
                      'holdings':holdings()}]
    assert len(captures)==1


def test_failed_hosted_response_preserves_original_error_and_partial_capture(monkeypatch):
    original=response();original.update(status='failed',reason='controlled_tool_failure')
    result,_,_=call(monkeypatch,original)
    assert result['status']=='failed' and result['reason']=='controlled_tool_failure'
    assert result['answer']==original['answer'] and result['captured_inputs']==original['captured_inputs']
