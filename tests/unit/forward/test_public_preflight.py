"""Actual native public request preparation; no vendor HTTP, secret reads, or GPU."""
import copy,json
from datetime import date,timedelta
import pytest
from finplan_model.forward import public_preflight as public
from finplan_model.forward.runtime import evaluate_approach


def source():
    first=date(2026,4,1)
    return {'synthetic':True,'investable_universe':['SPY','QQQ'],'input_snapshot_id':'snap_01ARZ3NDEKTSV4RRFFQ69G5FAV',
        'observations':[{'instrument_id':symbol,'session_date':(first+timedelta(days=n)).isoformat(),'close':100+n*(1 if symbol=='SPY' else .5),'available_at':'2026-07-10T00:00:00Z'} for n in range(65) for symbol in ('SPY','QQQ')]}


def holdings():return {'portfolio_id':'controlled-book','revision':3,'positions':[{'instrument_id':'SPY','quantity':10.}], 'cash':1000.,'weights':[{'instrument_id':'SPY','weight':.5},{'instrument_id':'QQQ','weight':0.}], 'cash_weight':.5,'portfolio_value':2000.}


def prepare(kind,*,data=None,book=None):
    captured=[]
    base={'run_id':'controlled-public-preflight','availability_cutoff':'2026-07-11T00:05:00Z'}
    result=public.preflight(kind,base,data or source(),book or holdings(),captured.append,dependency_reader=lambda k:{'status':'controlled_no_live_metadata'})
    return result,captured[0]


def test_jev_native_public_request_and_local_transforms_are_exact_not_consumed(monkeypatch):
    from finplan_model.benchmarks.jev import JevClient,encode_state
    monkeypatch.setattr(JevClient,'_send',lambda *a:pytest.fail('vendor transport called'))
    result,capture=prepare('jev')
    assert result['status']=='unavailable' and result['usage']['calls']==0 and capture['invocation_state']=='not_invoked'
    assert 'public_request' in capture,capture
    market,view,hv,private,raw=public._view(source(),holdings(),'2026-07-11T00:05:00Z')
    request=capture['public_request']['payload']
    assert request['state']==encode_state(view,hv)
    assert request['questions']['SPY']['type']=='choice'
    assert set(request['questions'])=={'SPY','QQQ'}
    assert capture['local_transformation']['native_weights']==hv.weights
    assert capture['local_transformation']['simple_return_values']==private['simple_return_values']
    assert not capture['public_request']['supplied_to_model'] and capture['outbound_guard']['status']=='passed'
    assert 'prices' not in json.dumps(request) and 'quantity' not in json.dumps(request)
    proposal=capture['managed_pilot_proposal']
    assert not proposal['available']  # This controlled forward-only source has no historical vintages.
    assert 'aligned completed bars' in proposal['reason']
    assert capture['controls']['configured_limits']['external_vendor_cost_cap_usd']==.01


def test_qwen_uses_actual_first_native_role_no_dependent_model_outputs_or_gpu(monkeypatch):
    monkeypatch.setattr(public,'QwenSwarm',public.QwenSwarm)
    result,capture=prepare('qwen')
    assert result['usage']['calls']==0 and result['status']=='unavailable'
    assert len(capture['prepared_role_messages'])==1
    first=capture['prepared_role_messages'][0]
    assert first['role']=='market_analyst' and not first['supplied_to_model']
    payload=json.loads(first['messages'][1]['content'])
    assert payload['evidence']['current_weights']=={i:capture['local_transformation']['native_weights'].get(i,0.) for i in source()['investable_universe']}
    assert capture['declared_roles']==list(public.ROLES)
    assert 'NOT_PREPARED' in capture['dependent_role_inputs']


def test_outbound_guard_rejection_still_captures_prepared_request_without_transport(monkeypatch):
    import finplan_model.benchmarks.jev as native
    monkeypatch.setattr(native,'encode_state',lambda *a:{'prices':[100.,101.,102.,103.]})
    result,capture=prepare('jev')
    assert capture['outbound_guard']['status']=='rejected' and capture['invocation_state']=='not_invoked'
    assert result['usage']['calls']==0 and not capture['public_request']['supplied_to_model']


@pytest.mark.parametrize('field,value',[('cash',float('nan')),('cash',-1)])
def test_malformed_holdings_never_form_a_vendor_request(field,value):
    book=holdings();book[field]=value
    result,capture=prepare('jev',book=book)
    assert result['status']=='unavailable' and capture['public_request'] is None and capture['preparation_error']


def test_runtime_input_labels_no_news_no_inference_and_capture_errors_stop(monkeypatch):
    captures=[]
    adapter=lambda base,s,h,before:public.preflight('jev',base,s,h,before,dependency_reader=lambda k:{'status':'controlled'})
    result=evaluate_approach('jev_systemone',source(),holdings(),cutoff='2026-07-11T00:05:00Z',run_id='controlled-run',input_id='controlled-input',capture=captures.append,external_adapters={'jev':adapter})
    assert result['status']=='unavailable' and not captures[0]['news']['consumed']
    assert captures[0]['invocation_state']=='not_invoked'
    assert captures[0]['numeric']['prepared']['public_request']['status']=='prepared_not_supplied'
    with pytest.raises(RuntimeError,match='capture unavailable'):
        evaluate_approach('jev_systemone',source(),holdings(),cutoff='2026-07-11T00:05:00Z',run_id='controlled-run',input_id='controlled-input',capture=lambda p:(_ for _ in ()).throw(RuntimeError('capture unavailable')),external_adapters={'jev':adapter})


def test_supported_managed_dry_run_requires_actual_historical_availability():
    data=source()
    for row in data['observations']:row['available_at']=row['session_date']+'T21:00:00Z'
    result,capture=prepare('jev',data=data)
    proposal=capture['managed_pilot_proposal']
    assert proposal['available'] and proposal['arguments']['dry_run'] is True
    assert not proposal['submitted'] and not proposal['same_current_book_forward_bridge']
    assert proposal['arguments']['job_type']=='jev_backtest' and proposal['managed_runtime_bound_seconds']==900
    assert result['usage']['calls']==0


def test_capture_failure_is_never_replaced_by_a_second_different_capture():
    calls=[]
    def fail(payload):calls.append(payload);raise RuntimeError('capture failed')
    with pytest.raises(RuntimeError,match='capture failed'):
        public.preflight('jev',{'run_id':'controlled','availability_cutoff':'2026-07-11T00:05:00Z'},source(),holdings(),fail,dependency_reader=lambda k:{'status':'controlled'})
    assert len(calls)==1


def test_untrusted_approval_or_malformed_proposed_policy_cannot_enable_transport():
    data=source();data['approved_jev_output']={'weights':{'SPY':2.},'cash':-1.};data['manual_approval']=True
    result,capture=prepare('jev',data=data)
    assert result['status']=='unavailable' and result['usage']['calls']==0
    assert 'target_weights' not in result and capture['controls']['manual_approved_managed_worker_required']


def test_metadata_checks_never_load_secret_values_and_use_native_cost_estimator(monkeypatch):
    from datetime import datetime,timezone
    from types import SimpleNamespace
    class MetadataOnly:
        def get_parameter(self,Name):
            assert Name.endswith('instance-prices')
            return {'Parameter':{'Value':json.dumps({'retrieved_at':datetime.now(timezone.utc).isoformat(),'usd_per_hour':{'ml.m5.xlarge':.23}})}}
        def describe_secret(self,**kwargs):return {'Name':'controlled-reference'}
        def get_secret_value(self,**kwargs):pytest.fail('secret value loaded')
    monkeypatch.setattr(public.boto3,'Session',lambda **kw:SimpleNamespace(client=lambda *a,**kw:MetadataOnly()))
    metadata=public.dependency_metadata('jev')
    assert metadata['credential_metadata_exists'] and metadata['secret_value_loaded'] is False
    assert metadata['vendor_authentication_tested'] is False
    cost=metadata['managed_cost_estimate']
    assert cost['status']=='native_cost_upper_bound' and cost['instance_type']=='ml.m5.xlarge'
    assert cost['max_runtime_seconds']==900 and cost['estimated_usd_upper_bound']==.0675
    assert cost['vendor_cost_cap_usd']==.01 and not cost['submitted'] and not cost['approval_granted']


def test_absent_qwen_metadata_is_not_a_local_only_or_approved_gpu_claim(monkeypatch):
    from types import SimpleNamespace
    from botocore.exceptions import ClientError
    class MetadataOnly:
        def get_parameter(self,Name):
            if Name.endswith('instance-prices'):raise ClientError({'Error':{'Code':'ParameterNotFound'}},'GetParameter')
            return {'Parameter':{'Value':'controlled-private-bucket'}}
        def get_object(self,**kwargs):raise ClientError({'Error':{'Code':'NoSuchKey'}},'GetObject')
    monkeypatch.setattr(public.boto3,'Session',lambda **kw:SimpleNamespace(client=lambda *a,**kw:MetadataOnly()))
    metadata=public.dependency_metadata('qwen')
    assert metadata['status']=='required_stage_metadata_missing'
    assert metadata['weights_loaded'] is False and metadata['full_weight_bytes_verified'] is False
    assert metadata['managed_cost_estimate']['reason']=='instance_price_missing'
