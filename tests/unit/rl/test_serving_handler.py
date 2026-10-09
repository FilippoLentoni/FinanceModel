import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from finplan_contracts import boundaries
from finplan_contracts.iam import Request, evaluate

from finplan_model import serving
from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.core.errors import FinplanError
from finplan_model.sim.market import synthetic_market
from infra.stacks.policies import inference_role_policy

UID = '01JABCDEFGHJKMNPQRSTVWXY01'


@pytest.fixture
def setup(monkeypatch):
    market = synthetic_market(('A','B'), n_sessions=100)
    artifacts = InMemoryArtifactStore()
    bundle = {'format':'finplan-strategy-bundle/1','mode':'advisory_paper','source_run_id':'run_'+UID,
              'configuration_id':'cfg_'+'a'*64, 'strategy_id':'equal_weight', 'parameters':{},
              'instruments':['A','B'], 'constraints':{}, 'members':[], 'available_after':'2026-01-01'}
    ref = artifacts.put_json(bundle, kind='policy_inference')
    pointer = json.dumps({'export_run_id':'run_'+UID,'artifact':ref.to_dict()})
    deps = SimpleNamespace(advisory_parameter=SimpleNamespace(read=lambda:pointer), artifacts=artifacts, platform=None)
    svc = SimpleNamespace(env='beta', d=deps, now=lambda:datetime(2026,10,9,tzinfo=UTC))
    content = SimpleNamespace(snapshot=SimpleNamespace(manifest_checksum='sha256:'+'b'*64,record={}),
                              payload={'observations':[{'instrument_id':i,'session_date':b.session_date.isoformat(),'close':b.close} for i in market.instruments for b in market.bars_of(i)]})
    monkeypatch.setattr('finplan_model.rl.advisory.load_market',lambda *a:(market,content))
    body = {'input_snapshot_id':'snap_'+UID,'as_of':market.sessions[-1].isoformat(),
            'holdings':{'weights':[],'cash_weight':1.,'portfolio_value':1000.,'high_watermark':1000.}}
    return svc, body, ref


def test_serving_returns_contract_valid_repeatable_recommendation_without_job_dependencies(setup):
    svc, body, _ = setup
    event = {'environment':'beta','request':body,'headers':{'X-Correlation-Id':'corr-test-0001'}}
    a = serving.handle(event, svc)
    assert serving.handle(event, svc) == a
    assert a['recommendation']['strategy'] == 'equal_weight'
    assert a['recommendation']['cash_weight'] == 0
    assert [w['weight'] for w in a['recommendation']['target_weights']] == [.5,.5]
    assert not hasattr(svc.d,'sagemaker') and not hasattr(svc.d,'run_io')


def test_corrupt_artifact_fails_checksum_validation(setup):
    svc, body, ref = setup
    svc.d.artifacts.objects[ref.artifact_id] = (ref, b'corrupt')
    with pytest.raises(FinplanError) as err:
        serving.handle({'environment':'beta','request':body},svc)
    assert err.value.details['reason'] == 'artifact_checksum_mismatch'


@pytest.mark.parametrize('case', ['environment','future','missing_holdings','unknown_holding'])
def test_serving_rejects_invalid_decision_context(setup,case):
    svc, body, _ = setup
    env = 'beta'
    if case == 'environment': env = 'prod'
    if case == 'future': body['as_of'] = '2027-01-01'
    if case == 'missing_holdings': del body['holdings']
    if case == 'unknown_holding': body['holdings']['weights'] = [{'instrument_id':'C','weight':.2}]
    with pytest.raises(FinplanError):
        serving.handle({'environment':env,'request':body},svc)


def test_inference_iam_has_no_training_selection_or_storage_write_permission():
    account = '<account-id>'
    kw = {'partition':'aws','region':'us-east-2','account':account}
    policy = inference_role_policy('beta',**kw)
    boundary = boundaries.env_permission_boundary('beta',**kw)
    def allowed(action, resource):
        return evaluate(Request(action,resource),{'identity':policy},boundary).allowed
    research = f'arn:aws:s3:::finplan-beta-financemodel-research-workspace-{account}'
    assert allowed('s3:GetObject', research+'/artifacts/policy_inference/example.json')
    assert not allowed('s3:GetObject', research+'/artifacts/rl_policy/example.zip')
    for action in ('sagemaker:CreateProcessingJob','dynamodb:PutItem','s3:PutObject','ssm:PutParameter'):
        assert not allowed(action,'*')
    root = f'arn:aws:execute-api:us-east-2:{account}:api/beta/'
    assert allowed('execute-api:Invoke', root+'GET/v1/portfolios/pf_example/state')
    assert allowed('execute-api:Invoke', root+'GET/v1/plans/pl_example')
    assert allowed('execute-api:Invoke', root+'GET/v1/snapshots/latest')
    assert not allowed('execute-api:Invoke', root+'PUT/v1/portfolios/pf_example/state')
    assert not allowed('execute-api:Invoke', root+'POST/v1/ingestions')
    param = f'arn:aws:ssm:us-east-2:{account}:parameter/finplan/beta/financialplanning/config/research-plan-ref'
    assert allowed('ssm:GetParameter', param)
    assert not allowed('ssm:GetParameter', param.replace('/beta/', '/gamma/'))
