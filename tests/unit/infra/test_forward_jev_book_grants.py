"""Managed current-book eligibility reads stay beta-only and never permit writes."""
from finplan_contracts import boundaries
from finplan_contracts.iam import Request,evaluate
from infra.stacks.policies import job_execution_policy

CTX={'partition':'aws','region':'us-east-2','account':'<account-id>'}
BOOK='pf_01KDVDNAZ83BAMMYCEGWF33DPM'
BASE='arn:aws:execute-api:us-east-2:<account-id>:api/v1/'


def allowed(env,method,path):
    return evaluate(Request('execute-api:Invoke',BASE+method+'/'+path,{}),[job_execution_policy(env,**CTX)],boundaries.research_permission_boundary(env,**CTX)).allowed


def test_only_beta_two_exact_book_get_paths_and_no_writes_or_other_children():
    for path in ('v1/portfolios/'+BOOK,'v1/portfolios/'+BOOK+'/state'):
        assert allowed('beta','GET',path)
        for env in ('gamma','prod'):assert not allowed(env,'GET',path)
        for method in ('POST','PUT','PATCH','DELETE'):assert not allowed('beta',method,path)
    for suffix in ('decisions','history','settings','state/extra'):
        assert not allowed('beta','GET','v1/portfolios/'+BOOK+'/'+suffix)


def test_higher_environment_policy_documents_remain_byte_identical():
    import hashlib,json
    parent={'gamma':'c9dda96323a224300be2abaff17f939c1d7aeec8fe9b1e8183c61de596debad8',
            'prod':'e05890fb165a018d6f97a67fefccd4c6e6313cef273fd0e978c321b21bbac2f4'}
    # Exact 8ec parent documents under the same symbolic context, not only absence of a new Sid.
    for env in ('gamma','prod'):
        policy=job_execution_policy(env,**CTX)
        assert hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()==parent[env]
        assert all(row.get('Sid')!='BetaIsolatedPaperBookEligibilityReads' for row in policy['Statement'])
        assert any(row.get('Sid')=='PlatformSnapshotReads' for row in policy['Statement'])
        assert any(row.get('Effect')=='Deny' for row in policy['Statement'])


def test_published_marker_roundtrips_existing_owned_ssm_reference_without_new_schema():
    import json
    from finplan_contracts import ssm
    from scripts.release import planned_parameters
    from tests.unit.infra.test_release_publish import _info,ACCT
    from finplan_model.control.settings import SsmSettings
    from finplan_model.core.config import load_config
    for env in ('beta','gamma','prod'):
        plan=planned_parameters(env,_info(),{},account=ACCT,deployed_job_types=['jev_backtest'])
        path,value=plan['job-jev-backtest'];doc=json.loads(value)
        assert ssm.check_write(path,ssm.Writer('financemodel',environment=env)).allowed
        assert ssm.validate_value(path,value)==[]
        class Parameters:
            def get_parameter(self,Name):
                assert Name==path
                return {'Parameter':{'Value':value}}
        assert SsmSettings(Parameters(),load_config(env)).job_definition('jev_backtest')==doc
        assert doc.get('forward_protocol')==('finplan-jev-forward/1' if env=='beta' else None)
        assert doc.get('forward_source_commit')==(_info().source_commit if env=='beta' else None)


def test_beta_worker_only_reads_owned_context_metadata_not_runs_leases_or_platform():
    table='arn:aws:dynamodb:us-east-2:<account-id>:table/finplan-beta-financemodel-job-control'
    policy=job_execution_policy('beta',**CTX);boundary=boundaries.research_permission_boundary('beta',**CTX)
    def check(action,key,resource=table):
        return evaluate(Request(action,resource,{'dynamodb:LeadingKeys':[key]}),[policy],boundary).allowed
    assert check('dynamodb:GetItem','FORWARD_CONTEXT#controlled')
    for key in ('RUN#controlled','LEASE#beta#cpu','IDEM#controlled','FORWARD_COST#controlled'):
        assert not check('dynamodb:GetItem',key)
    for action in ('dynamodb:PutItem','dynamodb:UpdateItem','dynamodb:Query','dynamodb:Scan','dynamodb:DeleteItem'):
        assert not check(action,'FORWARD_CONTEXT#controlled')
    assert not check('dynamodb:GetItem','FORWARD_CONTEXT#controlled',table.replace('financemodel-job-control','financialplanning-portfolio'))
    for env in ('gamma','prod'):
        higher=job_execution_policy(env,**CTX)
        assert not evaluate(Request('dynamodb:GetItem',table.replace('-beta-','-'+env+'-'),{'dynamodb:LeadingKeys':['FORWARD_CONTEXT#controlled']}),[higher],boundaries.research_permission_boundary(env,**CTX)).allowed
