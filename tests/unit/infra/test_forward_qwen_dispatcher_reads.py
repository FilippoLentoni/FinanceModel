"""Only the beta dispatcher gains two eligibility reads, paired with platform ownership guard."""
import hashlib,json
from finplan_contracts import boundaries
from finplan_contracts.iam import Request,evaluate
from infra.stacks.policies import control_role_policy
CTX={'partition':'aws','region':'us-east-2','account':'<account-id>'}
BASE='arn:aws:execute-api:us-east-2:<account-id>:api/v1/'
BOOK='pf_01KDVDNAZ83BAMMYCEGWF33DPM'


def allowed(env,kind,method,path):
    return evaluate(Request('execute-api:Invoke',BASE+method+'/'+path,{}),[control_role_policy(env,kind,**CTX)],boundaries.env_permission_boundary(env,**CTX)).allowed


def test_only_beta_dispatcher_two_book_reads_and_no_new_write_or_other_child():
    for path in ('v1/portfolios/'+BOOK,'v1/portfolios/'+BOOK+'/state'):
        assert allowed('beta','dispatcher','GET',path)
        for kind in ('api','state'):assert not allowed('beta',kind,'GET',path)
        for env in ('gamma','prod'):assert not allowed(env,'dispatcher','GET',path)
        for method in ('PUT','POST','PATCH','DELETE'):assert not allowed('beta','dispatcher',method,path)
    for suffix in ('history','decisions','settings','state/extra'):
        assert not allowed('beta','dispatcher','GET','v1/portfolios/'+BOOK+'/'+suffix)
    table='arn:aws:dynamodb:us-east-2:<account-id>:table/finplan-beta-financialplanning-portfolio'
    for action in ('dynamodb:GetItem','dynamodb:Scan','dynamodb:UpdateItem'):
        assert not evaluate(Request(action,table,{}),[control_role_policy('beta','dispatcher',**CTX)],boundaries.env_permission_boundary('beta',**CTX)).allowed


def test_all_six_higher_control_policy_documents_byteidentical_to_cef():
    expected={'gamma':{'api':'2dbc3ccae1b40ac76e2435852033cc87b412a5ccf2af14f772fa5e61226edf7a','dispatcher':'eb4d843286ac81fbe6517e1af33d7b64730689b836a08275fb4c943ae2116660','state':'08d3efa36ae508c1a0bee544c034832f0f3a4aa4b35a27f2d74daf6b949b955c'},'prod':{'api':'f770c8e69470268338403242e869750c398ceac076e9d9cf389380688b81a642','dispatcher':'635e003fb16c233ccf59bc208d896c46a257b068c618a6ce6c1c837f7f66ab08','state':'626d8f4944c425192cecf0996f265caf06cb8df2cdf967af58f455f0f22b95f4'}}
    for env,roles in expected.items():
        for kind,digest in roles.items():assert hashlib.sha256(json.dumps(control_role_policy(env,kind,**CTX),sort_keys=True,separators=(',',':')).encode()).hexdigest()==digest
