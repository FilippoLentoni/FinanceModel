import copy
import pytest
from finplan_model.paper_profile import initialize,approved_profile_instruments,identity


def test_initialization_is_separate_retry_safe_and_protects_existing_book():
    existing={'portfolio_id':'old','paper_state':{'cash_balance':10000},'revision':7};calls=[];created=None;state=None;receipts=[]
    def client(method,path,body=None):
        nonlocal created,state
        calls.append((method,path,copy.deepcopy(body)))
        if method=='GET' and path=='/v1/portfolios/old/state':return copy.deepcopy(existing)
        if method=='POST':
            created=created or {**body,'portfolio_id':'new'}
            return copy.deepcopy(created)
        if method=='PUT':
            state=state or {'portfolio_id':'new','paper_state':body['paper_state'],'revision':1}
            return copy.deepcopy(state)
        raise AssertionError(path)
    first=initialize(client,caller_id='synthetic-caller',profile_id='etf-stock-100',as_of='2026-01-02',persist=lambda x:receipts.append(x),protected_portfolio_ids=['old'])
    second=initialize(client,caller_id='synthetic-caller',profile_id='etf-stock-100',as_of='2026-01-02',receipt=first,protected_portfolio_ids=['old'])
    assert first==second and first['portfolio_id']=='new' and existing['paper_state']['cash_balance']==10000
    assert receipts[0].get('portfolio_id') is None and receipts[1]['portfolio_id']=='new'
    assert not any(method=='PUT' and '/old/' in path for method,path,_ in calls)
    with pytest.raises(ValueError):initialize(client,caller_id='other',profile_id='etf-stock-100',as_of='2026-01-02',receipt=first)


def test_profile_universe_is_explicit_and_defaults_preserved():
    profile={'caller_id':'caller','profile_id':'small','identity':identity('caller','small'),'initial_cash':100,'instruments':['VOO','GOOGL']}
    p={'portfolio_id':'new','synthetic':True,'settings':{'paper_research_profile':profile}}
    assert approved_profile_instruments(p,portfolio_id='new',available_instruments=['VOO','GOOGL','AAPL'])==['GOOGL','VOO']
    assert approved_profile_instruments({'portfolio_id':'old'},portfolio_id='old',available_instruments=['VOO','AAPL'])==['AAPL','VOO']
    profile['identity']='tampered'
    with pytest.raises(Exception):approved_profile_instruments(p,portfolio_id='new',available_instruments=['VOO','GOOGL'])


def test_eight_asset_profile_preserves_explicit_price_only_or_news_requirement():
    from finplan_model.paper_profile import DAILY_INSTRUMENTS
    profile={'caller_id':'caller','profile_id':'daily8','identity':identity('caller','daily8'),'initial_cash':100,'instruments':list(DAILY_INSTRUMENTS),'news_required':False}
    portfolio={'portfolio_id':'new','synthetic':True,'settings':{'paper_research_profile':profile}}
    assert approved_profile_instruments(portfolio,portfolio_id='new',available_instruments=[*DAILY_INSTRUMENTS,'NFLX'])==sorted(DAILY_INSTRUMENTS)
    profile['news_required']='false'
    with pytest.raises(Exception):approved_profile_instruments(portfolio,portfolio_id='new',available_instruments=DAILY_INSTRUMENTS)
