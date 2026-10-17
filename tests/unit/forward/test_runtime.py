from datetime import UTC,date,datetime,timedelta
import copy
import pytest
from finplan_model.forward.runtime import evaluate_approach,validate_policy
from finplan_model.forward.forecasts import freeze,reconcile,errors
from finplan_model.forward.paper import apply_next_session
from finplan_model.sim.market import synthetic_market
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum


def source():
    start=date(2026,5,1)
    return {'investable_universe':['A','B'],'observations':[{'instrument_id':i,'session_date':(start+timedelta(days=n)).isoformat(),
           'close':100+n*(1 if i=='A' else .5)+((n%3)-1),'available_at':'2026-07-10T00:00:00Z'} for n in range(65) for i in ('A','B')]}


def holdings():return {'positions':[],'cash':100000.,'cash_balance':100000.,'weights':[],'cash_weight':1.,'portfolio_value':100000.,'high_watermark':100000.,'revision':1}


def run(aid,**kw):
    captured=[]
    result=evaluate_approach(aid,source(),holdings(),cutoff='2026-07-11T00:00:00Z',run_id='run',input_id='input',capture=captured.append,**kw)
    return result,captured


def test_solver_actual_covariance_before_invocation(monkeypatch):
    import finplan_model.strategies.optimizers as solver
    original=solver.minimize;captured=[]
    def check(*a,**kw):
        assert captured and captured[0]['numeric']['prepared']['covariance']
        return original(*a,**kw)
    monkeypatch.setattr(solver,'minimize',check)
    outcome=evaluate_approach('mean_variance',source(),holdings(),cutoff='2026-07-11T00:00:00Z',run_id='r',input_id='i',capture=captured.append)
    assert outcome['status']=='recommended'
    assert captured[0]['news']['consumed'] is False
    assert captured[0]['numeric']['prepared']['mean']


def test_capture_failure_prevents_solver(monkeypatch):
    import finplan_model.strategies.optimizers as solver
    monkeypatch.setattr(solver,'minimize',lambda *a,**k:pytest.fail('must not invoke'))
    def fail(_):raise RuntimeError('storage unavailable')
    with pytest.raises(RuntimeError,match='storage unavailable'):
        evaluate_approach('mean_variance',source(),holdings(),cutoff='2026-07-11T00:00:00Z',run_id='r',input_id='i',capture=fail)


def test_unavailable_policy_still_captures_actual_preconditions():
    result,captured=run('sac')
    assert result['status']=='unavailable'
    assert captured[0]['invocation_state']=='not_invoked'
    assert captured[0]['news']['consumed'] is False


def test_no_missing_position_sell():
    result=validate_policy({'weights':{'A':1},'cash':0},['A','B'],holdings())
    assert result['intents'][1]['action']=='hold'


def test_no_lookahead():
    s=source();future=s['observations'][-1];future['available_at']='2099-01-01T00:00:00Z';future['close']=1e20;captures=[]
    evaluate_approach('equal_weight',s,holdings(),cutoff='2026-07-11T00:00:00Z',run_id='r',input_id='i',capture=captures.append)
    assert captures[0]['numeric']['prepared']['eligible_prices'][future['instrument_id']]!=future['close']
    assert 'price_history' not in captures[0]['numeric']


def test_forecast_pending_alignment_immutable_known_outcome():
    f=freeze({'status':'forecast','target_at':'2026-07-15T14:00:00Z','expected_value':102000,'baseline_value':100000,'units':'USD'},decision_at='2026-07-14T14:00:00Z',run_id='r',portfolio_id='p')
    original=copy.deepcopy(f)
    rows=[{'portfolio_id':'p','observation_id':str(n),'observed_at':at,'value':value,'observation_kind':kind,'quote_status':fresh,'net_external_cash_flows_since_decision':1000} for n,at,value,kind,fresh in (
        (1,'2026-07-15T13:00:00Z',100000,'valid','fresh'),(2,'2026-07-15T14:00:00Z',100000,'stale','stale'),
        (3,'2026-07-15T14:30:00Z',104000,'valid','fresh'),(4,'2026-07-15T15:00:00Z',110000,'valid','fresh'))]
    assert reconcile(f,rows[:2]) is None
    matched=reconcile(f,rows)
    assert matched['observation_id']=='3' and matched['signed_error']==1000
    assert errors([matched])['rmse']==1000
    assert reconcile(f,rows[3:],previous_match=matched)==matched
    assert f==original


def test_paper_closed_market_does_not_fill():
    market=synthetic_market(start=date(2026,7,1),n_sessions=4,instruments=('A','B'))
    state={'positions':[],'cash_balance':100000.,'high_watermark':100000.,'as_of':'2026-07-01','base_currency':'USD','mode':'paper','source':'paper_initialization'}
    result=apply_next_session({'target_weights':[{'instrument_id':'A','weight':.5},{'instrument_id':'B','weight':.5}],'cash_weight':0},state,market,decision_at='2026-07-10T22:00:00Z',observed_at='2026-07-10T22:00:00Z')
    assert result['status']=='pending_market_data'
    assert result['fills']==[]


def test_controlled_buy_sell_replay_uses_common_costs_and_conserves_cash():
    from finplan_model.sim.market import Bar,MarketData
    days=[date(2026,1,n) for n in (12,13,14)]
    market=MarketData(days,[Bar('A',d,100.,datetime(d.year,d.month,d.day,21,tzinfo=UTC),open=100.,volume=1000.) for d in days],synthetic=True)
    state={'positions':[],'cash_balance':10000.,'high_watermark':10000.,'as_of':'2026-01-12','base_currency':'USD','mode':'paper','source':'paper_initialization'}
    config={'fees':{'proportional_bps':10,'fixed_per_trade':1},'spread':{'half_spread_bps':5},'slippage':{'model':'linear_participation','coefficient_bps':10},'liquidity':{'participation_cap':.05,'unfilled':'cancel'}}
    bought=apply_next_session({'target_weights':[{'instrument_id':'A','weight':.5}],'cash_weight':.5},state,market,decision_at='2026-01-12T22:00:00Z',observed_at='2026-01-13T22:00:00Z',config=config)
    assert bought['status']=='applied'
    assert bought['fills'][0]['quantity']==50
    assert bought['execution']['transaction_cost']==pytest.approx(8.75)
    assert bought['state']['cash_balance']==pytest.approx(4991.25)
    assert bought['execution']['before_value']==pytest.approx(bought['execution']['after_value']+8.75)
    sold=apply_next_session({'target_weights':[{'instrument_id':'A','weight':0}],'cash_weight':1},bought['state'],market,decision_at='2026-01-13T22:00:00Z',observed_at='2026-01-14T22:00:00Z',config=config)
    assert sold['fills'][0]['quantity']==-50
    assert sold['state']['cash_balance']==pytest.approx(9982.5)
    assert sold['state']['positions']==[{'instrument_id':'A','quantity':0.0}]
    assert sold['execution']['evaluator_version']==bought['execution']['evaluator_version']


def point_quotes(*,at='2026-01-13T15:05:00Z',price=100.,volume=1000):
    return {'A':{'instrument_id':'A','price':price,'volume':volume,'status':'fresh','market_open':True,
                 'available_at':at,'quoted_at':'2026-01-13T15:05:00Z','bar_started_at':'2026-01-13T15:04:00Z',
                 'regular_session':{'start':'2026-01-13T14:30:00Z','end':'2026-01-13T21:00:00Z'}}}


def test_point_price_common_kernel_matches_controlled_daily_fill_costs():
    from finplan_model.evaluate.evaluator import evaluate_point_rebalance
    from finplan_model.sim.config import SimulationConfig
    config=SimulationConfig.from_dict({'fees':{'proportional_bps':10,'fixed_per_trade':1},'spread':{'half_spread_bps':5},'slippage':{'model':'linear_participation','coefficient_bps':10},'liquidity':{'participation_cap':.05,'unfilled':'cancel'}})
    result=evaluate_point_rebalance({'weights':{'A':.5},'cash':.5},positions={},cash=10000,quotes=point_quotes(),config=config,
               decision_at='2026-01-13T14:05:00Z',opportunity_at='2026-01-13T15:05:00Z',recorded_at='2026-01-13T15:05:01Z',universe=['A'])
    assert result['fills'][0]['quantity']==50
    assert result['transaction_cost']==pytest.approx(8.75)
    assert result['cash']==pytest.approx(4991.25)
    assert result['execution_rule']=='next_hourly_regular_session_quote'
    assert result['before_value']==pytest.approx(result['after_value']+8.75)


def test_point_price_rejects_hindsight_stale_off_session_and_predecision_quotes():
    from finplan_model.evaluate.evaluator import evaluate_point_rebalance
    from finplan_model.sim.config import SimulationConfig
    from finplan_model.core.errors import FinplanError
    config=SimulationConfig.from_dict({})
    def check(quotes,recorded='2026-01-13T15:05:01Z'):
        return evaluate_point_rebalance({'weights':{'A':1},'cash':0},positions={},cash=10000,quotes=quotes,config=config,
            decision_at='2026-01-13T14:05:00Z',opportunity_at='2026-01-13T15:05:00Z',recorded_at=recorded,universe=['A'])
    with pytest.raises(FinplanError,match='eligible window'):check(point_quotes(),'2026-01-13T15:10:00Z')
    q=point_quotes();q['A']['market_open']=False
    with pytest.raises(FinplanError,match='off-session'):check(q)
    q=point_quotes();q['A']['available_at']='2026-01-13T15:06:00Z'
    with pytest.raises(FinplanError,match='eligible window'):check(q)
    q=point_quotes();q['A']['bar_started_at']='2026-01-13T14:04:00Z'
    with pytest.raises(FinplanError,match='eligible window'):check(q)


def test_point_price_partial_liquidity_and_cash_limited_buy_cannot_overdraw():
    from finplan_model.evaluate.evaluator import evaluate_point_rebalance
    from finplan_model.sim.config import SimulationConfig
    result=evaluate_point_rebalance({'weights':{'A':1},'cash':0},positions={},cash=10000,quotes=point_quotes(volume=100),config=SimulationConfig.from_dict({}),
               decision_at='2026-01-13T14:05:00Z',opportunity_at='2026-01-13T15:05:00Z',recorded_at='2026-01-13T15:05:01Z',universe=['A'])
    assert result['positions']['A']==5
    assert result['unfilled'][0]['reason']=='liquidity_cap'
    result=evaluate_point_rebalance({'weights':{'A':1},'cash':0},positions={},cash=10000,quotes=point_quotes(volume=10000000),config=SimulationConfig.from_dict({'fees':{'proportional_bps':100}}),
               decision_at='2026-01-13T14:05:00Z',opportunity_at='2026-01-13T15:05:00Z',recorded_at='2026-01-13T15:05:01Z',universe=['A'])
    assert result['cash']>=0
    assert any(r['reason']=='cash_limited' for r in result['unfilled'])


def test_real_daily_reconstruction_cannot_create_forward_open_fill():
    from finplan_model.sim.market import Bar,MarketData
    days=[date(2026,1,n) for n in (12,13)]
    market=MarketData(days,[Bar('A',d,100.,datetime(d.year,d.month,d.day,21,tzinfo=UTC),open=100.,volume=1000.) for d in days],synthetic=False)
    state={'positions':[],'cash_balance':10000.,'high_watermark':10000.,'as_of':'2026-01-12','base_currency':'USD','mode':'paper','source':'paper_initialization'}
    result=apply_next_session({'target_weights':[{'instrument_id':'A','weight':.5}],'cash_weight':.5},state,market,decision_at='2026-01-12T22:00:00Z',observed_at='2026-01-13T22:00:00Z')
    assert result['status']=='pending_market_data' and result['fills']==[]



def test_codex_capture_never_substitutes_shared_raw_history_for_consumed_input():
    def adapter(base,source,holdings,before):
        before({'boundary':'controlled_mock_codex_request','actual_source':{'news':[],'market_descriptors':{'trend':'positive'}},'messages':[{'role':'user','content':'controlled derived descriptors'}]})
        return {'weights':{'A':.5,'B':.5},'cash':0}
    outcome,captured=run('direct_codex_news',external_adapters={'codex':adapter})
    assert outcome['status']=='recommended'
    assert 'price_history' not in captured[0]['numeric']
    assert captured[0]['collected_not_consumed']['price_history']['status']=='not_supplied_to_this_approach'
    assert captured[0]['numeric']['prepared']['actual_source']['market_descriptors']['trend']=='positive'
