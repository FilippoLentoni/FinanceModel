from datetime import UTC,date,datetime
import pytest
from finplan_model.core.errors import FinplanError
from finplan_model.sim.market import Bar,MarketData
from finplan_model.rl.performance import decompose


def inputs():
    window={'start':'2026-01-05','end':'2026-01-06'}
    market=MarketData([date(2026,1,5),date(2026,1,6)],[Bar('SYNTH',date(2026,1,5),100,datetime(2026,1,5,21,tzinfo=UTC)),Bar('SYNTH',date(2026,1,6),110,datetime(2026,1,6,21,tzinfo=UTC))])
    pub={'publication_id':'pub_synthetic','plan_version_id':'pv_synthetic','plan_version_checksum':'sha256:synthetic'}
    version={'plan_version_id':'pv_synthetic','checksum':'sha256:synthetic','content':{'allocation':{'weights':[{'instrument_id':'SYNTH','weight':.5}],'cash_weight':.5}}}
    record={'execution_id':'exec_synthetic','mode':'paper','ledger':{'payload':{'window':window,'start_value':1000.,'end_value':1029.,'cash_flows':0.,'realized_pnl':10.,'unrealized_pnl':20.,'fees':1.}}}
    request={'window':window,'actual_source':'paper','realized_snapshot_id':'snap_synthetic'}
    return pub,version,[record],market,request


def test_accounting_recovers_planted_exposure_and_fee_gap_without_inventing_causes():
    ev=decompose(*inputs())
    assert ev['reconciliation']['plan_path']['end_value']==pytest.approx(1050)
    assert ev['gap']['total']==pytest.approx(-21)
    assert ev['gap']['components']['execution']==pytest.approx(-20)
    assert ev['gap']['components']['cost']==-1
    assert sum(x for x in ev['gap']['components'].values() if x is not None)==pytest.approx(ev['gap']['total'])
    assert ev['forecast']['status']=='not_available'
    assert ev['whys'][1]['status']=='unresolved'
    assert all(not a['automatic'] for a in ev['feedback'])


def test_unreconciled_statement_or_missing_actuals_never_produce_a_narrative():
    pub,version,records,market,req=inputs()
    records[0]['ledger']['payload']['end_value']=1040
    with pytest.raises(FinplanError,match='does not reconcile'):decompose(pub,version,records,market,req)
    with pytest.raises(FinplanError,match='account ledger'):decompose(pub,version,[],market,req)
    with pytest.raises(FinplanError,match='checksum'):decompose(pub,{**version,'checksum':'changed'},records,market,req)
