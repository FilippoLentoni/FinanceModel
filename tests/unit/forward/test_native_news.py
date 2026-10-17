import copy
from types import SimpleNamespace
import pytest
from finplan_model.forward.native_news import read_native_news
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from tests.unit.test_etf_news import data

SYMBOLS=['AAPL','AMZN','GOOGL','META','MSFT','NVDA','TSLA','VOO']
CUTOFF='2026-01-02T14:00:00Z'


def response(symbols):
    result=data();result['symbols']=list(symbols)
    return result


def test_eight_symbol_queries_remain_bounded_and_identical_deliveries_are_deduplicated():
    calls=[]
    def get(symbols,at):calls.append((symbols,at));return response(symbols)
    out=read_native_news(SimpleNamespace(get_research_news=get),SYMBOLS,CUTOFF)
    assert calls==[(SYMBOLS[:5],CUTOFF),(SYMBOLS[5:],CUTOFF)]
    assert len(out['deliveries'])==1 and out['query_status']=='read'
    assert len(out['query_coverage'])==8 and all(r['query_status']=='read' for r in out['query_coverage'])
    assert all(b['response_checksum']==sha256_checksum(canonical_json_bytes(b['response'])) for b in out['query_batches'])
    assert out['deliveries'][0]==data()['deliveries'][0]


def test_partial_failure_retains_first_original_metadata_and_explicit_failed_symbols():
    def get(symbols,at):
        if len(symbols)==3:raise RuntimeError('stored query unavailable')
        return response(symbols)
    out=read_native_news(SimpleNamespace(get_research_news=get),SYMBOLS,CUTOFF)
    assert len(out['deliveries'])==1 and out['query_status']=='partial_failure'
    assert out['query_failures'][0]['symbols']==SYMBOLS[5:]
    assert all(r['query_status']=='failed' for r in out['query_coverage'][5:])


@pytest.mark.parametrize('corruption',['symbols','cutoff','future_available','schema'])
def test_invalid_response_never_enters_native_merged_metadata(corruption):
    def get(symbols,at):
        row=response(symbols)
        if corruption=='symbols':row['symbols']=['SPY']
        if corruption=='cutoff':row['decision_at']='2026-01-02T15:00:00Z'
        if corruption=='future_available':row['deliveries'][0]['available_at']='2026-01-02T15:00:00Z'
        if corruption=='schema':row['contract_version']=123
        return row
    out=read_native_news(SimpleNamespace(get_research_news=get),SYMBOLS,CUTOFF)
    assert out['deliveries']==[] and out['query_status']=='failed' and len(out['query_failures'])==2
    assert all(b['response'] for b in out['query_batches'])


def test_conflicting_delivery_identity_keeps_original_and_marks_later_batch_failed():
    calls=[]
    def get(symbols,at):
        row=response(symbols);calls.append(symbols)
        if len(calls)>1:row['deliveries'][0]['records'][0]['title']='Conflicting content'
        return row
    out=read_native_news(SimpleNamespace(get_research_news=get),SYMBOLS,CUTOFF)
    assert out['deliveries']==data()['deliveries'] and out['query_status']=='partial_failure'
    assert 'conflicting' in out['query_failures'][0]['error']


def test_equivalent_offset_cutoff_is_accepted_without_rewriting_original_response():
    def get(symbols,at):return response(symbols)
    out=read_native_news(SimpleNamespace(get_research_news=get),['AAPL'],'2026-01-02T09:00:00-05:00')
    assert out['query_status']=='read'
    assert out['query_batches'][0]['response']['decision_at']==CUTOFF
    assert out['decision_at']=='2026-01-02T09:00:00-05:00'



def test_successful_empty_query_is_visible_separately_from_other_failed_batch():
    def get(symbols,at):
        if len(symbols)==3:raise RuntimeError('stored query unavailable')
        row=response(symbols);row['deliveries']=[];return row
    out=read_native_news(SimpleNamespace(get_research_news=get),SYMBOLS,CUTOFF)
    assert out['query_status']=='partial_failure' and out['deliveries']==[]
    assert [b['status'] for b in out['query_batches']]==['read','failed']
    assert all(r['query_status']=='read' for r in out['query_coverage'][:5])
    assert all(r['returned_delivery_ids']==[] for r in out['query_coverage'])
