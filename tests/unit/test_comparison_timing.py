import copy
import pytest
from finplan_model.comparison_timing import news_alignment
from finplan_model.core.errors import FinplanError

def rec(day='2026-10-17',checksum='a'):
    return {'as_of':'2026-10-16','decision_at':day+'T13:00:00Z','target_weights':[{'instrument_id':'AAPL','weight':.5}],
            'portfolio_state':{'portfolio_value':100,'cash_balance':100},'diagnostics':{'news_evidence':{'consumed':True,'feature_protocol':'synthetic-intraday','provenance':{'document_checksum':'sha256:'+checksum*64,'source_day':day,'window_start':day+'T11:45:00Z','window_end':day+'T12:45:00Z','lineage':{'source_available_at':day+'T12:45:00Z','central_available_at':day+'T12:50:00Z'}}}}}

def test_same_completed_price_and_different_real_formations_sources_are_temporal_news():
    a,b=rec(),rec('2026-10-18','b');out=news_alignment(a,b,same_model=True,same_snapshot=True)
    assert out['temporal_news_change'] and out['same_price_as_of'] and out['news_state_changed']
    assert not out['holdings_state_changed'] and not out['identical_state_replay']
    assert out['current_news_window']['window_end']=='2026-10-18T12:45:00Z'

def test_reversed_news_formation_rejected_but_same_cutoff_crossmethod_preserved():
    with pytest.raises(FinplanError,match='formation follows'):news_alignment(rec('2026-10-18'),rec(),same_model=True,same_snapshot=True)
    out=news_alignment(rec(),rec(),same_model=False,same_snapshot=True)
    assert not out['temporal_news_change'] and not out['identical_state_replay']

def test_same_state_replay_is_labelled_for_reproducibility_not_fake_temporal_history():
    out=news_alignment(rec(),rec(),same_model=True,same_snapshot=True)
    assert out['identical_state_replay'] and not out['temporal_news_change'] and not out['formation_changed']


def test_news_protocol_change_is_not_controlled_temporal_news_comparison():
    a,b=rec(),rec('2026-10-18','b');b['diagnostics']['news_evidence']['feature_protocol']='different-daily-source'
    out=news_alignment(a,b,same_model=True,same_snapshot=True)
    assert out['news_protocol_changed'] and out['news_state_changed'] and not out['temporal_news_change']
