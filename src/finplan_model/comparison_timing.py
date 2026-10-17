"""Distinguish immutable news formation, price date, holdings and model changes."""
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.etf_news import dt


def news_alignment(previous,current,*,same_model,same_snapshot,pointer='/previous_decision_id'):
    news=[r.get('diagnostics',{}).get('news_evidence',{}) for r in (previous,current)]
    formations=[r.get('decision_at') or n.get('decision_time') or n.get('formation_time') for r,n in zip((previous,current),news)]
    proofs=[n.get('provenance') if isinstance(n.get('provenance'),dict) else {} for n in news]
    identities=[p.get('document_checksum') or n.get('corpus_checksum') for p,n in zip(proofs,news)]
    consumed=all(n.get('consumed') is True for n in news)
    protocol_changed=news[0].get('feature_protocol')!=news[1].get('feature_protocol')
    if consumed and all(formations) and dt(formations[0])>dt(formations[1]):
        raise FinplanError.validation('previous news formation follows current formation',pointer=pointer,reason='reversed_news_formation')
    changed=all(identities) and identities[0]!=identities[1]
    time_changed=all(formations) and dt(formations[0])!=dt(formations[1])
    holdings=[sha256_checksum(canonical_json_bytes(r.get('portfolio_state',{}))) for r in (previous,current)]
    same_universe={r['instrument_id'] for r in previous['target_weights']}=={r['instrument_id'] for r in current['target_weights']}
    def window(p):
        lineage=p.get('lineage',{})
        return {'source_day':p.get('source_day'),'window_start':p.get('window_start') or lineage.get('window_start'),
                'window_end':p.get('window_end') or lineage.get('window_end'),'source_available_at':lineage.get('source_available_at'),
                'central_available_at':lineage.get('central_available_at'),'quality_status':p.get('quality_status')}
    return {'previous_decision_at':formations[0],'current_decision_at':formations[1],
            'previous_news_protocol':news[0].get('feature_protocol'),'current_news_protocol':news[1].get('feature_protocol'),
            'previous_news_source_checksum':identities[0],'current_news_source_checksum':identities[1],
            'previous_news_window':window(proofs[0]),'current_news_window':window(proofs[1]),
            'news_protocol_changed':protocol_changed,'news_state_changed':bool(consumed and changed),'formation_changed':bool(time_changed),
            'holdings_state_changed':holdings[0]!=holdings[1],'same_price_as_of':previous['as_of']==current['as_of'],
            'same_universe':same_universe,
            'temporal_news_change':bool(consumed and changed and time_changed and same_model and same_universe and not protocol_changed),
            'identical_state_replay':bool(consumed and all(formations) and all(identities) and not changed and not time_changed and same_model and same_snapshot and same_universe and not protocol_changed and holdings[0]==holdings[1])}
