"""Bounded reads of already stored licensed metadata, preserving original API contracts.

This adapter never calls a news provider. Each platform request obeys the five
symbol contract. Its explicit aggregate is preprocessing evidence, not a claim
that an eight-symbol API response existed or that missing coverage was neutral.
"""
from __future__ import annotations
import copy
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.core.outcome import require_valid
from finplan_model.etf_news import dt


def read_native_news(client, symbols, decision_at):
    symbols=list(symbols);cutoff=dt(decision_at)
    if not 1<=len(symbols)<=8 or len(set(symbols))!=len(symbols):
        raise ValueError('native news cohort must contain1 to8 unique symbols')
    batches=[];deliveries={};failures=[];coverage=[]
    for offset in range(0,len(symbols),5):
        selected=symbols[offset:offset+5]
        batch={'symbols':selected,'decision_at':decision_at,'status':'failed','schema':'api/research-news-response'}
        response=None
        try:
            response=client.get_research_news(selected,decision_at)
            raw=canonical_json_bytes(response)
            batch.update(response_checksum=sha256_checksum(raw),response=copy.deepcopy(response))
            require_valid(response,'api/research-news-response')
            if set(response['symbols'])!=set(selected) or dt(response['decision_at'])!=cutoff:
                raise ValueError('stored news response symbols/cutoff mismatch')
            additions={}
            for delivery in response['deliveries']:
                if dt(delivery['available_at'])>cutoff or dt(delivery['retrieved_at'])>dt(delivery['available_at']):
                    raise ValueError('stored news delivery unavailable at shared cutoff')
                rid=delivery['record_id']
                prior=additions.get(rid,deliveries.get(rid))
                if prior is not None and canonical_json_bytes(prior)!=canonical_json_bytes(delivery):
                    raise ValueError('conflicting immutable news delivery ID')
                additions[rid]=copy.deepcopy(delivery)
            deliveries.update(additions)
            batch.update(status='read',delivery_count=len(response['deliveries']))
        except Exception as exc:
            batch['error']=type(exc).__name__+': '+str(exc)
            failures.append({'symbols':selected,'error':batch['error']})
        for symbol in selected:
            coverage.append({'symbol':symbol,'query_status':batch['status'],
                'returned_delivery_ids':[d['record_id'] for d in (response or {}).get('deliveries',[]) if symbol in d.get('symbols',[])] if batch['status']=='read' else [],
                'coverage_meaning':'stored metadata query result; eligible model coverage determined by native rights/time extractor'})
        batches.append(batch)
    return {'format':'finplan.batched-native-news/1','adapter_version':'bounded-platform-news-read/1',
            'decision_at':decision_at,'symbols':symbols,'deliveries':sorted(deliveries.values(),key=lambda d:(dt(d['available_at']),d['record_id'])),
            'historical_coverage':'forward_collection_only','query_batches':batches,'query_failures':failures,'query_coverage':coverage,
            'query_status':'partial_failure' if failures and any(b['status']=='read' for b in batches) else 'failed' if failures else 'read',
            'transformation':'validated platform responses; deduplicate identical record_id, never synthesize text or metadata'}
