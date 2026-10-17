"""Operator-only forward corpus checkpoint; never schedules jobs or promotes actors."""
from __future__ import annotations
import copy
from datetime import datetime, timedelta, UTC
from pathlib import Path
import json
import time
import math
from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.etf_news import dt, extract
from finplan_model.rl.etf_news import FrozenEtfNewsProvider

FIRST_CUTOFF = '2026-10-30T20:15:00Z'


def freeze_corpus(client, symbols, end, days=14):
    """Union bounded seven-day API reads; retain original provenance and availability."""
    if days != 14:
        raise ValueError('this checkpoint requires fourteen calendar days')
    finish=dt(end); start=finish-timedelta(days=days); deliveries={}; reads=[]; template=None
    for cutoff in (start+timedelta(days=6), start+timedelta(days=12), finish):
        doc=client.get_research_news(symbols, cutoff.isoformat())
        template=copy.deepcopy(doc)
        rows=doc['deliveries']; reads.append({'decision_at':cutoff.isoformat(),'count':len(rows),'checksum':sha256_checksum(canonical_json_bytes(doc))})
        if len(rows)>=168:
            raise ValueError('bounded API read saturated; completeness cannot be established')
        for row in rows:
            if not start<=dt(row['available_at'])<=min(finish,cutoff):
                continue
            key=row['record_id']
            if key in deliveries and canonical_json_bytes(deliveries[key])!=canonical_json_bytes(row):
                raise ValueError('conflicting delivery revision for immutable record_id')
            deliveries[key]=copy.deepcopy(row)
    template.update(symbols=list(symbols),decision_at=finish.isoformat(),deliveries=sorted(deliveries.values(),key=lambda r:(r['available_at'],r['record_id'])))
    # Validation is deliberately performed on the final union before writing or training.
    provider=FrozenEtfNewsProvider(template)
    return template, {'start':start.isoformat(),'end':finish.isoformat(),'reads':reads,'delivery_count':len(deliveries),'checksum':provider.corpus_checksum}


def prepare(dataset, prices, *, now=None):
    """Price bundle is operator-frozen approved snapshot arrays, never synthetic backfill."""
    now=now or datetime.now(UTC)
    symbols=prices['instruments']; times=[dt(v) for v in prices['decision_times']]
    if not times or times!=sorted(set(times)):raise ValueError('price cutoffs must be unique and ordered')
    if not symbols or len(symbols)!=len(set(symbols)):raise ValueError('price universe must be unique')
    if any(t>now for t in times):raise ValueError('future price observations are forbidden')
    for field in ('close','exec_price','volume','has_bar'):
        if len(prices[field])!=len(times) or any(len(row)!=len(symbols) for row in prices[field]):raise ValueError('unaligned price array: '+field)
    start=dt(prices['news_start']); end=dt(prices['checkpoint_at'])
    if end-start!=timedelta(days=14):raise ValueError('news interval must be fourteen calendar days')
    indices=[i for i,t in enumerate(times) if start<=t<=end]
    blockers=[]; coverage=[]
    if now<end:blockers.append('checkpoint_not_due')
    if prices.get('snapshot_status')!='approved' or not prices.get('input_snapshot_id') or not prices.get('snapshot_checksum'):blockers.append('approved_price_snapshot_required')
    if prices.get('synthetic') is not False:blockers.append('real_price_provenance_required')
    if len(indices)<10:blockers.append('insufficient_forward_sessions')
    window=60
    if not indices or indices[0]<window:blockers.append('insufficient_price_warmup')
    for i in indices:
        cutoff=times[i].isoformat()
        infer,_,_=extract(dataset,symbols=symbols,decision_time=cutoff)
        train,_,_=extract(dataset,symbols=symbols,decision_time=cutoff,training=True)
        coverage.append({'decision_at':cutoff,'inference_missing':{s:infer[s]['missing'] for s in symbols},'training_missing':{s:train[s]['missing'] for s in symbols},'event_count':{s:infer[s]['event_count'] for s in symbols}})
    if any(any(r['training_missing'].values()) for r in coverage):
        statuses={r['rights']['status'] for r in dataset['deliveries']}
        blockers.append('rights_review_pending' if 'research_only' in statuses else 'training_rights_or_coverage_missing')
    if any(any(r['inference_missing'].values()) for r in coverage):blockers.append('forward_news_coverage_missing')
    plan={'schema':'finplan.ppo_forward_checkpoint/1','status':'blocked' if blockers else 'ready','blockers':sorted(set(blockers)),
          'checkpoint_at':end.isoformat(),'corpus_checksum':FrozenEtfNewsProvider(dataset).corpus_checksum,
          'price_checksum':sha256_checksum(canonical_json_bytes(prices)),'input_snapshot_id':prices.get('input_snapshot_id'),
          'instruments':symbols,'sessions':len(indices),'coverage':coverage,'seeds':[0,1,2,3,4],
          'features':['log_return_window','current_weights','etf_news_metadata_v1'],'window':window,
          'hyperparameters':{'total_timesteps':256,'n_steps':32,'batch_size':16,'n_epochs':2,'eval_every':128,'patience':2,'net_arch':[32,32]},
          'max_runtime_seconds':900,'budget_scope':'all ten learner/export operations combined, including overhead','promotion':'never_automatic','performance_status':'short_history_candidate_not_validated',
          'matched_fields':['price_checksum','instruments','simulation','windows','seeds','hyperparameters'],
          'rights_note':'research_only is a conservative review status, not an asserted provider blanket training ban'}
    if len(indices)>=10 and indices[0]>=window:
        first=indices[0]; n=len(indices); train_n=n-4
        plan['windows']={'train':[first-window,first+train_n-1],'validation':[first+train_n,first+train_n+1],'test':[first+train_n+2,indices[-1]]}
    return plan


def execute(plan, dataset, prices, budget, output, *, now=None):
    """Explicit local CPU training/export using existing learners; no AWS submission/promotion.

    Fresh operator budget evidence reserves the whole matched run under existing caps.
    No learner is imported before all data and budget checks have passed.
    """
    now=now or datetime.now(UTC)
    refreshed=prepare(dataset,prices,now=now)
    if canonical_json_bytes(refreshed)!=canonical_json_bytes(plan):raise ValueError('checkpoint plan changed; freeze again')
    if plan['blockers']:return {**plan,'jobs_submitted':0,'actors_exported':0}
    def number(key):
        value=budget[key]
        if isinstance(value,bool) or not math.isfinite(float(value)) or float(value)<0:raise ValueError('invalid budget evidence')
        return float(value)
    age=(now-dt(budget['checked_at'])).total_seconds()
    estimate=number('estimated_total_usd_upper_bound')
    if not 0<=age<=3600 or not budget.get('estimate_basis') or estimate<=0 or estimate>.5 or number('project_spent_usd')+number('project_reserved_usd')+estimate>100 or number('weekly_spent_usd')+number('weekly_reserved_usd')+estimate>.5:
        raise ValueError('fresh bounded project/weekly budget reservation required')
    # Reserve-aware local execution is deliberately operator serial; concurrency must be
    # excluded by the caller's existing research lease, attested in the budget evidence.
    if budget.get('exclusive_research_lease') is not True:raise ValueError('exclusive research lease required')
    import numpy as np
    from .arrays import PriceArrays
    from .spec import EnvSpec
    from .train import train_policy, preflight_news_training
    from .inference_v2 import export_actor
    from finplan_model.sim.config import SimulationConfig
    arrays=PriceArrays(tuple(dt(t).date() for t in prices['decision_times']),tuple(prices['instruments']),np.array(prices['close']),np.array(prices['exec_price']),np.array(prices['volume']),np.array(prices['has_bar'],dtype=bool),tuple(dt(t) for t in prices['decision_times']))
    config=SimulationConfig.from_dict(prices['simulation']); provider=FrozenEtfNewsProvider(dataset)
    spec=EnvSpec.from_dict({'window':60,'features':plan['features'],'step_sessions':1,'decision_frequency':'daily'})
    windows=plan['windows'];preflight_news_training(arrays,spec,train_window=windows['train'],validation_window=windows['validation'],news_provider=provider)
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True);results=[];deadline=time.monotonic()+900
    for branch in ('price_only','news_enriched'):
        selected=spec if branch=='news_enriched' else EnvSpec.from_dict({'window':60,'features':['log_return_window','current_weights'],'step_sessions':1,'decision_frequency':'daily'})
        for seed in plan['seeds']:
            if time.monotonic()>=deadline:raise TimeoutError('matched checkpoint runtime exhausted; no promotion')
            cid=sha256_checksum(canonical_json_bytes({'plan':plan,'branch':branch,'seed':seed}))
            policy=train_policy(arrays,selected,config,algorithm='ppo',seed=seed,hyperparameters=plan['hyperparameters'],train_window=windows['train'],validation_window=windows['validation'],configuration_id=cid,deadline=deadline,news_provider=provider if branch=='news_enriched' else None)
            actor=export_actor(policy.model_bytes,'ppo')
            path=directory/f'{branch}-seed{seed}'
            path.with_suffix('.zip').write_bytes(policy.model_bytes)
            path.with_suffix('.json').write_text(json.dumps({'actor':actor,'env':selected.document(),'instruments':prices['instruments'],'corpus_checksum':plan['corpus_checksum'],'configuration_id':cid},allow_nan=False))
            results.append({'branch':branch,**policy.summary()})
    return {**plan,'status':'candidate_exported','jobs_submitted':0,'actors_exported':len(results),'results':results,'test_status':'reserved_unopened_requires_common_evaluator','serving_promoted':False}
