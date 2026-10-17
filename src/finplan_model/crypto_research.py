"""Forward hourly crypto paper research. No wallet or exchange execution code."""
from __future__ import annotations
import copy
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
import numpy as np
from .classical.math import settings, solve, objective
from scipy.optimize import minimize_scalar
from .classical.storage import public
from .core.artifacts import canonical_json_bytes, sha256_checksum
from .core.errors import FinplanError
from .core.outcome import require_valid

PRODUCTS = ['BTC-USD','ETH-USD']
ALGORITHMS = ['min_variance','mean_variance','cvar']
BOOKS = ['cash','btc_buy_hold','equal_buy_hold',*ALGORITHMS]
FEE = .006
SLIPPAGE = .001
IMPLEMENTATION = 'crypto-hourly-paper-v1'


def dt(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(UTC)


def iso(value):
    return value.astimezone(UTC).isoformat().replace('+00:00','Z')


def validate_dataset(doc, now):
    require_valid(doc,'api/crypto-dataset-response')
    end, available = dt(doc['as_of']), dt(doc['available_at'])
    if end > available or available > now or (now-available).total_seconds()>5400 or doc['products'] != PRODUCTS:
        raise FinplanError.precondition('crypto snapshot is stale, future dated or incompatible',reason='crypto_snapshot_timing')
    starts = None
    for product in PRODUCTS:
        rows=doc['bars'][product]
        times=[dt(row['start']) for row in rows]
        if times[-1]+timedelta(hours=1)!=end or any(b-a!=timedelta(hours=1) for a,b in zip(times,times[1:])) or (starts is not None and times!=starts):
            raise FinplanError.precondition('crypto candles are incomplete or unaligned',reason='crypto_candle_gap')
        if any(min(r['open'],r['close'],r['low'])<=0 or not r['low']<=min(r['open'],r['close'])<=max(r['open'],r['close'])<=r['high'] for r in rows):
            raise FinplanError.precondition('crypto candles contain invalid prices',reason='crypto_prices')
        starts=times
    return doc


def recommendations(dataset, current_books=None):
    px=np.array([[r['close'] for r in dataset['bars'][p]] for p in PRODUCTS],dtype=float).T
    returns=px[1:]/px[:-1]-1
    cov=np.cov(returns,rowvar=False)+np.eye(2)*1e-10
    cfg=settings({'lookback_days':min(167,len(returns)), 'horizon_sessions':1,'cash_weight':.1,'max_weight':.6,'turnover_penalty':.001})
    out={}
    for algorithm in ALGORITHMS:
        inputs={'algorithm':algorithm,'instruments':PRODUCTS,'settings':cfg,'current_weights':([current_books[algorithm]['quantities'][p]*dataset['bars'][p][-1]['close']/(current_books[algorithm]['cash']+sum(current_books[algorithm]['quantities'][q]*dataset['bars'][q][-1]['close'] for q in PRODUCTS)) for p in PRODUCTS] if current_books else [.45,.45]),
                'covariance':cov.tolist(),'expected_returns':returns.mean(axis=0).tolist(),'scenarios':returns.tolist()}
        result=solve(inputs)
        status=result['status']
        if status not in ('optimal','feasible'):
            # Two-asset fixed-cash feasible region is a convex interval. The
            # bounded scalar solve also checks endpoints and turnover kink.
            fit=minimize_scalar(lambda x:-objective(inputs,[float(x),.9-float(x)]),bounds=(.3,.6),method='bounded',options={'xatol':1e-12})
            candidates=[.3,.45,.6]+([float(fit.x)] if fit.success else [])
            x=max(candidates,key=lambda x:objective(inputs,[x,.9-x]))
            if np.isfinite(objective(inputs,[x,.9-x])):
                result={'status':'feasible','weights':[x,.9-x]};status='feasible'
        if status not in ('optimal','feasible'):
            out[algorithm]={'eligible':False,'reason':'solver_'+status};continue
        weights=result['weights']
        if isinstance(weights,dict): weights=[weights[p] for p in PRODUCTS]
        out[algorithm]={'eligible':True,'weights':{p:float(w) for p,w in zip(PRODUCTS,weights)},'cash_weight':1-float(sum(weights)),
                        'decision_as_of':dataset['as_of'],'status':status,'policy_id':IMPLEMENTATION+'/'+algorithm,'lookback_hours':len(px),
                        'model_assumptions':'historical hourly returns; fixed 10% cash, 60% per-asset cap; not a calibrated forecast'}
    out['ppo']={'eligible':False,'reason':'no_validated_hourly_crypto_policy; daily equity PPO is incompatible'}
    out['llm']={'eligible':False,'reason':'no_frozen_validated_crypto_llm_policy; conversational explanation is not a trading-policy benchmark'}
    return out


def new_book():
    return {'cash':100.0,'quantities':{p:0.0 for p in PRODUCTS},'value':100.0,'peak':100.0,'max_drawdown':0.0,'cumulative_fees':0.0,'cumulative_slippage':0.0,'traded_notional':0.0,'pending':[],'fills':[]}


def rebalance(book, weights, opens, when):
    """Sell first; scale buys to available cash including modeled fees and slippage."""
    value=book['cash']+sum(book['quantities'][p]*opens[p] for p in PRODUCTS)
    targets={p:value*weights.get(p,0.0) for p in PRODUCTS}
    changes={p:targets[p]-book['quantities'][p]*opens[p] for p in PRODUCTS}
    fills=[]
    for p in PRODUCTS:
        if changes[p]>=-1e-8:continue
        q=min(book['quantities'][p],-changes[p]/opens[p]);price=opens[p]*(1-SLIPPAGE);notional=q*price;fee=notional*FEE
        book['quantities'][p]-=q;book['cash']+=notional-fee
        fills.append({'product':p,'side':'sell','quantity':q,'price':price,'fee':fee,'slippage_cost':q*opens[p]*SLIPPAGE,'notional':notional,'bar_start':when})
    needed=sum(max(0,x)*(1+SLIPPAGE)*(1+FEE) for x in changes.values())
    scale=min(1.0,max(0,book['cash'])/needed) if needed else 0
    for p in PRODUCTS:
        if changes[p]<=1e-8:continue
        q=changes[p]/opens[p]*scale;price=opens[p]*(1+SLIPPAGE);notional=q*price;fee=notional*FEE
        book['quantities'][p]+=q;book['cash']-=notional+fee
        fills.append({'product':p,'side':'buy','quantity':q,'price':price,'fee':fee,'slippage_cost':q*opens[p]*SLIPPAGE,'notional':notional,'bar_start':when})
    if book['cash'] < -1e-7 or any(q< -1e-10 for q in book['quantities'].values()):raise ValueError('paper book violated cash/long-only constraints')
    book['cash']=max(0,book['cash'])
    for fill in fills:
        book['cumulative_fees']+=fill['fee'];book['cumulative_slippage']+=fill['slippage_cost'];book['traded_notional']+=fill['notional']
    return fills


def advance(previous, dataset, plans, decision_time):
    books=copy.deepcopy(previous['books']) if previous else {name:new_book() for name in BOOKS}
    if previous and dt(dataset['as_of'])<=dt(previous['as_of']):raise ValueError('paper book cannot move backwards')
    times=[r['start'] for r in dataset['bars'][PRODUCTS[0]]]
    for name,book in books.items():
        book['fills']=[]
        remaining=[]
        for pending in book['pending']:
            candidates=[(i,t) for i,t in enumerate(times) if dt(t)>dt(pending['available_at']) and (not previous or dt(t)>=dt(previous['as_of']))]
            if not candidates:
                remaining.append(pending);continue
            i,t=candidates[0]
            next_hour=dt(pending['available_at']).replace(minute=0,second=0,microsecond=0)+timedelta(hours=1)
            if dt(t)!=next_hour:
                raise FinplanError.precondition('missing next-bar execution evidence',reason='crypto_fill_gap')
            book['fills'].extend(rebalance(book,pending['weights'],{p:dataset['bars'][p][i]['open'] for p in PRODUCTS},t))
        book['pending']=remaining
        closes={p:dataset['bars'][p][-1]['close'] for p in PRODUCTS}
        book['value']=book['cash']+sum(book['quantities'][p]*closes[p] for p in PRODUCTS)
        book['peak']=max(book['peak'],book['value']);book['max_drawdown']=max(book['max_drawdown'],1-book['value']/book['peak'])
        book['net_return']=book['value']/100-1;book['turnover']=book['traded_notional']/100
        weights=None
        if name in ALGORITHMS and plans[name]['eligible']:weights=plans[name]['weights']
        elif not previous and name=='btc_buy_hold':weights={'BTC-USD':1.0,'ETH-USD':0.0}
        elif not previous and name=='equal_buy_hold':weights={'BTC-USD':.5,'ETH-USD':.5}
        if weights is not None:book['pending'].append({'weights':weights,'available_at':iso(decision_time),'decision_as_of':dataset['as_of']})
    return books


def run_hour(service, *, scheduled_at=None):
    if service.env!='beta':raise FinplanError.precondition('crypto research is beta only',reason='crypto_disabled')
    budget=service.d.project_budget()
    if not np.isfinite([budget['spent'],budget['limit']]).all() or budget['spent']+5>min(100,budget['limit']):
        raise FinplanError.precondition('crypto research lacks project headroom',reason='crypto_budget')
    dataset=validate_dataset(service.d.platform.get_crypto_dataset(),service.now())
    if scheduled_at and dt(dataset['as_of']) != dt(scheduled_at).replace(minute=0,second=0,microsecond=0):
        raise FinplanError.precondition('scheduled slot lacks its matching crypto dataset',reason='crypto_schedule_slot_mismatch')
    slot=dt(dataset['as_of']).strftime('%Y%m%d%H');claim_key='crypto_hourly/slot/'+slot
    existing=service.store.get_claim(claim_key)
    if existing:return public(service.store.get(existing['analysis_id']))
    # Recover a committed record even if a prior invocation stopped before
    # publishing its slot claim/index; identity is fixed by slot and version.
    from .classical.storage import analysis_id
    expected=analysis_id({'kind':'crypto_experiment','identity':{'slot':slot,'implementation':IMPLEMENTATION}})
    try:
        recovered=service.store.get(expected)
    except FinplanError as exc:
        if exc.code!='NOT_FOUND':raise
    else:
        recovered=service.store.put(recovered)  # repair a record whose index write failed
        service.store.claim(claim_key,{'analysis_id':expected})
        return public(recovered)
    rows=service.store.list(kind='crypto_experiment',limit=1)
    previous=rows[0]['experiment'] if rows else None
    if previous and dt(dataset['as_of'])<=dt(previous['as_of']):
        raise FinplanError.precondition('out-of-order crypto research run',reason='crypto_out_of_order')
    now=service.now()
    books=advance(previous,dataset,{name:{'eligible':False} for name in ALGORITHMS},now)
    plans=recommendations(dataset,books)
    now=service.now()  # policy formation time follows the numerical solves
    if previous is None:
        for book in books.values():
            for pending in book['pending']:pending['available_at']=iso(now)
    for name in ALGORITHMS:
        if plans[name]['eligible']:
            books[name]['pending'].append({'weights':plans[name]['weights'],'available_at':iso(now),'decision_as_of':dataset['as_of']})
    exp={'experiment_id':IMPLEMENTATION,'as_of':dataset['as_of'],'started_at':previous['started_at'] if previous else iso(now),
         'initial_capital_usd':100,'synthetic':True,'market_data_synthetic':dataset['market_data_synthetic'],'observation_count':previous['observation_count']+1 if previous else 1,
         'parent_analysis_id':rows[0]['analysis_id'] if rows else None,'dataset_checksum':dataset['dataset_checksum'],'dataset_available_at':dataset['available_at'],
         'books':books,'models':plans,'cost_assumptions':{'fee_bps':60,'slippage_bps':10,'cash_return':0},
         'supplemental_data':{'status':'baseline_free_data','paid_evidence_count':0,'historical_news_training_eligible':False},
         'limitations':['Forward paper simulation; no guarantee of live returns.','Synthetic opens approximate fills; no order-book liquidity model.','Volatility and Sharpe require a sufficiently long comparable history and are not reported at initialization.'],
         'implementation_checksum':sha256_checksum(Path(__file__).read_bytes())}
    result=service.issue('crypto_experiment',{'summary':'Hourly synthetic USD 100 crypto portfolios; cash and buy-and-hold controls; no live orders.','experiment':exp},
                         {'slot':slot,'implementation':IMPLEMENTATION})
    if not service.store.claim(claim_key,{'analysis_id':result['analysis_id']}):
        return public(service.store.get(service.store.get_claim(claim_key)['analysis_id']))
    return result


def handle(service,operation,body):
    if service.env!='beta':raise FinplanError.precondition('crypto research is beta only',reason='crypto_disabled')
    if operation=='list_crypto_history':
        rows=service.store.list(kind='crypto_experiment',limit=body.get('limit',12))
        return service.issue('crypto_history',{'summary':'Immutable forward crypto paper portfolio observations.','history':[{'analysis_id':r['analysis_id'],'as_of':r['experiment']['as_of'],'books':{n:{k:b[k] for k in ['value','net_return','max_drawdown','turnover','cumulative_fees','cumulative_slippage']} for n,b in r['experiment']['books'].items()}} for r in rows]}, {'ids':[r['analysis_id'] for r in rows]})
    if operation=='discover_research_data':
        from .research_discovery import discover
        result=discover(body.get('query','crypto market news and research'))
        import re
        raw=result.pop('untrusted_catalog_evidence')
        result.pop('source',None)
        result['catalog_evidence_checksum']=sha256_checksum(canonical_json_bytes(raw))
        import json
        resources=[]
        for block in raw.get('content',[]):
            try:catalog=json.loads(block.get('text',''))
            except (ValueError,AttributeError):continue
            if not isinstance(catalog,dict):continue
            for item in catalog.get('tools',[])[:3]:
                if not isinstance(item,dict):continue
                meta=item.get('_meta',{})
                accepts=meta.get('x402/payment-required',{}).get('accepts',[])
                description=re.sub(r'https?://[^\s"<>]+','[endpoint withheld]',str(item.get('description','')))[:400]
                resources.append({'name':str(item.get('name','unnamed'))[:120], 'description':description,
                    'observed_price_usd':str(meta.get('x402/guidance',{}).get('costObservedUsd','unknown'))[:30],
                    'networks':[str(a.get('network',''))[:64] for a in accepts[:3] if isinstance(a,dict)],
                    'metadata_trust':'unverified_catalog', 'payment_eligible':False})
        result['resources']=resources[:3]
        result['catalog_summary']='\n'.join(r['name']+': '+r['description']+'; observed catalog price USD '+r['observed_price_usd'] for r in resources[:3]) or 'No usable bounded catalog entries were returned.' 
        result['purchase_status']='operator_approval_required; hosted signing disabled'
        return service.issue('research_discovery',{'summary':'Free Coinbase x402 Bazaar discovery; purchases require the separate operator approval path.','discovery':result},{'query':body.get('query'),'at':iso(service.now())})
    rows=service.store.list(kind='crypto_experiment',limit=1)
    if not rows:raise FinplanError.precondition('hourly crypto experiment is not initialized',reason='crypto_not_initialized')
    doc=rows[0]
    if operation=='get_crypto_experiment':return public(doc)
    if (service.now()-dt(doc['experiment']['dataset_available_at'])).total_seconds()>5400:
        raise FinplanError.precondition('latest crypto recommendation is stale',reason='crypto_stale_recommendation')
    algorithm=body.get('algorithm','min_variance')
    plan=doc['experiment']['models'][algorithm]
    return service.issue('crypto_recommendation',{'summary':'Crypto paper recommendation; next observed open only, no live orders.','recommendation':{'algorithm':algorithm,**plan,'source_analysis_id':doc['analysis_id'],'as_of':doc['experiment']['as_of'],'synthetic':True}}, {'source':doc['analysis_id'],'algorithm':algorithm})
