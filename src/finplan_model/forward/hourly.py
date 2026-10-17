"""Durable EC2 beta hourly workflow. DynamoDB is authoritative, local files are exports.

Permanent slot keys and lease claims prevent repeated inference and repeated fills.
Collected source is frozen once. Missing historical slots are recorded as skipped,
never evaluated using today's collection. No training, paid vendor call or GPU launch.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import socket
import traceback
from datetime import UTC,datetime,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import boto3
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum,S3ArtifactStore
from finplan_model.core.platform import HttpPlatformClient,SnapshotReader
from finplan_model.jobs.market_loader import market_from_content
from .registry import registry,APPROACHES,Approach
from .runtime import evaluate_approach,input_market,validate_policy
from .research import review,stamp
from .forecasts import freeze,reconcile
from .paper import apply_next_session,apply_forward_quotes

DEFAULT_EXPERIMENT='beta-forward-hourly-v1'
REGION='us-east-2'
POLICY={'version':'beta-forward-hourly/2','initial_capital':100000.0,'base_currency':'USD','decision_opportunity':'hourly',
        'execution_timing':'next_hourly_regular_session_quote','execution_price_freshness_seconds':120,'market_calendar':'XNYS','initialization':'cash_only','fees':{'proportional_bps':1.0,'fixed_per_trade':0.0},
        'spread':{'half_spread_bps':1.0},'slippage':{'model':'linear_participation','coefficient_bps':10.0},
        'liquidity':{'participation_cap':0.05,'unfilled':'cancel'},'constraints':{'long_only':True,'min_weight':0.0,'max_weight':1.0,'min_cash':0.0,'max_cash':1.0},
        'cash_rate_annual':0.0,'report_timezone':'America/New_York','forecast_alignment':'first_valid_observation_at_or_after_target',
        'tie_break':'lower_drawdown_then_approach_id','research_mode':'forward_only','price_freshness_seconds':5400}


class NotClaimed(RuntimeError):pass


def evidence_backend():
    from finplan_platform.core.modeling_evidence import ModelingEvidence
    return ModelingEvidence.from_ec2(region=REGION)


def identifier(prefix,*parts):
    return prefix+'_'+hashlib.sha256(canonical_json_bytes(list(parts))).hexdigest()


def slot_now():return datetime.now(UTC).replace(minute=0,second=0,microsecond=0).isoformat().replace('+00:00','Z')


def policy_approach_id(bundle):
    algorithm=bundle.get('strategy_id','ppo')
    if algorithm not in ('ppo','sac'):raise ValueError('frozen export algorithm must be honest PPO or SAC')
    features=bundle.get('environment',{}).get('features',[])
    suffix='_gdelt_intraday' if 'gdelt_intraday_organization_tone_v1' in features else '_gdelt' if 'gdelt_organization_tone_v1' in features else '_etf_news' if 'etf_news_metadata_v1' in features else ''
    return algorithm+suffix

def load_current_inputs(session=None,cohort=None):
    """Approved snapshots through signed platform API; preserve raw and actor basis separately."""
    session=session or boto3.Session(region_name=REGION);ssm=session.client('ssm');s3=session.client('s3')
    param=lambda name:ssm.get_parameter(Name=name)['Parameter']['Value']
    setting=json.loads(param('/finplan/beta/financemodel/config/advisory-policy'))
    artifacts=S3ArtifactStore(s3,param('/finplan/beta/financemodel/config/research-storage-ref'))
    policy=json.loads(artifacts.get(setting['artifact']))
    if cohort=='eight_assets':
        listing=s3.list_objects_v2(Bucket=artifacts._bucket,Prefix='artifacts/policy_inference/',MaxKeys=100)
        if listing.get('IsTruncated'):raise ValueError('policy inventory exceeded bound')
        candidates=[]
        for row in listing.get('Contents',[]):
            if row['Size']>2_000_000:continue
            candidate=json.loads(s3.get_object(Bucket=artifacts._bucket,Key=row['Key'])['Body'].read())
            if len(candidate.get('instruments',[]))==8 and not any('news' in f or 'gdelt' in f for f in candidate.get('environment',{}).get('features',[])):candidates.append(candidate)
        if not candidates:raise ValueError('eight-asset frozen policy unavailable')
        policy=sorted(candidates,key=lambda p:p['configuration_id'])[0]
    endpoint=param('/finplan/beta/financialplanning/api/plan-endpoint')
    client=HttpPlatformClient(endpoint,region=REGION,credentials=session.get_credentials())
    latest=client.get_latest_snapshot('finance/equity-etf-daily/research-universe')['snapshot']
    content=SnapshotReader(client).load(latest['input_snapshot_id'])
    if content.snapshot.synthetic or content.payload.get('synthetic'):raise ValueError('forward experiment requires actual approved market source')
    market=market_from_content(content);universe=list(policy['instruments'])
    actual_available=content.snapshot.record['created_at']
    raw=[];feature=[]
    for row in content.payload['observations']:
        if row['instrument_id'] not in universe:continue
        raw.append({**row,'available_at':max(row.get('available_at',actual_available),actual_available),
                    'availability_basis':'actual_approved_snapshot_created_at; no historical vintage claim'})
    # Model-native adjustments are retained, not accidentally used as executable raw quotes.
    for instrument in universe:
        for b in market.bars_of(instrument):
            feature.append({'instrument_id':instrument,'session_date':b.session_date.isoformat(),'close':b.close,
                            'open':b.open,'high':b.high,'low':b.low,'volume':b.volume,'kind':b.kind,'available_at':actual_available,
                            'price_basis':(content.payload.get('universe') or {}).get('return_basis','close'),
                            'availability_basis':'actual_approved_snapshot_created_at; no historical vintage claim'})
    dates=sorted({r['session_date'] for r in feature})[-max(252,int(policy['environment'].get('window',60))+1):]
    feature=[r for r in feature if r['session_date'] in dates];raw=[r for r in raw if r['session_date'] in dates]
    source={'dataset_id':content.snapshot.dataset_id,'input_snapshot_id':content.snapshot.input_snapshot_id,
            'checksum':content.snapshot.manifest_checksum,'verified_artifacts':content.verified,
            'investable_universe':universe,'observations':feature,'execution_observations':raw,
            'snapshot_created_at':actual_available,'price_basis':(content.payload.get('universe') or {}).get('return_basis','close'),
            'limitations':['Fixed trained-policy investable universe; full S&P500 research is wider than this fair cohort',
                           'Historical price history first known here at actual snapshot creation, not a historical vintage'],
            'model_native_cadence':'completed_daily','trigger':os.environ.get('FINPLAN_EXPERIMENT_TRIGGER','manual_validation')}
    policies={policy_approach_id(policy):policy}
    listed=s3.list_objects_v2(Bucket=artifacts._bucket,Prefix='artifacts/policy_inference/',MaxKeys=100)
    if listed.get('IsTruncated'):raise ValueError('policy inventory exceeded bounded100 exports; explicit pagination required')
    exports=[]
    for item in listed.get('Contents',[]):
        if item['Size']>2_000_000:raise ValueError('frozen policy export exceeds bound')
        data=s3.get_object(Bucket=artifacts._bucket,Key=item['Key'])['Body'].read()
        exports.append({'artifact':{'artifact_id':item['Key'].rsplit('/',1)[-1],'kind':'policy_inference','checksum':sha256_checksum(data),'content_type':'application/json','size_bytes':len(data)}})
    # Each discovered exported frozen policy gets its own honest book and identity.
    for ref in [v['artifact'] for v in setting.get('portfolio_policies',{}).values()]+[r['artifact'] for r in exports]:
        bundle=json.loads(artifacts.get(ref))
        if bundle==policy:continue
        name=bundle.get('strategy_id','ppo')+'_policy_'+ref['checksum'].split(':')[-1][:12]
        policies[name]=bundle
    source['code_versions']={'financemodel':os.environ.get('FINPLAN_MODEL_SOURCE_COMMIT','workspace_commit_not_supplied'),
                             'financialplanning':os.environ.get('FINPLAN_PLATFORM_SOURCE_COMMIT','workspace_commit_not_supplied'),
                             'financeagent':os.environ.get('FINPLAN_AGENT_SOURCE_COMMIT','workspace_commit_not_supplied')}
    from finplan_model.etf_news import FINANCIAL_NEWS_UNIVERSE
    if set(universe)<=FINANCIAL_NEWS_UNIVERSE:
        try:
            source['paid_news_dataset']=client.get_research_news(universe,stamp())
        except Exception as exc:source['paid_news_source_status']={'status':'unavailable','reason':type(exc).__name__+': '+str(exc)}
    else:source['paid_news_source_status']={'status':'unavailable','reason':'native_supported_news_universe_excludes_requested_symbols'}
    source['native_policy_inputs']={}
    for aid,bundle in policies.items():
        if 'gdelt_intraday_organization_tone_v1' not in bundle.get('environment',{}).get('features',[]) or bundle['instruments']!=universe:continue
        try:
            from finplan_model.rl.gdelt_intraday import formation_cutoff,read_live
            native=formation_cutoff(stamp())
            dataset,features,provenance=read_live(client,universe,native)
            source['native_policy_inputs'][aid]={'status':'available','decision_time':native,'dataset':dataset,'features':features,'provenance':provenance,
              'cadence_status':'reused_exact_native_09:00_formation_inputs','acquired_at':stamp()}
        except Exception as exc:
            source['native_policy_inputs'][aid]={'status':'unavailable','reason':type(exc).__name__+': '+str(exc),
               'query':{'symbols':universe,'availability_mode':'platform','native_formation':'09:00 America/New_York'},'acquired_at':stamp()}
    source['frozen_policies']=copy.deepcopy(policies)
    return source,policies


def evaluator_config():
    from finplan_model.sim.config import SimulationConfig
    allowed=set(SimulationConfig.__dataclass_fields__)
    return {**{k:copy.deepcopy(v) for k,v in POLICY.items() if k in allowed and k!='execution_timing'},'execution_timing':'next_open','initial_cash':POLICY['initial_capital'],'rebalance_frequency':'daily'}

def _portfolio_holdings(book,source):
    state=book['paper_state'];marks={}
    for row in sorted(source.get('execution_observations',source.get('observations',[])),key=lambda r:r['session_date']):marks[row['instrument_id']]=row['close']
    shares={r['instrument_id']:r['quantity'] for r in state['positions']}
    cash=state['cash_balance'];missing=[i for i,q in shares.items() if q and i not in marks]
    value=None if missing else cash+sum(q*marks[i] for i,q in shares.items() if q)
    weights=[{'instrument_id':i,'weight':0.0 if not value else shares.get(i,0)*marks.get(i,0)/value} for i in source['investable_universe']]
    return {**copy.deepcopy(state),'cash':cash,'weights':weights,'cash_weight':1.0 if not value else cash/value,
            'portfolio_value':value,'high_watermark':max(state['high_watermark'],value or 0),'portfolio_id':book['portfolio_id'],'revision':book['revision']}


def _graph_adapter(base,source,holdings,before):
    from finplan_agent.modeling.hosted import invoke_graph
    arguments={'input_snapshot_id':source['input_snapshot_id'],'as_of':max(r['session_date'] for r in source['observations']),
               'holdings':{k:holdings[k] for k in ('weights','cash_weight','portfolio_value','high_watermark')},
               }
    response=invoke_graph(arguments,capture_before=before)
    # This is an orchestration approach: its numerical recommendation is actually served PPO.
    response['framework']='LangGraph';response['allocation_origin']='FinanceModel served frozen PPO via graph tool'
    return response


def _codex_adapter(output):
    def adapter(base,source,holdings,before):return review(source,holdings,Path(output)/'codex-policy.json',capture=before)
    return adapter


def _history(backend,experiment,day,kind='runs'):
    rows=[];token=None
    for _ in range(20):
        page,token=backend.query_day(experiment,day,kind=kind,page_size=100,next_token=token);rows.extend(page)
        if not token:return rows
    raise ValueError('bounded daily history exceeded; increase partition granularity before scaling')


def _current_news(backend,experiment,cutoff):
    # Interim research is stored in a distinct experiment; it cannot create primary fills.
    day=datetime.fromisoformat(cutoff.replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date()
    candidates=[]
    for d in (day-timedelta(days=1),day):
        for run in _history(backend,experiment+'-interim',d.isoformat()):
            if run['approach_id']=='direct_codex_news' and run.get('input_id'):
                captured=backend.read_input(run['input_id'])['payload']
                source=captured.get('research_source')
                if source and source.get('available_at','9999')<=cutoff:candidates.append(source)
    return max(candidates,key=lambda s:s['available_at']) if candidates else None


def _register(backend,experiment,approaches,source,at,slot):
    policy={**POLICY,'investable_universe':source['investable_universe'],'first_scheduled_slot':slot,'scheduled_start_at':slot,'cadence_seconds':3600,'decision_offset_seconds':300}
    old=backend.read_registry(experiment)
    if old:
        if {k:v for k,v in old['comparison_policy'].items() if k not in ('first_scheduled_slot','scheduled_start_at')}!={k:v for k,v in policy.items() if k not in ('first_scheduled_slot','scheduled_start_at')}:raise ValueError('existing experiment comparison policy is immutable; new experiment required')
        if [a['approach_id'] for a in old['approaches']]!=[a['approach_id'] for a in approaches]:
            raise ValueError('registry variants changed; create versioned experiment rather than replace books')
    else:backend.register_approaches(experiment,approaches,policy,at)


def record_slot_observation(backend,experiment,approach_id,slot,run_id,input_id,book,holdings,source,at,outcome):
    oid=identifier('mo',experiment,approach_id,slot,'valuation')
    if backend.repo.get('portfolio_observation',oid):return
    last_price=max((r['session_date'] for r in source.get('execution_observations',[])),default=None)
    payload={'run_id':run_id,'input_id':input_id,'portfolio_id':book['portfolio_id'],'observed_at':at,
             'value':holdings['portfolio_value'],'cash':holdings['cash'],'holdings':holdings['positions'],'revision':holdings.get('revision',book['revision']),
             'observation_kind':'pending' if source.get('historical_slot_missing') or last_price is None else 'stale',
             'quote_status':'missing' if last_price is None else 'stale_daily',
             'quote_provenance':{'input_snapshot_id':source.get('input_snapshot_id'),'checksum':source.get('checksum'),'price_as_of':last_price},
             'net_external_cash_flows_since_decision':0,'transaction_cost':0,'model_cost':outcome.get('usage',{}).get('cost_usd'),
             'reason':'No eligible fresh current quote; completed daily marks explicitly carried forward',
             'decision_status':outcome['status'],'execution_status':outcome.get('execution_status','not_applicable')}
    held={p['instrument_id'] for p in holdings['positions'] if p['quantity']}
    action_rows=[r for r in source.get('execution_observations',[]) if r['instrument_id'] in held and
                 r['session_date']>=holdings.get('as_of','0000') and (r.get('dividend') or r.get('split_ratio') not in (None,0,1))]
    payload['corporate_action_status']='unreconciled_captured_event' if action_rows else 'no_captured_daily_event; current_quote_event_coverage_separate'
    payload['corporate_action_evidence']=action_rows
    # Root current-price adapter returns real unadjusted quotes only for valuation. Never
    # copy these intraday values into frozen daily policy observations or solver history.
    quote_bundle=source.get('current_quotes') or {}
    quotes=quote_bundle.get('quotes',{})
    if isinstance(quotes,list):quotes={q['instrument_id']:q for q in quotes}
    positions={r['instrument_id']:r['quantity'] for r in holdings['positions']}
    needed=[i for i,q in positions.items() if q]
    if needed and all(i in quotes and quotes[i].get('status')=='fresh' and quotes[i].get('available_at','9999')<=at and quotes[i].get('quoted_at','9999')<=at and 0<=quotes[i].get('freshness_seconds',999999)<=POLICY['price_freshness_seconds'] for i in needed):
        value=holdings['cash']+sum(q*quotes[i]['price'] for i,q in positions.items() if q)
        payload.update(value=value,observation_kind='valid',quote_status='fresh',reason='Actual eligible current quote marks',
                       quote_provenance={i:quotes[i] for i in needed})
    elif not needed and not source.get('historical_slot_missing'):
        payload.update(value=holdings['cash'],observation_kind='valid',quote_status='fresh',
                       reason='Cash-only synthetic book; no market quote is required for its actual cash balance',
                       quote_provenance={'basis':'authoritative_paper_cash_balance','revision':holdings.get('revision',book['revision'])})
    if action_rows or any(quotes.get(i,{}).get('corporate_actions') for i in needed):
        payload.update(observation_kind='pending',quote_status='corporate_action_unreconciled',value=None,reason='Captured dividend/split requires authoritative action reconciliation; ranking blocked')
    backend.record_observation(experiment,approach_id,slot,book['portfolio_id'],payload,at)

def reconcile_durable(backend,experiment,source,at,slot):
    """Bounded seven-day pending fills and forecasts, preserving original predictions/inputs."""
    day=datetime.fromisoformat(at.replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date()
    runs=[];observations=[]
    for offset in range(8):
        partition=(day-timedelta(days=offset)).isoformat()
        runs.extend(_history(backend,experiment,partition))
        observations.extend(_history(backend,experiment,partition,'observations'))
    books={}
    for run in sorted(runs,key=lambda r:r['started_at'],reverse=True):
        outcome=run.get('outcome') or {};aid=run['approach_id']
        if run['slot']>=slot or aid in books or run.get('execution_ref') or outcome.get('status') not in ('succeeded','hold','reused'):continue
        book=backend.get_portfolio(run['portfolio_id'])
        if book['revision']!=run['portfolio_revision']:continue
        books[aid]=book
        for older in runs:
            if older['approach_id']!=aid or older['started_at']>=run['started_at'] or older.get('execution_ref') or (older.get('outcome') or {}).get('status') not in ('succeeded','hold','reused'):continue
            oid=identifier('mo',experiment,aid,older['slot'],'execution_disposition')
            if not backend.repo.get('portfolio_observation',oid):
                backend.record_observation(experiment,aid,older['slot'],older['portfolio_id'],{'run_id':older['run_id'],'status':'superseded',
                    'replacement_run_id':run['run_id'],'reason':'Latest common-hourly policy supersedes earlier pending policy before an eligible next-open fill',
                    'actual_fills':[],'observed_at':at,'observation_kind':'execution_disposition'},at,kind='execution_disposition')
        if not outcome.get('target_weights'):continue
        try:
            opportunity=(datetime.fromisoformat(slot.replace('Z','+00:00'))+timedelta(minutes=5)).isoformat().replace('+00:00','Z')
            result=apply_forward_quotes(outcome,book['paper_state'],source.get('current_quotes',{}),decision_at=run.get('finished_at',run['started_at']),
                opportunity_at=opportunity,recorded_at=at,config=evaluator_config(),universe=source['investable_universe'])
            if result['status'] in ('applied','no_op'):
                backend.commit_paper_step(run['run_id'],book['revision'],result['state'],result['fills'],result['execution'],stamp())
        except Exception as exc:
            # Persist a retryable execution failure separately; never rewrite the recommendation.
            backend.record_observation(experiment,aid,at,run['portfolio_id'],{'run_id':run['run_id'],'status':'execution_failed',
                'reason':type(exc).__name__+': '+str(exc),'observed_at':at,'value':None,'observation_kind':'pending','quote_status':'missing'},at,kind='execution_attempt')
    # Pending forecasts have a sparse durable index and are not lost when older than
    # a report window. Query actual observation time, not rounded scheduling slot.
    token=None
    for _ in range(100):
        pending,token=backend.query_pending_forecasts(experiment,at,page_size=100,next_token=token)
        for run in pending:
            frozen=(run.get('outcome') or {}).get('forecast') or {}
            oid=identifier('mo',experiment,run['approach_id'],run['slot'],'forecast_match')
            if backend.repo.get('portfolio_observation',oid):
                backend.mark_forecast_realized(run['run_id'],oid,at);continue
            values=[];cursor=None
            for _ in range(100):
                page,cursor=backend.query_observations_after(experiment,run['approach_id'],frozen['target_at'],at,page_size=100,next_token=cursor)
                for row in page:
                    if row.get('evidence_error'):continue
                    payload=row.get('payload') or {}
                    values.append({**payload,'observation_id':row['observation_id'],'portfolio_id':row['portfolio_id']})
                match=reconcile(frozen,values)
                if match:
                    saved=backend.record_observation(experiment,run['approach_id'],run['slot'],run['portfolio_id'],match,at,kind='forecast_match')
                    backend.mark_forecast_realized(run['run_id'],saved['observation_id'],at);break
                if not cursor:break
            if cursor and not match:raise ValueError('forecast observation query exceeded bounded10000 records; pending head retained for recovery')
        if not token:break
    if token:raise ValueError('pending forecast query exceeded bounded10000 records; heads retained for recovery')

def run_hourly(slot,experiment,output,*,backend=None,source=None,policies=None,research=None,now=None,external_adapters=None,account_missed=False,cohort=None):
    backend=backend or evidence_backend();at=now or stamp();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    deadline=datetime.fromisoformat(at.replace('Z','+00:00'));slot_dt=datetime.fromisoformat(slot.replace('Z','+00:00'))
    if slot_dt>deadline:raise ValueError('future scheduled slot forbidden')
    if slot_dt.minute or slot_dt.second:raise ValueError('hourly slot must be hour aligned')
    missed=account_missed or (deadline-slot_dt).total_seconds()>3600
    prior_shared_id=identifier('mi',experiment,'__shared__',slot,'shared_source')
    prior_shared=backend.repo.get('research_input',prior_shared_id)
    if prior_shared:
        source=backend.read_input(prior_shared_id)['payload'];policies=source.get('frozen_policies',policies or {})
        at=source['availability_cutoff']
    elif source is None:
        persisted_registry=backend.read_registry(experiment)
        if missed and persisted_registry:
            source={'investable_universe':persisted_registry['comparison_policy']['investable_universe'],'observations':[],'execution_observations':[],
                    'historical_slot_missing':True,'registry_approaches':persisted_registry['approaches']}
        else:
            try:source,policies=load_current_inputs(cohort=cohort)
            except Exception as exc:
                if not persisted_registry:raise
                source={'investable_universe':persisted_registry['comparison_policy']['investable_universe'],'observations':[],'execution_observations':[],
                        'acquisition_failure':type(exc).__name__+': '+str(exc),'registry_approaches':persisted_registry['approaches']}
        if now is None:at=stamp();deadline=datetime.fromisoformat(at.replace('Z','+00:00'))
    if not prior_shared and not missed:
        quote_id=identifier('mi',experiment,'__valuation_quotes__',slot,'valuation_quotes')
        stored_quotes=backend.repo.get('research_input',quote_id)
        if stored_quotes:
            source['current_quotes']=backend.read_input(quote_id)['payload']
            source['valuation_quote_evidence']={'input_id':quote_id,'checksum':stored_quotes.doc['checksum'],'usage':'valuation_only; never model input or fill price'}
        else:
            source['current_quotes']={'quotes':[],'status':'unavailable_platform_quote_collection','source_artifacts':[]}
    policies=policies or {}
    extras=[]
    for name,policy in policies.items():
        if name in {a.approach_id for a in APPROACHES}:continue
        features=policy.get('environment',{}).get('features',[])
        layer=next((f for f in features if 'news' in f or 'gdelt' in f),'not_consumed')
        extras.append(Approach(name,name.replace('_',' '),'reinforcement_learning','actor',news_layer=layer,requires='verified_frozen_policy_bundle',version=policy['configuration_id']).to_dict())
    approaches=source.get('registry_approaches') or registry(extras)
    cohort_id='cohort_'+hashlib.sha256(canonical_json_bytes(sorted(source['investable_universe']))).hexdigest()[:16]
    comparison_policy_id=sha256_checksum(canonical_json_bytes({**POLICY,'investable_universe':source['investable_universe']}))
    for approach in approaches:
        book=backend.ensure_portfolio(experiment,approach['approach_id'],POLICY['initial_capital'],source['investable_universe'],at)
        approach.update(trigger=os.environ.get('FINPLAN_EXPERIMENT_TRIGGER','manual_validation'),code_versions=source.get('code_versions',{}),portfolio_id=book['portfolio_id'],cohort_id=cohort_id,comparison_policy_id=comparison_policy_id,
                        initial_capital=POLICY['initial_capital'],universe=source['investable_universe'])
    _register(backend,experiment,approaches,source,at,slot)
    # Recover original shared input after crash/retry; never replace it with newly fetched data.
    shared_id=identifier('mi',experiment,'__shared__',slot,'shared_source')
    existing=backend.repo.get('research_input',shared_id)
    if existing:
        shared=existing.doc;source=backend.read_input(shared_id)['payload'];policies=source.get('frozen_policies',policies);at=source['availability_cutoff']
    else:
        if not missed:
            research=research or _current_news(backend,experiment.removesuffix('-cohort8'),at)
        source={**copy.deepcopy(source),'news':copy.deepcopy([n for n in (research or {}).get('news',[]) if n.get('quality')!='quarantined_missing_publication_time' and n.get('published_at') and n['published_at']<=at and n.get('available_at','9999')<=at]),'raw_news':copy.deepcopy((research or {}).get('news',[])),
                'research_universe':copy.deepcopy((research or {}).get('universe')),'research_coverage':copy.deepcopy((research or {}).get('coverage',[])),
                'research_workers':copy.deepcopy((research or {}).get('workers',[])),
                'research_available_at':(research or {}).get('available_at'),'availability_cutoff':at,'decision_time':at,
                'scheduled_slot':slot,'historical_slot_missing':missed}
        if missed:
            source={'investable_universe':source['investable_universe'],'observations':[],'execution_observations':[],
                    'news':[],'availability_cutoff':at,'decision_time':at,'scheduled_slot':slot,'historical_slot_missing':True,
                    'reason':'skipped_missing_point_in_time; no historical input fabricated'}
        if not missed:
            reconcile_durable(backend,experiment,source,at,slot)
        frozen_holdings={}
        for approach in approaches:
            snapshot_book=backend.get_portfolio(approach['portfolio_id'])
            frozen_holdings[approach['approach_id']]=_portfolio_holdings(snapshot_book,source)
        if now is None:at=stamp()
        source.update(availability_cutoff=at,decision_time=at,comparison_holdings=frozen_holdings)
        shared=backend.capture_shared_source(experiment,slot,source,at)
    owner=socket.gethostname()+':'+str(os.getpid());summary=[]
    adapters={'codex':_codex_adapter(output),'langgraph':_graph_adapter,**(external_adapters or {})}
    for approach in approaches:
        aid=approach['approach_id'];rid=identifier('mr',experiment,aid,slot);iid=identifier('mi',experiment,aid,slot,'model_input')
        book=backend.ensure_portfolio(experiment,aid,POLICY['initial_capital'],source['investable_universe'],at)
        old=backend.repo.get('research_run',rid)
        if old and old.doc['status']!='running':
            original=backend.read_input(old.doc['input_id'])['payload'];completed=backend.read_run(rid)
            original_holdings=original['holdings']
            record_slot_observation(backend,experiment,aid,slot,rid,iid,book,original_holdings,source,at,completed['outcome'])
            summary.append({'approach_id':aid,'run_id':rid,'status':old.doc['status'],'retry':'existing_terminal'});continue
        holdings=copy.deepcopy(source.get('comparison_holdings',{}).get(aid))
        if holdings is None:
            holdings=_portfolio_holdings(book,{'investable_universe':source['investable_universe'],'observations':[],'execution_observations':[]})
            holdings['capture_blocker']='Original scheduled holdings were not captured; current head cannot be backdated'
        if holdings.get('revision')!=book['revision']:
            holdings['capture_blocker']='Isolated holdings revision changed after shared cutoff; no inference or fill authorized for stale context'
        existing_input=backend.repo.get('research_input',iid)
        if existing_input:
            if not old:
                backend.start_run(experiment,aid,slot,iid,book['portfolio_id'],existing_input.doc['captured_at'],metadata=approach)
            # A crashed invocation may have executed after capture. Do not repeat it with changed
            # inputs or risk duplicate vendor costs; mark explicit recovered failure for review.
            if not backend.claim_run(rid,owner,stamp(),lease_seconds=600):continue
            outcome={'status':'failed','reason':'recovered_interrupted_invocation_original_input_preserved','forecast':{'available':False}}
            backend.finish_run(rid,outcome,stamp());original=backend.read_input(iid)['payload'];record_slot_observation(backend,experiment,aid,slot,rid,iid,book,original['holdings'],source,at,outcome);summary.append({'approach_id':aid,'run_id':rid,**outcome});continue
        def capture(payload):
            payload={**payload,'slot':slot,'approach_id':aid,'run_id':rid,'input_id':iid}
            boundary_at=now or stamp();payload['invocation_at']=boundary_at
            item=backend.capture_input(experiment,aid,slot,payload,shared,boundary_at)
            backend.start_run(experiment,aid,slot,item['input_id'],book['portfolio_id'],boundary_at,metadata=approach)
            if not backend.claim_run(rid,owner,stamp(),lease_seconds=600):raise NotClaimed('another owner claimed permanent run')
        if missed or holdings.get('capture_blocker') or source.get('acquisition_failure') and aid!='cash':
            capture({'schema_version':'finplan-model-input/1','approach_id':aid,'run_id':rid,'input_id':iid,'decision_time':at,'availability_cutoff':at,
                     'invocation_state':'not_invoked','news':{'consumed':False,'reason':'historical input unavailable'},'numeric':{},'holdings':holdings})
            outcome={'status':'skipped' if missed else 'unavailable','reason':'skipped_missing_point_in_time' if missed else holdings.get('capture_blocker') or source['acquisition_failure'],'forecast':{'available':False}}
        else:
            try:
                outcome=evaluate_approach(aid,source,holdings,cutoff=at,run_id=rid,input_id=iid,capture=capture,config=evaluator_config(),
                    policy=policies.get(aid),news_dataset=source.get('native_policy_inputs',{}).get(aid,{}).get('dataset'),external_adapters=adapters,approach_metadata=approach,native_decision_time=source.get('native_policy_inputs',{}).get(aid,{}).get('decision_time'))
            except NotClaimed:
                summary.append({'approach_id':aid,'run_id':rid,'status':'running_other_owner'});continue
            if outcome['status']=='recommended':outcome['status']='succeeded'
            if outcome.get('forecast'):
                issued_at=now or stamp();outcome['recommendation_issued_at']=issued_at
                outcome['forecast']=freeze(outcome['forecast'],decision_at=issued_at,run_id=rid,portfolio_id=book['portfolio_id'])
            outcome['execution_status']='pending_next_hourly_regular_session_quote' if outcome['status'] in ('succeeded','hold') else 'not_applicable'
            outcome['market_input_status']='completed_daily_history_reused_until_next_valid_snapshot'
            outcome['price_as_of']=max((r['session_date'] for r in source.get('observations',[])),default=None)
            outcome['news_research_status']='available' if source.get('news') else 'missing_coverage'
            if aid in source.get('native_policy_inputs',{}):outcome['native_input_status']=source['native_policy_inputs'][aid]['status'];outcome['native_input_detail']={k:v for k,v in source['native_policy_inputs'][aid].items() if k!='dataset'}
        if outcome.get('captured_inputs'):
            extra_payload={'schema_version':'graph-model-sub-boundaries/1','run_id':rid,'approach_id':aid,'input_id':iid,
                           'availability_cutoff':at,'captured_inputs':outcome['captured_inputs'],
                           'activity_receipts':outcome.get('capture_receipts',[])}
            outcome['graph_sub_boundary_ref']=backend.capture_input(experiment,aid,slot,extra_payload,shared,at,kind='graph_sub_boundary_inputs')
        backend.finish_run(rid,outcome,stamp())
        state=backend.get_portfolio(book['portfolio_id'])
        current=_portfolio_holdings(state,source)
        record_slot_observation(backend,experiment,aid,slot,rid,iid,state,current,source,at,outcome)
        summary.append({'approach_id':aid,'run_id':rid,'input_id':iid,'portfolio_id':book['portfolio_id'],'status':outcome['status']})
    result={'slot':slot,'experiment_id':experiment,'generated_at':stamp(),'approaches':summary,'shared_input_id':shared['input_id']}
    (output/'hourly-result.json').write_bytes(canonical_json_bytes(result));return result


def run_interim(slot,experiment,output,*,backend=None):
    """Thirty-minute collection + actual Codex review, persisted separately, never a primary fill."""
    backend=backend or evidence_backend();interim=experiment+'-interim';output=Path(output);output.mkdir(parents=True,exist_ok=True)
    aid='direct_codex_news';rid=identifier('mr',interim,aid,slot);iid=identifier('mi',interim,aid,slot,'model_input')
    def receipt():
        reread=backend.read_run(rid)
        if reread['status']=='running':raise NotClaimed('interim invocation still running; no terminal receipt')
        value={'slot':slot,'experiment_id':interim,'run_id':rid,'input_id':iid,'status':reread['outcome']['status'],
               'input_checksum':reread['input_checksum'],'output_checksum':reread['output_ref']['checksum']}
        (output/'cycle-receipt.json').write_bytes(canonical_json_bytes(value));return value
    old=backend.repo.get('research_run',rid)
    existing_input=backend.repo.get('research_input',iid)
    if old or existing_input:
        if not old:
            captured=backend.read_input(iid)
            book=backend.ensure_portfolio(interim,aid,POLICY['initial_capital'],captured['payload']['numeric']['prepared']['actual_source']['investable_universe'],stamp())
            backend.start_run(interim,aid,slot,iid,book['portfolio_id'],stamp(),metadata={'family':'language_model'})
        if old and old.doc['status']!='running':return receipt()
        if not backend.claim_run(rid,socket.gethostname()+':'+str(os.getpid()),stamp(),lease_seconds=600):raise NotClaimed('interim lease active')
        backend.finish_run(rid,{'status':'failed','reason':'recovered_interrupted_interim_original_inputs_preserved','forecast':{'available':False}},stamp())
        return receipt()
    shared_id=identifier('mi',interim,'__shared__',slot,'shared_source')
    prior_shared=backend.repo.get('research_input',shared_id)
    if prior_shared:
        shared=prior_shared.doc;source=backend.read_input(shared_id)['payload'];data=source['research_source'];at=stamp()
    else:
        data=None;failure=None
        try:
            research_source_id=identifier('mi',interim,'__research_source__',slot,'research_source')
            data=backend.read_input(research_source_id)['payload']
            if data.get('status')=='failed' or data.get('acquisition_failure'):raise ValueError('Platform research acquisition failed: '+str(data.get('acquisition_failure',data.get('reason'))))
            source,policies=load_current_inputs()
        except Exception as exc:
            failure=type(exc).__name__+': '+str(exc)
            registered=backend.read_registry(experiment)
            if not registered:raise
            source={'investable_universe':registered['comparison_policy']['investable_universe'],'observations':[],'execution_observations':[],
                    'acquisition_failure':failure,'code_versions':{'financemodel':os.environ.get('FINPLAN_MODEL_SOURCE_COMMIT','not_supplied')},
                    'trigger':os.environ.get('FINPLAN_EXPERIMENT_TRIGGER','manual_validation')}
            if data is None:data={'news':[],'universe':None,'coverage':[],'workers':[{'worker_id':'source_acquisition','status':'failed','error':failure}],
                                  'available_at':stamp(),'acquisition_failure':failure}
        at=stamp()
        source={**source,'news':[n for n in data['news'] if n.get('published_at') and n['published_at']<=at and n.get('quality')!='quarantined_missing_publication_time'],
                'raw_news':data['news'],'research_universe':data['universe'],'research_coverage':data['coverage'],'research_workers':data['workers'],
                'research_source':data,'availability_cutoff':at,'decision_time':at,'scheduled_slot':slot}
        shared=backend.capture_shared_source(interim,slot,source,at)
    # Review against the actual current primary book, while the interim record belongs to its
    # own nontrading evidence book. Neither recommendation automatically fills at half hours.
    book=backend.ensure_portfolio(interim,aid,POLICY['initial_capital'],source['investable_universe'],at)
    primary=backend.ensure_portfolio(experiment,aid,POLICY['initial_capital'],source['investable_universe'],at)
    holdings=_portfolio_holdings(primary,source)
    def capture(prepared):
        payload={'schema_version':'finplan-model-input/1','approach_id':aid,'run_id':rid,'input_id':iid,'slot':slot,
                 'decision_time':at,'availability_cutoff':source['availability_cutoff'],'model':{'approach_id':aid,'version':'installed_headless_codex'},
                 'holdings':holdings,'research_source':data,'news':{'consumed':not bool(source.get('acquisition_failure')),'layer':'selected_headlines' if not source.get('acquisition_failure') else 'not_consumed',
                 'items':prepared['actual_source']['news'],'prepared_context':prepared['messages']},
                 'numeric':{'prepared':prepared},'invocation_state':'not_invoked' if source.get('acquisition_failure') else 'invoking','purpose':'interim_research_no_primary_execution','trigger':os.environ.get('FINPLAN_EXPERIMENT_TRIGGER','manual_validation'),'code_versions':source.get('code_versions',{})}
        backend.capture_input(interim,aid,slot,payload,shared,at)
        backend.start_run(interim,aid,slot,iid,book['portfolio_id'],at,metadata={'family':'language_model','native_cadence':'thirty_minute_interim','trigger':os.environ.get('FINPLAN_EXPERIMENT_TRIGGER','manual_validation'),'code_versions':source.get('code_versions',{})})
        if not backend.claim_run(rid,socket.gethostname()+':'+str(os.getpid()),stamp(),lease_seconds=600):raise NotClaimed('interim already claimed')
    try:
        if source.get('acquisition_failure'):
            capture({'boundary':'not_invoked_source_acquisition_failed','actual_source':{'news':[]},'messages':[],
                     'error':source['acquisition_failure']})
            raise RuntimeError(source['acquisition_failure'])
        outcome=review(source,holdings,output/'recommendation.json',capture=capture)
        if outcome['status']=='recommended':outcome['status']='succeeded'
        issued_at=stamp();outcome['recommendation_issued_at']=issued_at
        outcome['forecast']=freeze(outcome.get('forecast'),decision_at=issued_at,run_id=rid,portfolio_id=primary['portfolio_id'])
        outcome['purpose']='interim_research_no_primary_execution';outcome['execution_status']='not_applied_interim'
    except Exception as exc:
        if not backend.repo.get('research_run',rid):raise
        outcome={'status':'failed','reason':type(exc).__name__+': '+str(exc),'public_final_response':getattr(exc,'public_final_response',None),'reviewer_invocation':getattr(exc,'invocation',None),'forecast':{'available':False},'purpose':'interim_research_no_primary_execution','trigger':os.environ.get('FINPLAN_EXPERIMENT_TRIGGER','manual_validation'),'code_versions':source.get('code_versions',{})}
    backend.finish_run(rid,outcome,stamp())
    receipt()
    return {'experiment_id':interim,'slot':slot,'run_id':rid,'input_id':iid,'status':outcome['status'],
            'workers':len(data['workers']),'universe_count':len((data['universe'] or {}).get('members',[])),'news_count':len(data['news'])}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--slot',default=None);parser.add_argument('--experiment',default=DEFAULT_EXPERIMENT)
    parser.add_argument('--output',required=True);parser.add_argument('--research');parser.add_argument('--recover-before');parser.add_argument('--account-missed',action='store_true');parser.add_argument('--cohort',choices=['eight_assets'])
    args=parser.parse_args();slot=args.slot or slot_now()
    research=json.loads(Path(args.research).read_text()) if args.research else None
    print(json.dumps(run_hourly(slot,args.experiment,args.output,research=research,account_missed=args.account_missed,cohort=args.cohort)))

if __name__=='__main__':main()
