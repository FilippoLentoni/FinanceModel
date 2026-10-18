"""Exact native preparation, with no paid vendor transport or GPU execution.

These are approval-gated preflights, NOT evaluated Jev/Qwen recommendations.
Only an independently approved managed worker may perform the missing invocation.
"""
from __future__ import annotations
import copy,json,math
from dataclasses import asdict
import numpy as np
from datetime import UTC
import boto3
from botocore.config import Config
from datetime import datetime
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.core.config import load_config
from finplan_model.core.context import RunContext
from finplan_model.core.errors import FinplanError,contract_version
from finplan_model.datasets.outbound import check_outbound_payload
from finplan_model.sim.market import HoldingsView,PointInTimeView
from finplan_model.benchmarks.jev import JevStrategy
from finplan_model.benchmarks.qwen import QwenSwarm,MODEL_ID,REVISION,ROLES
from finplan_model.benchmarks.protocol import pilot_window
from finplan_model.benchmarks.weights import PREFIX
from .runtime import input_market


class PreparedOnly(RuntimeError):pass


def _view(source,holdings,cutoff):
    universe=list(source['investable_universe']);market=input_market(source,cutoff)
    view=PointInTimeView(market,max(market.sessions),datetime.fromisoformat(cutoff.replace('Z','+00:00')),tuple(universe))
    if len(set(universe))!=len(universe) or not universe:raise ValueError('distinct supported universe required')
    positions=holdings.get('positions',[]);ids=[p['instrument_id'] for p in positions]
    if len(ids)!=len(set(ids)) or set(ids)-set(universe):raise ValueError('holdings have unsupported/duplicate instrument')
    shares={p['instrument_id']:p['quantity'] for p in positions};cash=holdings.get('cash',holdings.get('cash_balance',0.))
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in [cash,*shares.values()]):raise ValueError('holdings require finite nonnegative cash/quantities')
    prices={i:view.latest(i) for i in universe}
    if any(v is None for v in prices.values()):raise ValueError('eligible native prices missing')
    value=cash+sum(shares.get(i,0.)*prices[i] for i in universe)
    if value<=0:raise ValueError('positive native holdings value required')
    hv=HoldingsView(shares,float(cash),prices,value)
    dates,returns=view.returns(60)
    if len(returns)<20:raise FinplanError.precondition('native preparation requires20 completed returns',reason='insufficient_history')
    private={'layer':'local_preparation_only; NOT_CONSUMED_BY_MODEL','supplied_to_model':False,
        'transformation_version':'benchmarks.jev.encode_state/1','definitions':{
        'recent_trend':'product(1+last20simple_returns)-1; edges[-.05,-.01,.01,.05]',
        'volatility':'population_std(last20simple_returns)*sqrt252; edges[.15,.30,.60]',
        'current_exposure':'native marked position value/equity; edges[.01,.10,.30,.60]'},
        'derived_statistics':{i:{'momentum20':float(np.prod(1.+returns[-20:,j])-1.),'annualized_population_volatility20':float(np.std(returns[-20:,j])*np.sqrt(252.)),'exposure':hv.weights.get(i,0.)} for j,i in enumerate(universe)},
        'price_basis':source.get('price_basis','supplied_native_close'),'return_dtype':str(returns.dtype),'return_units':'fractional simple return',
        'return_order':universe,'return_session_dates':[d.isoformat() for d in dates],'simple_return_values':returns.tolist(),
        'native_mark_prices':prices,'native_weights':hv.weights,'native_cash_weight':hv.cash_weight,
        'holdings_revision':holdings.get('revision'),'native_session':view.decision_session.isoformat()}
    raw=[v for i in universe for b in market.bars_of(i) for v in (b.open,b.high,b.low,b.close,b.volume) if v is not None]
    return market,view,hv,private,raw


def _managed_proposal(source,market,kind,run_id):
    if not source.get('input_snapshot_id'):return {'available':False,'reason':'approved input_snapshot_id required; no synthetic reference invented'}
    try:window,scope=pilot_window(market,source['investable_universe'],as_of=max(market.sessions))
    except FinplanError as exc:return {'available':False,'reason':exc.details.get('reason',exc.message)}
    body={'domain':'finance','domain_schema_version':'1.0','job_type':'jev_backtest' if kind=='jev' else 'swarm_mode_a',
        'purpose':'research','dry_run':True,'input_snapshot_id':source['input_snapshot_id'],'contract_version':contract_version(),
        'synthetic':bool(source.get('synthetic',False)),'configuration':{'domain':'finance','domain_schema_version':'1.0',
        'payload':{'strategy':'jev' if kind=='jev' else 'qwen_swarm','objective':'llm_benchmark','universe':source['investable_universe'],
        'lookback_days':60,'rebalance_frequency':'daily','constraints':{'long_only':True,'max_weight':.6},'fees':{'transaction_cost_bps':2}}},
        'evaluation_window':window,'idempotency_key':'forward-preflight-'+run_id}
    from finplan_model.core.outcome import require_valid
    require_valid(body,'tools/submit-experiment-request')
    return {'available':True,'tool':'submit_experiment','arguments':body,'scope':scope,'submitted':False,
        'managed_runtime_bound_seconds':900,'same_current_book_forward_bridge':False,'warning':'Native managed pilot evaluates its own chronological historical book; cannot substitute for this current synthetic book decision.',
        'approval_instructions':['Review supported dry_run cost/remaining budget. No job or paid call is created by this preflight.',
        'Only an authorized operator may submit dry_run=false. Existing control service puts Jev/GPU work in awaiting_approval.',
        'Independent human approver uses scripts/approve_run.py --env beta RUN_ID (or signed POST /v1/jobs/RUN_ID/approve). Never auto-approve.',
        'Exact current-book managed inference capture/result bridge remains required before primary comparison execution.']}


def dependency_metadata(kind):
    """Bounded own-account metadata only. Never read a key or download weights."""
    session=boto3.Session(region_name='us-east-2');config=Config(connect_timeout=5,read_timeout=10,retries={'max_attempts':1})
    checked=datetime.now(UTC).isoformat().replace('+00:00','Z')
    def error(exc):return {'status':'metadata_check_unavailable','error_type':type(exc).__name__,
        'aws_code':getattr(exc,'response',{}).get('Error',{}).get('Code')}
    estimate_metadata={'status':'unavailable','approval_granted':False,'submitted':False}
    try:
        from finplan_model.control.costs import estimate
        from finplan_model.control.settings import SsmSettings
        cfg=load_config('beta');job=cfg.job_types['jev_backtest' if kind=='jev' else 'swarm_mode_a']
        prices=SsmSettings(session.client('ssm',config=config),cfg).instance_prices()
        value=estimate(cfg,job,instance_type=job.default_instance_type,instance_count=1,max_runtime_seconds=job.default_runtime_seconds,prices=prices,now=datetime.now(UTC))
        estimate_metadata={**asdict(value),'status':'native_cost_upper_bound','approval_granted':False,'submitted':False,'budget_remaining_checked':False,'vendor_cost_cap_usd':cfg.raw['llm_benchmarks'][kind].get('external_vendor_cost_cap_usd'),'scope':'conditional managed pilot estimate, not an hourly current-book invocation or approval'}
    except Exception as exc:
        estimate_metadata.update(error(exc));estimate_metadata['reason']=getattr(exc,'details',{}).get('reason')
    if kind=='jev':
        try:
            session.client('secretsmanager',config=config).describe_secret(SecretId=load_config('beta').raw['llm_benchmarks']['jev']['secret_name'])
            return {'managed_cost_estimate':estimate_metadata,'checked_at':checked,'credential_metadata_exists':True,'secret_value_loaded':False,'vendor_authentication_tested':False}
        except Exception as exc:return {'managed_cost_estimate':estimate_metadata,'checked_at':checked,'secret_value_loaded':False,**error(exc)}
    try:
        bucket=session.client('ssm',config=config).get_parameter(Name='/finplan/beta/financemodel/config/research-storage-ref')['Parameter']['Value']
        client=session.client('s3',config=config);raw={}
        for name in ('STAGED.json','MANIFEST.sha256'):
            body=client.get_object(Bucket=bucket,Key=PREFIX+name)['Body'].read(524289)
            if len(body)>524288:raise ValueError('staging metadata exceeds bound')
            raw[name]=body
        staged=json.loads(raw['STAGED.json'])
        matches=staged.get('model_id')==MODEL_ID and staged.get('revision')==REVISION and staged.get('manifest_checksum')==sha256_checksum(raw['MANIFEST.sha256'])
        return {'managed_cost_estimate':estimate_metadata,'checked_at':checked,'status':'staged_metadata_matches' if matches else 'staging_identity_or_manifest_mismatch',
            'model_id':staged.get('model_id'),'revision':staged.get('revision'),'manifest_checksum':staged.get('manifest_checksum'),
            'metadata_identity_verified':matches,'full_weight_bytes_verified':False,'weights_loaded':False}
    except Exception as exc:
        code=getattr(exc,'response',{}).get('Error',{}).get('Code')
        return {'managed_cost_estimate':estimate_metadata,'checked_at':checked,**error(exc),'status':'required_stage_metadata_missing' if code in ('NoSuchKey','404','NotFound') else 'metadata_check_unavailable',
            'weights_loaded':False,'full_weight_bytes_verified':False}


def preflight(kind,base,source,holdings,before,*,dependency_reader=None):
    config=copy.deepcopy(load_config('beta').raw['llm_benchmarks'][kind]);captured=False
    controls={'manual_approved_managed_worker_required':True,'managed_worker_started':False,'approval_not_bypassed':True,
        'credential_status':'not_read_or_tested_by_preflight; not a missing-credential claim','configured_limits':config,
        'dependency_metadata':(dependency_reader or dependency_metadata)(kind)}
    def capture(prepared):
        nonlocal captured
        captured=True
        before({'boundary':'not_invoked_'+kind+'_approved_managed_worker_required','invocation_state':'not_invoked',
                'model_consumed_inputs':False,'news_consumed':False,'controls':controls,**prepared})
    try:
        market,view,hv,private,raw=_view(source,holdings,base['availability_cutoff'])
        proposal=_managed_proposal(source,market,kind,base['run_id'])
        ctx=RunContext(environment='beta',purpose='research',run_id=base['run_id'],synthetic=bool(source.get('synthetic',False)))
        if kind=='jev':
            class CaptureClient:
                model=config['model']
                def ask(self,payload,*,ctx,raw_values=()):
                    guard={'status':'passed'}
                    try:check_outbound_payload(payload,ctx=ctx,raw_values=raw_values,destination='typesafe_systemone')
                    except FinplanError as exc:guard={'status':'rejected','reason':exc.details.get('reason',exc.code)}
                    capture({'public_request':{'payload':copy.deepcopy(payload),'checksum':sha256_checksum(canonical_json_bytes(payload)),
                            'supplied_to_model':False,'status':'prepared_not_supplied'},'outbound_guard':guard,'local_transformation':private,
                            'native_sizing':{'threshold':.20,'max_delta':.10,'confidence_minimum':.5,'sales_first':True,'purchases_cash_scaled':True},
                            'managed_pilot_proposal':proposal})
                    raise PreparedOnly()
            JevStrategy(CaptureClient(),ctx=ctx,raw_values=raw).decide(view,hv)
        else:
            def capture_backend(messages):
                capture({'prepared_role_messages':[{'role':'market_analyst','messages':copy.deepcopy(messages),'supplied_to_model':False}],
                    'declared_roles':list(ROLES),'dependent_role_inputs':'NOT_PREPARED: require actual preceding model outputs; no summaries/proposals fabricated',
                    'model_id':MODEL_ID,'revision':REVISION,'local_transformation':private,'managed_pilot_proposal':proposal})
                raise PreparedOnly()
            QwenSwarm(capture_backend,run_id=base['run_id']).decide(view,hv)
    except PreparedOnly:pass
    except Exception as exc:
        if captured:raise
        capture({'preparation_error':{'type':type(exc).__name__,'reason':getattr(exc,'details',{}).get('reason',str(exc))},'public_request':None})
    if not captured:capture({'preparation_error':{'reason':'insufficient_native_history; no model request formed'},'public_request':None})
    return {'status':'unavailable','reason':'approved_managed_'+kind+'_worker_required',
        'execution_status':'not_applicable','invocation_state':'not_invoked','required_coverage_blocker':True,
        'remaining_execution_work':'Approved managed inference must consume this exact current-book request and capture actual response; historical pilot output is not a compatible substitute.',
        'usage':{'calls':0,'model_input_tokens':0,'model_output_tokens':0,'vendor_cost_usd':0.,'cost_basis':'no vendor/model invocation'},
        'forecast':{'available':False,'reason':'no model invocation or forecast'}}


def jev_adapter(base,source,holdings,before):return preflight('jev',base,source,holdings,before)
def qwen_adapter(base,source,holdings,before):return preflight('qwen',base,source,holdings,before)
