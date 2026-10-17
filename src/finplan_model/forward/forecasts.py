"""Freeze forecasts and match first eligible observations without rewriting predictions."""
from __future__ import annotations
import copy
import math
from datetime import datetime
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum


def dt(value):return datetime.fromisoformat(value.replace('Z','+00:00'))


def freeze(forecast,*,decision_at,run_id,portfolio_id):
    if not forecast or forecast.get('status')=='not_available' or forecast.get('available') is False:
        return {'status':'not_available','available':False,'run_id':run_id,'portfolio_id':portfolio_id}
    result=copy.deepcopy(forecast)
    target=result.get('target_at',result.get('target_timestamp'))
    expected=result.get('expected_value',result.get('value'))
    baseline=result.get('baseline_value')
    if not target or dt(target)<=dt(decision_at):raise ValueError('forecast target must follow actual decision time')
    if result.get('units')!='USD':raise ValueError('forecast units must be USD')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in (expected,baseline)):raise ValueError('forecast values must be finite nonnegative USD')
    result.update(run_id=run_id,portfolio_id=portfolio_id,decision_at=decision_at,target_at=target,expected_value=expected,
                  status='pending',available=True,alignment_rule='first_valid_observation_at_or_after_target',next_eligible_observation_at=target)
    result['forecast_checksum']=sha256_checksum(canonical_json_bytes(result))
    return result


def reconcile(frozen,observations,*,previous_match=None):
    """Returns a separate immutable match. Later quotes cannot replace an existing match."""
    if previous_match is not None:
        if previous_match['forecast_checksum']!=frozen['forecast_checksum']:raise ValueError('forecast checksum mismatch')
        return copy.deepcopy(previous_match)
    if frozen.get('status')!='pending':return None
    eligible=[o for o in observations if o.get('portfolio_id')==frozen['portfolio_id'] and
              o.get('observation_kind',o.get('kind'))=='valid' and o.get('quote_status')=='fresh' and
              dt(o['observed_at'])>=dt(frozen['target_at'])]
    if not eligible:return None
    chosen=min(eligible,key=lambda o:(dt(o['observed_at']),o['observation_id']))
    value=chosen['value'];flows=chosen.get('net_external_cash_flows_since_decision',0)
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (value,flows)):raise ValueError('invalid realized value or cash flow')
    aligned=value-flows
    error=aligned-frozen['expected_value']
    result={'forecast_checksum':frozen['forecast_checksum'],'run_id':frozen['run_id'],'portfolio_id':frozen['portfolio_id'],
            'observation_id':chosen['observation_id'],'observed_at':chosen['observed_at'],'quote_provenance':chosen.get('quote_provenance'),
            'realized_value':value,'net_external_cash_flows':flows,'flow_adjusted_realized_value':aligned,
            'signed_error':error,'absolute_error':abs(error),'squared_error':error*error,'units':'USD','status':'realized'}
    result['match_checksum']=sha256_checksum(canonical_json_bytes(result))
    return result


def errors(matches):
    if not matches:return {'matched_count':0,'mae':None,'rmse':None,'calibration':'insufficient_distributions_and_observations'}
    return {'matched_count':len(matches),'mae':sum(m['absolute_error'] for m in matches)/len(matches),
            'rmse':math.sqrt(sum(m['squared_error'] for m in matches)/len(matches)),
            'calibration':'not_computed_without_forecast_distributions'}
