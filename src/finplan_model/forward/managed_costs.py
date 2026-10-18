"""Ancillary cost outbox import. Model runs/artifacts remain authoritative; no ledger copy."""
from __future__ import annotations
import json
import os
from datetime import UTC,datetime
from zoneinfo import ZoneInfo
from finplan_model.benchmarks.forward_jev import checksum
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum


def terminal_cost_event(run):
    descriptor=run['forward_jev'];result=run.get('result') or {}
    section=(result.get('payload') or {}).get('forward_jev') or {}
    usage=section.get('usage')
    not_started=not run.get('job_started') and int(run.get('attempt',0))==0
    known_no_vendor=not_started or isinstance(usage,dict) and usage.get('transport_calls',usage.get('calls'))==0
    occurred=run.get('job_finished_at') or result.get('completed_at') or run['completed_at']
    refs=[descriptor['binding_ref'],*result.get('artifacts',[])]
    return {'protocol':'finplan-managed-model-cost/1','event_id':'managed:'+run['run_id'],
        'experiment_id':descriptor['experiment_id'],'approach_id':descriptor['approach_id'],
        'portfolio_id':descriptor['portfolio_id'],'managed_run_id':run['run_id'],
        'occurred_at':occurred,'occurred_at_basis':'actual_managed_job_end' if run.get('job_finished_at') else 'verified_worker_completed_at' if result.get('completed_at') else 'controller_terminal_observed_at',
        'job_state':run['state'],'invocation_state':section.get('invocation_state','not_invoked' if not_started else 'unknown_or_indeterminate'),
        'binding_ref':descriptor['binding_ref'],'binding_checksum':descriptor['binding_checksum'],
        'source_evidence_refs':refs,'compute_observed_usd':0. if not_started else None,
        'compute_estimated_usd':(run.get('cost_estimate') or {}).get('estimated_usd_upper_bound'),
        'compute_cost_basis':'observed no managed compute started; unspent native upper-bound estimate separate' if not_started else 'billing unavailable; native preflight upper-bound estimate separate, never elapsed-time observed billing',
        'vendor_observed_usd':0. if known_no_vendor else None,
        'vendor_estimated_usd':usage.get('external_vendor_estimated_usd') if isinstance(usage,dict) else 0. if known_no_vendor else None,
        'vendor_cost_basis':usage.get('vendor_price_basis') if isinstance(usage,dict) else 'observed no vendor invocation' if known_no_vendor else 'unknown vendor side effect/usage; no zero fabricated',
        'usage':usage,'worker_image_digest':descriptor['worker_image_digest'],
        'approval':run.get('approval'),'worker_source_commit':descriptor.get('worker_source_commit')}


def import_pending(backend,reader,experiment,*,at,max_events=100,refresh_dir=None):
    """At most100 immutable pending events; ack only after original byte readback."""
    rows,token=reader.runs.query_forward_cost_events(experiment,limit=max_events)
    imported=[];days=[]
    for event in rows:
        if event.get('protocol')!='finplan-managed-model-cost/1' or event.get('experiment_id')!=experiment or event.get('event_id')!='managed:'+event.get('managed_run_id','') or event.get('approach_id')!='jev_systemone':
            raise ValueError('managed cost outbox event scope invalid')
        kind='managed_model_cost:'+event['managed_run_id'];slot=datetime.fromisoformat(event['occurred_at'].replace('Z','+00:00')).astimezone(UTC).isoformat().replace('+00:00','Z')
        oid='mo_'+sha256_checksum(canonical_json_bytes([experiment,event['approach_id'],slot,kind])).split(':')[1]
        existing=backend.repo.get('portfolio_observation',oid)
        if existing:
            receipt=existing.doc
            if receipt.get('experiment_id')!=experiment or receipt.get('approach_id')!=event['approach_id'] or receipt.get('portfolio_id')!=event['portfolio_id'] or receipt.get('kind')!=kind or receipt.get('slot')!=slot or receipt.get('checksum')!=checksum(event) or backend.read_artifact_bytes(receipt)!=canonical_json_bytes(event):
                raise ValueError('existing permanent managed cost receipt conflicts')
        else:
            receipt=backend.record_observation(experiment,event['approach_id'],slot,event['portfolio_id'],event,at,kind=kind)
            if receipt['checksum']!=checksum(event) or backend.read_artifact_bytes(receipt)!=canonical_json_bytes(event):
                raise ValueError('managed cost receipt readback mismatch')
        ref={'observation_id':receipt['observation_id'],'checksum':receipt['checksum']}
        day=datetime.fromisoformat(slot.replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date().isoformat()
        directory=refresh_dir or os.environ.get('FINPLAN_REPORT_REFRESH_DIR')
        if not directory:raise ValueError('durable FINPLAN_REPORT_REFRESH_DIR required before managed cost acknowledgement')
        from .report_refresh import enqueue
        enqueue(directory,{'protocol':'finplan-report-refresh/1','experiment_id':experiment,
            'reporting_day':day,'event_id':event['event_id'],'observation_id':ref['observation_id'],'checksum':ref['checksum']})
        reader.runs.ack_forward_cost_event(experiment,event['event_id'],ref,at)
        imported.append({'event_id':event['event_id'],'observation_ref':ref})
        day=datetime.fromisoformat(slot.replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date().isoformat()
        days.append(day)
    return {'status':'pending_more' if token else 'complete','imported':imported,
        'affected_reporting_days':sorted(set(days)),'bounded_limit':max_events,'incomplete':bool(token)}
