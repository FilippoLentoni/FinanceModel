"""Ancillary cost outbox import. Model runs/artifacts remain authoritative; no ledger copy."""
from __future__ import annotations
import base64
import json
import os
from datetime import UTC,datetime
from zoneinfo import ZoneInfo
from finplan_model.benchmarks.forward_jev import checksum
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum


def terminal_cost_event(run):
    descriptor=run.get('forward_jev') or run['forward_qwen'];result=run.get('result') or {}
    section=(result.get('payload') or {}).get('forward_jev') or (result.get('payload') or {}).get('forward_qwen') or {}
    usage=section.get('usage')
    not_started=not run.get('job_started') and int(run.get('attempt',0))==0
    known_no_vendor=bool(run.get('forward_qwen')) or not_started or isinstance(usage,dict) and usage.get('transport_calls',usage.get('calls'))==0
    occurred=run.get('job_finished_at') or result.get('completed_at') or run['completed_at']
    refs=[descriptor['binding_ref'],*([descriptor['weights_proof_ref']] if descriptor.get('weights_proof_ref') else []),*result.get('artifacts',[])]
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
        if event.get('protocol')!='finplan-managed-model-cost/1' or event.get('experiment_id')!=experiment or event.get('event_id')!='managed:'+event.get('managed_run_id','') or event.get('approach_id') not in ('jev_systemone','qwen_swarm_mode_a'):
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
        from .managed_inputs import captured_event
        input_kind='managed_inference_evidence:'+event['managed_run_id']
        input_id='mo_'+sha256_checksum(canonical_json_bytes([experiment,event['approach_id'],slot,input_kind])).split(':')[1]
        old_inputs=backend.repo.get('portfolio_observation',input_id)
        if old_inputs:
            input_receipt=old_inputs.doc
            input_bytes=backend.read_artifact_bytes(input_receipt)
            input_payload=json.loads(input_bytes)
            if input_receipt.get('experiment_id')!=experiment or input_receipt.get('approach_id')!=event['approach_id'] or input_receipt.get('portfolio_id')!=event['portfolio_id'] or input_receipt.get('kind')!=input_kind or input_receipt.get('slot')!=slot or input_receipt.get('checksum')!=sha256_checksum(input_bytes) or input_payload.get('protocol')!='finplan-managed-inference-evidence/1' or input_payload.get('cost_event_checksum')!=checksum(event) or input_payload.get('managed_run_id')!=event['managed_run_id']:
                raise ValueError('existing managed original input receipt conflicts')
        else:
            input_payload=captured_event(event,reader)
            input_receipt=backend.record_observation(experiment,event['approach_id'],slot,event['portfolio_id'],input_payload,at,kind=input_kind)
            if input_receipt['checksum']!=checksum(input_payload) or backend.read_artifact_bytes(input_receipt)!=canonical_json_bytes(input_payload):
                raise ValueError('managed original input receipt readback mismatch')
        input_ref={'observation_id':input_receipt['observation_id'],'checksum':input_receipt['checksum']}
        day=datetime.fromisoformat(slot.replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date().isoformat()
        directory=refresh_dir or os.environ.get('FINPLAN_REPORT_REFRESH_DIR')
        if not directory:raise ValueError('durable FINPLAN_REPORT_REFRESH_DIR required before managed cost acknowledgement')
        from .report_refresh import enqueue
        enqueue(directory,{'protocol':'finplan-report-refresh/1','experiment_id':experiment,
            'reporting_day':day,'event_id':event['event_id'],'observation_id':ref['observation_id'],'checksum':ref['checksum']})
        enqueue(directory,{'protocol':'finplan-report-refresh/1','experiment_id':experiment,
            'reporting_day':day,'event_id':input_payload['event_id'],'observation_id':input_ref['observation_id'],'checksum':input_ref['checksum']})
        # A genuine worker policy may be issued before NY midnight and become
        # terminal later. Link the original two receipts into its issue-day query;
        # do not copy bytes, retime costs or create another trading opportunity.
        outputs=[]
        for artifact in input_payload['artifacts']:
            original=json.loads(base64.b64decode(artifact['raw_base64']))
            if original.get('kind')=='forward_result':outputs.append(original)
        if len(outputs)>1:raise ValueError('managed worker has ambiguous policy issue receipts')
        if outputs:
            issued=outputs[0]['issued_at']
            issued_instant=datetime.fromisoformat(issued.replace('Z','+00:00'))
            if issued_instant.tzinfo is None:raise ValueError('managed policy issue time lacks timezone')
            if issued_instant>datetime.fromisoformat(slot.replace('Z','+00:00')):raise ValueError('managed terminal cost precedes actual policy issue time')
            issue_day=issued_instant.astimezone(ZoneInfo('America/New_York')).date().isoformat()
            if issue_day!=day:
                issue_slot=issued_instant.astimezone(UTC).isoformat().replace('+00:00','Z')
                issue_kind='managed_inference_issue_link:'+event['managed_run_id']
                issue_payload={'protocol':'finplan-managed-inference-issue-link/1',
                    'event_id':'managed_inputs_issue:'+event['managed_run_id'],
                    'managed_run_id':event['managed_run_id'],'issued_at':issued,
                    'terminal_occurred_at':event['occurred_at'],'experiment_id':experiment,
                    'approach_id':event['approach_id'],'portfolio_id':event['portfolio_id'],
                    'inference_receipt':input_ref,'cost_receipt':ref}
                issue_id='mo_'+sha256_checksum(canonical_json_bytes([experiment,event['approach_id'],issue_slot,issue_kind])).split(':')[1]
                old_issue=backend.repo.get('portfolio_observation',issue_id)
                if old_issue:
                    issue_receipt=old_issue.doc
                    if any(issue_receipt.get(key)!=value for key,value in {'experiment_id':experiment,'approach_id':event['approach_id'],'portfolio_id':event['portfolio_id'],'kind':issue_kind,'slot':issue_slot}.items()) or backend.read_artifact_bytes(issue_receipt)!=canonical_json_bytes(issue_payload):
                        raise ValueError('existing managed policy issue link conflicts')
                else:
                    issue_receipt=backend.record_observation(experiment,event['approach_id'],issue_slot,event['portfolio_id'],issue_payload,at,kind=issue_kind)
                if issue_receipt['checksum']!=checksum(issue_payload) or backend.read_artifact_bytes(issue_receipt)!=canonical_json_bytes(issue_payload):raise ValueError('managed policy issue link readback mismatch')
                enqueue(directory,{'protocol':'finplan-report-refresh/1','experiment_id':experiment,
                    'reporting_day':issue_day,'event_id':issue_payload['event_id'],
                    'observation_id':issue_receipt['observation_id'],'checksum':issue_receipt['checksum']})
                days.append(issue_day)
        reader.runs.ack_forward_cost_event(experiment,event['event_id'],ref,at)
        imported.append({'event_id':event['event_id'],'observation_ref':ref,'inference_evidence_ref':input_ref})
        day=datetime.fromisoformat(slot.replace('Z','+00:00')).astimezone(ZoneInfo('America/New_York')).date().isoformat()
        days.append(day)
    return {'status':'pending_more' if token else 'complete','imported':imported,
        'affected_reporting_days':sorted(set(days)),'bounded_limit':max_events,'incomplete':bool(token)}
