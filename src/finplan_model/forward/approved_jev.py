"""Read-only reuse of an approved, completed, exactly bound managed Jev decision.

This adapter never submits jobs, loads credentials/secrets, or calls the vendor.
No source-supplied approval/result object is trusted.
"""
from __future__ import annotations

import copy
from datetime import timedelta
from finplan_model.benchmarks.forward_jev import (PROTOCOL, checksum, context_identity,
    index_id, instant, managed_id, read_named, validate_binding, journal_id)
from finplan_model.core.errors import FinplanError


def current_context(base, source, holdings, prepared, experiment):
    return {'experiment_id': experiment, 'approach_id': 'jev_systemone',
        'source': {k: source[k] for k in ('input_snapshot_id', 'checksum', 'snapshot_created_at', 'investable_universe', 'price_basis')},
        'holdings': copy.deepcopy(holdings), 'constraints': copy.deepcopy(base['constraints']),
        'request_checksum': prepared['public_request']['checksum'],
        'local_transformation': copy.deepcopy(prepared['local_transformation']),
        'native_sizing': copy.deepcopy(prepared['native_sizing'])}


class ApprovedForwardReader:
    def __init__(self, artifacts, runs):
        self.artifacts, self.runs = artifacts, runs

    def lookup(self, context, cutoff):
        index = self.runs.get_forward_context(context_identity(context))
        if index is None:
            return {'status': 'no_exact_context_proposal'}
        if index['context_id'] != context_identity(context):
            raise ValueError('approved forward context index mismatch')
        binding = validate_binding(self.artifacts.get_json(index['binding_ref']))
        if checksum(binding) != index['binding_checksum'] or context_identity(binding) != context_identity(context):
            raise ValueError('approved forward binding/current context mismatch')
        rid = managed_id(binding)
        if rid != index['managed_run_id']:
            raise ValueError('approved forward permanent run identity mismatch')
        run = self.runs.get_run(rid)
        if run is None:
            return {'status': 'proposal_reservation_without_job', 'managed_run_id': rid}
        descriptor = run.get('forward_jev') or {}
        if descriptor.get('binding_ref') != index['binding_ref'] or descriptor.get('binding_checksum') != checksum(binding) or run['input_snapshot_id'] != binding['source']['input_snapshot_id']:
            raise ValueError('managed run binding mismatch')
        if run['state'] in ('succeeded', 'failed', 'cancelled', 'timed_out'):
            platform_available_at = run.get('completed_at')
            if not platform_available_at:
                return {'status': 'managed_terminal_availability_missing', 'managed_run_id': rid}
            if instant(platform_available_at) > instant(cutoff):
                return {'status': 'managed_result_not_yet_platform_available', 'managed_run_id': rid,
                    'platform_available_at': platform_available_at}
        if run['state'] != 'succeeded':
            return {'status': 'managed_' + run['state'], 'managed_run_id': rid, 'approval_required': True}
        approval = run.get('approval') or {}
        if not run.get('approval_required') or not approval.get('approved_by') or approval['approved_by'] == run.get('principal_name') or run.get('purpose') != 'research' or run.get('job_type') != 'jev_backtest':
            raise ValueError('managed forward approval absent or invalid')
        result = run.get('result') or {}
        if result.get('completion_status') != 'succeeded' or result.get('run_id') != rid or result.get('input_snapshot_id') != run['input_snapshot_id'] or result.get('configuration_id') != run['configuration_id']:
            raise ValueError('managed result/run identity mismatch')
        section = (result.get('payload') or {}).get('forward_jev') or {}
        ref = section.get('result_ref')
        if section.get('binding_checksum') != checksum(binding) or ref not in result.get('artifacts', []):
            raise ValueError('managed forward output receipt missing')
        output = self.artifacts.get_json(ref)
        if output.get('protocol') != PROTOCOL or output.get('kind') != 'forward_result' or output.get('binding_checksum') != checksum(binding) or output.get('managed_run_id') != rid or output.get('request_checksum') != binding['request_checksum'] or output.get('ledger_updated') is not False:
            raise ValueError('managed forward output binding invalid')
        if instant(output['issued_at']) > instant(platform_available_at):raise ValueError('managed terminal receipt predates original output')
        if instant(output['issued_at']) > instant(cutoff):
            raise ValueError('managed response unavailable at current decision cutoff')
        if output['response_journal_ref'].get('artifact_id') != journal_id(binding, 'response') or output['invocation_claim_ref'].get('artifact_id') != journal_id(binding, 'claim'):
            raise ValueError('managed output does not point to original permanent invocation journal')
        claim = self.artifacts.get_json(output['invocation_claim_ref'])
        if claim.get('binding_checksum') != checksum(binding) or claim.get('managed_run_id') != rid or claim.get('invocation_input_ref') != output['invocation_input_ref']:
            raise ValueError('permanent invocation claim mismatch')
        input_receipt = self.artifacts.get_json(output['invocation_input_ref'])
        if input_receipt.get('worker_image_digest') != descriptor.get('worker_image_digest') or input_receipt.get('worker_source_commit') != descriptor.get('worker_source_commit'):
            raise ValueError('original managed invocation worker identity differs from approved descriptor')
        response_journal = self.artifacts.get_json(output['response_journal_ref'])
        if input_receipt.get('binding_checksum') != checksum(binding) or input_receipt.get('managed_run_id') != rid or input_receipt.get('request_checksum') != binding['request_checksum'] or input_receipt.get('public_request') != binding['request'] or input_receipt.get('approval') != {'approved_by': approval['approved_by'], 'approved_at': approval['approved_at']} or response_journal.get('managed_run_id') != rid or response_journal.get('binding_checksum') != checksum(binding) or response_journal.get('request_checksum') != binding['request_checksum'] or response_journal.get('invocation_input_ref') != output['invocation_input_ref'] or response_journal.get('invocation_claim_ref') != output['invocation_claim_ref'] or response_journal.get('public_response') != output['public_response']:
            raise ValueError('managed invocation boundary receipt mismatch')
        if not (instant(approval['approved_at']) <= instant(input_receipt['captured_at']) <= instant(claim['claimed_at']) <= instant(response_journal['received_at']) <= instant(output['issued_at'])):
            raise ValueError('managed invocation timestamps invalid')
        signal_formed_at = response_journal['received_at']
        if output.get('signal_formed_at', signal_formed_at) != signal_formed_at:
            raise ValueError('managed original signal formation timestamp differs')
        if instant(cutoff) >= instant(signal_formed_at) + timedelta(days=1):
            return {'status': 'native_daily_signal_expired', 'managed_run_id': rid,
                'signal_formed_at': signal_formed_at, 'platform_available_at': platform_available_at}
        return {'status': 'approved_exact_result', 'managed_run_id': rid, 'binding': binding,
            'binding_ref': index['binding_ref'], 'result_ref': ref, 'output': output,
            'invocation_input': input_receipt, 'response_journal': response_journal,
            'signal_formed_at': signal_formed_at, 'platform_available_at': platform_available_at,
            'approval': copy.deepcopy(approval),
            'managed_cost_event': copy.deepcopy(run['forward_cost_event'])}

    @classmethod
    def from_ec2(cls):
        import boto3
        from finplan_contracts import boundaries
        from finplan_model.core.config import load_config
        from finplan_model.core.aws_clients import s3_client
        from finplan_model.core.artifacts import S3ArtifactStore
        from finplan_model.control.store import DynamoRunStore
        cfg = load_config('beta'); session = boto3.Session(region_name=cfg.region)
        bucket = session.client('ssm').get_parameter(Name=cfg.ssm_name('config', 'research-storage-ref'))['Parameter']['Value']
        return cls(S3ArtifactStore(s3_client(cfg.region, session=session), bucket),
            DynamoRunStore(session.client('dynamodb'), boundaries.resource_name('beta', 'financemodel', 'job-control')))


def adapter(experiment, *, reader_factory=ApprovedForwardReader.from_ec2, dependency_reader=None):
    reader = None
    def invoke(base, source, holdings, before):
        nonlocal reader
        from .public_preflight import preflight
        captured = []
        unavailable = preflight('jev', base, source, holdings, captured.append, dependency_reader=dependency_reader)
        prepared = captured[0]
        try:
            if not prepared.get('public_request') or prepared.get('outbound_guard', {}).get('status') != 'passed':
                lookup = {'status': 'native_preparation_ineligible'}
            else:
                context = current_context(base, source, holdings, prepared, experiment)
                if reader is None:
                    reader = reader_factory()
                lookup = reader.lookup(context, base['availability_cutoff'])
        except Exception as exc:
            lookup = {'status': 'managed_forward_readback_invalid', 'error_type': type(exc).__name__}
        if lookup['status'] != 'approved_exact_result':
            before({**prepared, 'managed_forward_readback': lookup})
            return {**unavailable, 'managed_forward_readback': lookup}
        origin = lookup['binding']; output = lookup['output']
        before({**prepared, 'boundary': 'approved_managed_jev_result_reuse',
            'invocation_state': 'reused_approved_native_daily_signal', 'model_consumed_inputs': True,
            'public_request': {**prepared['public_request'], 'status': 'supplied_to_original_approved_managed_invocation', 'supplied_to_model': True},
            'public_response': copy.deepcopy(output['public_response']),
            'original_invocation_input': copy.deepcopy(lookup['invocation_input']),
            'original_preflight_input_id': origin['origin_input_id'], 'original_preflight_run_id': origin['origin_run_id'],
            'original_worker_run_id': lookup['managed_run_id'], 'original_worker_input_ref': output['invocation_input_ref'],
            'original_model_availability_cutoff': origin['availability_cutoff'],
            'managed_run_id': lookup['managed_run_id'], 'managed_binding_ref': lookup['binding_ref'],
            'managed_result_ref': lookup['result_ref'], 'model_issued_at': output['issued_at'], 'signal_formed_at':lookup['signal_formed_at'], 'platform_available_at':lookup['platform_available_at'],
            'actual_returned_model': output['public_response']['model'], 'approval': lookup['approval'],
            'reuse_policy': 'same exact approved snapshot/public request/private preprocessing/book revision; <=24h native signal age; no new provider call'})
        from finplan_model.benchmarks.forward_jev import prepare_native, _ReplayClient
        from finplan_model.benchmarks.jev import JevStrategy
        from finplan_model.core.context import RunContext
        ctx = RunContext(environment='beta', purpose='research', run_id=base['run_id'], synthetic=bool(source.get('synthetic')))
        view, hv, raw = prepare_native(origin, source, ctx=ctx)
        replay = JevStrategy(_ReplayClient(origin['request']['model'], lookup['response_journal']), ctx=ctx, raw_values=raw).decide(view, hv)
        expected = output['policy']
        if dict(replay.weights) != expected['weights'] or replay.cash != expected['cash'] or replay.solution_status != expected['solution_status']:
            raise ValueError('approved native response/policy sizing mismatch')
        return {**copy.deepcopy(expected), 'status': 'reused', 'managed_run_id': lookup['managed_run_id'],
            'signal_origin_run_id': lookup['managed_run_id'], 'proposal_origin_run_id': origin['origin_run_id'], 'signal_issued_at': output['issued_at'], 'signal_formed_at':lookup['signal_formed_at'], 'platform_available_at':lookup['platform_available_at'],
            'returned_model': output['public_response']['model'], 'model_invocation_this_slot': False,
            'model_cost': 0., 'model_cost_basis': 'observed no new vendor/managed invocation; original managed event counted separately; local EC2 allocation unpriced',
            'managed_cost_event': copy.deepcopy(lookup['managed_cost_event']),
            'usage': {'calls': 0, 'vendor_cost_usd': 0., 'cost_basis': 'approved result reuse; original managed invocation cost retained separately', 'original_usage': output['usage']}}
    return invoke
