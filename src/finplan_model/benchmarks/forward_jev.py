"""One approval-gated Jev decision; immutable inputs/results, never a portfolio ledger.

This internal protocol intentionally does not extend the public submission schema.
Only the trusted backend producer may attach a binding to a managed job.
"""
from __future__ import annotations

import copy
import json
from datetime import UTC, datetime, timedelta

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError

PROTOCOL = 'finplan-jev-forward/1'
OBJECTIVE = 'forward_paper_decision'


def checksum(value):
    return sha256_checksum(canonical_json_bytes(value))


def instant(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('timezone required')
    return parsed.astimezone(UTC)


def context_identity(binding):
    return checksum({key: binding[key] for key in ('source', 'holdings', 'constraints',
        'request_checksum', 'local_transformation', 'native_sizing', 'experiment_id', 'approach_id')})


def index_id(binding):
    return 'forward_jev_context_' + context_identity(binding).split(':')[1][:40]


def journal_id(binding, kind):
    return 'forward_jev_' + kind + '_' + checksum(binding).split(':')[1][:40]


def read_named(store, artifact_id):
    ref = store.resolve(kind='run_artifact', artifact_id=artifact_id)
    if ref.size_bytes is not None and ref.size_bytes > 2**20:
        raise ValueError('forward journal exceeds bound')
    return ref, store.get_json(ref)


def verify_current_book(platform, binding):
    if platform is None:
        raise ValueError('current paper book eligibility cannot be verified')
    expected = binding['holdings']
    metadata = platform.get_portfolio(expected['portfolio_id'])
    settings = metadata.get('settings') or {}
    if metadata.get('synthetic') is not True or settings.get('isolated_experiment') is not True or settings.get('experiment_id') != binding['experiment_id'] or settings.get('approach_id') != binding['approach_id']:
        raise ValueError('managed forward request is not the declared isolated experiment book')
    current = platform.get_portfolio_state(expected['portfolio_id'])
    if current.get('synthetic') is not True or current.get('portfolio_id') != expected['portfolio_id'] or current.get('revision') != expected['revision']:
        raise ValueError('current paper book revision changed')
    state = current['paper_state']
    if state.get('mode') != 'paper' or checksum(state['positions']) != checksum(expected['positions']) or state['cash_balance'] != expected.get('cash_balance', expected.get('cash')):
        raise ValueError('current paper book quantities/cash changed')
    return current


def managed_id(binding):
    # A permanent run head protects paid invocation even after native idempotency TTL.
    value = int(checksum(binding).split(':')[1][:32], 16)
    alphabet = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'
    chars = []
    for _ in range(26):
        chars.append(alphabet[value & 31]); value >>= 5
    return 'run_' + ''.join(reversed(chars))


def make_binding(captured, source, *, experiment_id, captured_checksum):
    """Copy only the exact prepared public request and its private identity/context."""
    prepared = captured['numeric']['prepared']
    public = prepared['public_request']
    if captured['approach_id'] != 'jev_systemone' or public.get('supplied_to_model'):
        raise ValueError('an original Jev not-invoked preparation is required')
    if prepared.get('outbound_guard', {}).get('status') != 'passed':
        raise ValueError('original preparation did not pass outbound policy')
    if checksum(captured) != captured_checksum:
        raise ValueError('original captured bytes checksum mismatch')
    binding = {'protocol': PROTOCOL, 'experiment_id': experiment_id,
        'approach_id': captured['approach_id'], 'origin_run_id': captured['run_id'],
        'origin_input_id': captured['input_id'], 'origin_input_checksum': captured_checksum,
        'scheduled_slot': captured['slot'], 'availability_cutoff': captured['availability_cutoff'],
        'source': {k: source[k] for k in ('input_snapshot_id', 'checksum', 'snapshot_created_at', 'investable_universe', 'price_basis')},
        'shared_source_checksum': captured['shared_source_checksum'],
        'holdings': copy.deepcopy(captured['holdings']), 'constraints': copy.deepcopy(captured['constraints']),
        'request': copy.deepcopy(public['payload']), 'request_checksum': public['checksum'],
        'local_transformation': copy.deepcopy(prepared['local_transformation']),
        'native_sizing': copy.deepcopy(prepared['native_sizing']),
        'code_versions': copy.deepcopy(captured['code_versions'])}
    validate_binding(binding)
    return binding


def validate_binding(binding):
    from finplan_model.control.validation import find_storage_location
    required = {'protocol', 'experiment_id', 'approach_id', 'origin_run_id', 'origin_input_id',
        'origin_input_checksum', 'scheduled_slot', 'availability_cutoff', 'source',
        'shared_source_checksum', 'holdings', 'constraints', 'request', 'request_checksum',
        'local_transformation', 'native_sizing', 'code_versions'}
    if set(binding) != required or binding['protocol'] != PROTOCOL or binding['approach_id'] != 'jev_systemone':
        raise ValueError('invalid typed Jev forward binding')
    if find_storage_location(binding) is not None:
        raise ValueError('forward binding never carries storage locations')
    if binding['holdings'].get('mode') != 'paper' or not binding['holdings'].get('portfolio_id') or not isinstance(binding['holdings'].get('revision'), int):
        raise ValueError('isolated paper book and exact revision required')
    if checksum(binding['request']) != binding['request_checksum']:
        raise ValueError('prepared public request checksum mismatch')
    if instant(binding['source']['snapshot_created_at']) > instant(binding['availability_cutoff']):
        raise ValueError('snapshot was unavailable at original decision cutoff')
    if instant(binding['scheduled_slot']) > instant(binding['availability_cutoff']):
        raise ValueError('decision cutoff precedes scheduled slot')
    if list(binding['request']['questions']) != binding['source']['investable_universe']:
        # Canonical JSON sorts object keys; compare sets and preserve explicit universe order.
        if set(binding['request']['questions']) != set(binding['source']['investable_universe']):
            raise ValueError('public request universe mismatch')
    if binding['native_sizing'] != {'threshold': .20, 'max_delta': .10, 'confidence_minimum': .5, 'sales_first': True, 'purchases_cash_scaled': True}:
        raise ValueError('unsupported native sizing')
    return binding


def source_from_snapshot(content, market, binding):
    """Preserve the hourly adapter's actual snapshot availability and adjusted model prices."""
    source = binding['source']
    if content.snapshot.input_snapshot_id != source['input_snapshot_id'] or content.snapshot.manifest_checksum != source['checksum'] or content.snapshot.record['created_at'] != source['snapshot_created_at']:
        raise ValueError('approved snapshot identity changed')
    universe = source['investable_universe']
    observations = []
    for instrument in universe:
        for bar in market.bars_of(instrument):
            observations.append({'instrument_id': instrument, 'session_date': bar.session_date.isoformat(),
                'close': bar.close, 'open': bar.open, 'high': bar.high, 'low': bar.low, 'volume': bar.volume,
                'kind': bar.kind, 'available_at': source['snapshot_created_at']})
    return {**source, 'observations': observations}


def prepare_native(binding, source, *, ctx):
    """Verify original numeric preprocessing AND exact public bytes before any transport."""
    from finplan_model.forward.public_preflight import _view
    from finplan_model.benchmarks.jev import JevStrategy
    validate_binding(binding)
    market, view, holdings, private, raw = _view(source, binding['holdings'], binding['availability_cutoff'])
    if checksum(private) != checksum(binding['local_transformation']):
        raise ValueError('prepared numeric values/holdings/transformation no longer match original')
    class CapturingClient:
        model = binding['request']['model']
        def ask(self, payload, **kwargs):
            if canonical_json_bytes(payload) != canonical_json_bytes(binding['request']):
                raise ValueError('native public request differs from captured original')
            from finplan_model.datasets.outbound import check_outbound_payload
            check_outbound_payload(payload, ctx=ctx, raw_values=raw, destination='typesafe_systemone')
            raise _Prepared()
    try:
        JevStrategy(CapturingClient(), ctx=ctx, raw_values=raw).decide(view, holdings)
    except _Prepared:
        return view, holdings, raw
    raise ValueError('native strategy did not form a request')


class _Prepared(Exception):
    pass


def run_forward(inp, *, client_factory=None, context_reader=None):
    """Managed worker only. Immutable before-input receipt precedes secret/client creation."""
    from finplan_model.benchmarks.jev import JevStrategy
    from finplan_model.jobs.results import succeeded_result
    from finplan_model.forward.runtime import validate_policy
    from finplan_model.core.outcome import failed_job_result
    import time
    started = time.monotonic()
    descriptor = inp.spec['forward_jev']
    binding = validate_binding(inp.artifacts.get_json(descriptor['binding_ref']))
    if descriptor['binding_checksum'] != checksum(binding) or managed_id(binding) != inp.spec['run_id']:
        raise ValueError('managed forward binding identity mismatch')
    if descriptor.get('worker_image_digest') != inp.ctx.image_digest:
        raise ValueError('approved managed worker image differs from actual worker')
    if not descriptor.get('approval_verified') or descriptor.get('approved_by') is None:
        raise ValueError('managed forward job lacks independently verified approval')
    before_ref = None
    response_ref = None
    client = None
    fresh_claim_owned = False
    original_no_call_verified = False
    transport_calls = 0
    try:
        source = source_from_snapshot(inp.snapshot, inp.market, binding)
        view, holdings, raw = prepare_native(binding, source, ctx=inp.ctx)
        context_reader = context_reader or getattr(inp, 'forward_context_reader', None) or _owned_context_reader(inp)
        try: _, existing_claim = read_named(inp.artifacts, journal_id(binding, 'claim'))
        except FinplanError as missing:
            if missing.code != 'NOT_FOUND': raise
            existing_claim = None
        def eligibility():
            if instant(inp.ctx.now_ts()) >= instant(binding['availability_cutoff']) + timedelta(hours=1):
                raise FinplanError.precondition('approved forward request expired before vendor invocation', reason='forward_request_expired')
            head = context_reader(descriptor['context_id'])
            if not head or head.get('managed_run_id') != inp.spec['run_id'] or head.get('binding_checksum') != checksum(binding):
                raise FinplanError.precondition('forward context superseded before vendor invocation', reason='forward_context_superseded')
            verify_current_book(inp.platform, binding)
        if existing_claim is None:
            eligibility()
        before_ref = inp.artifacts.put_json({'protocol': PROTOCOL, 'kind': 'invocation_input',
            'managed_run_id': inp.spec['run_id'], 'binding_checksum': checksum(binding),
            'binding_ref': descriptor['binding_ref'], 'public_request': binding['request'],
            'request_checksum': binding['request_checksum'], 'captured_at': inp.ctx.now_ts(),
            'local_transformation': binding['local_transformation'], 'supplied_layer': 'public_request_only',
            'approval': {'approved_by': descriptor['approved_by'], 'approved_at': descriptor['approved_at']},
            'status': 'prepared_before_permanent_invocation_claim',
            'worker_image_digest': inp.ctx.image_digest, 'worker_source_commit': descriptor['worker_source_commit'],
            'origin_processing_code_versions': binding['code_versions'],
            'implementation_checksums': _implementation_checksums()},
            kind='run_artifact', domain='finance', synthetic=inp.ctx.synthetic)
        import uuid
        claim = {'protocol': PROTOCOL, 'binding_checksum': checksum(binding),
            'invocation_input_ref': before_ref.to_dict(), 'attempt_token': uuid.uuid4().hex,
            'claimed_at': inp.ctx.now_ts(), 'managed_run_id': inp.spec['run_id']}
        owned = inp.artifacts.create_once(canonical_json_bytes(claim), kind='run_artifact', artifact_id=journal_id(binding, 'claim'))
        fresh_claim_owned = owned
        recovered = not owned
        claim_ref, durable_claim = read_named(inp.artifacts, journal_id(binding, 'claim'))
        if durable_claim.get('binding_checksum') != checksum(binding) or durable_claim.get('managed_run_id') != inp.spec['run_id']:
            raise ValueError('permanent invocation claim binding mismatch')
        if recovered:
            from finplan_model.core.artifacts import ArtifactRef
            before_ref = ArtifactRef.from_dict(durable_claim['invocation_input_ref'])
            original_input = inp.artifacts.get_json(before_ref)
            if original_input.get('public_request') != binding['request'] or original_input.get('binding_checksum') != checksum(binding):
                raise ValueError('original claimed invocation input mismatch')
            try:
                response_ref, saved = read_named(inp.artifacts, journal_id(binding, 'response'))
            except FinplanError as exc:
                if exc.code == 'NOT_FOUND':
                    try: _, no_call = read_named(inp.artifacts, journal_id(binding, 'no_call'))
                    except FinplanError as absent:
                        if absent.code != 'NOT_FOUND': raise
                        no_call = None
                    if no_call and no_call.get('protocol') == PROTOCOL and no_call.get('managed_run_id') == inp.spec['run_id'] and no_call.get('binding_checksum') == checksum(binding) and no_call.get('invocation_state') == 'not_invoked':
                        proof = inp.artifacts.get_json(no_call['failure_ref'])
                        if proof.get('protocol') != PROTOCOL or proof.get('kind') != 'failed_forward_result' or proof.get('managed_run_id') != inp.spec['run_id'] or proof.get('binding_checksum') != checksum(binding) or proof.get('invocation_state') != 'not_invoked' or proof.get('worker_image_digest') != inp.ctx.image_digest or (proof.get('usage') or {}).get('transport_calls') != 0:
                            raise ValueError('original no-call proof invalid')
                        original_no_call_verified = True
                        raise FinplanError.precondition('original worker definitely did not invoke; a new independently approved binding is required', reason='forward_original_not_invoked') from None
                    raise FinplanError.precondition('prior paid invocation may have occurred; independent human resolution required', reason='forward_invocation_indeterminate') from None
                raise
            if saved.get('binding_checksum') != checksum(binding) or saved.get('request_checksum') != binding['request_checksum']:
                raise ValueError('saved native response belongs to another binding')
            client = _ReplayClient(binding['request']['model'], saved)
        else:
            eligibility()
            if client_factory is None:
                client_factory = _native_client
            client = client_factory(inp)
        if client.model != binding['request']['model']:
            raise ValueError('requested model differs from approved public request')
        transport_calls = 0
        if not recovered:
            native_transport = client.transport
            def eligible_transport(payload):
                nonlocal transport_calls
                eligibility()  # ALSO each bounded native retry, immediately before transport.
                transport_calls += 1
                return native_transport(payload)
            client.transport = eligible_transport
        def actual_usage():
            return client.usage if recovered else {**client.usage, 'transport_calls': transport_calls}
        # The wrapper checks each retry/cache access; native JevClient retains its egress/cap guards.
        class ExactClient:
            model = client.model
            def ask(self, payload, **kwargs):
                if canonical_json_bytes(payload) != canonical_json_bytes(binding['request']):
                    raise ValueError('transport payload differs from approved public request')
                nonlocal response_ref
                answer = client.ask(payload, **kwargs)
                if not recovered:
                    saved = {'protocol': PROTOCOL, 'binding_checksum': checksum(binding),
                        'request_checksum': binding['request_checksum'], 'public_response': answer,
                        'usage': actual_usage(), 'received_at': inp.ctx.now_ts(),
                        'managed_run_id': inp.spec['run_id'], 'invocation_input_ref': before_ref.to_dict(),
                        'invocation_claim_ref': claim_ref.to_dict()}
                    if not inp.artifacts.create_once(canonical_json_bytes(saved), kind='run_artifact', artifact_id=journal_id(binding, 'response')):
                        raise ValueError('native response journal unexpectedly already exists')
                    response_ref = inp.artifacts.resolve(kind='run_artifact', artifact_id=journal_id(binding, 'response'))
                return answer
        strategy = JevStrategy(ExactClient(), ctx=inp.ctx, raw_values=raw)
        target = strategy.decide(view, holdings)
        output = {'weights': dict(target.weights), 'cash': target.cash,
            'solution_status': target.solution_status, 'forecast': {'available': False, 'reason': 'classification is not a portfolio value forecast'}}
        validate_policy(output, binding['source']['investable_universe'], binding['holdings'])
        response = client.cache.get(binding['request_checksum'])
        if response is None:
            raise ValueError('native response capture missing')
        result_ref = inp.artifacts.put_json({'protocol': PROTOCOL, 'kind': 'forward_result',
            'managed_run_id': inp.spec['run_id'], 'binding_checksum': checksum(binding),
            'binding_ref': descriptor['binding_ref'], 'invocation_input_ref': before_ref.to_dict(),
            'request_checksum': binding['request_checksum'], 'public_response': response,
            'policy': output, 'usage': actual_usage(), 'issued_at': saved['received_at'] if recovered else inp.ctx.now_ts(),
            'signal_formed_at': inp.artifacts.get_json(response_ref)['received_at'],
            'response_journal_ref': response_ref.to_dict(), 'invocation_claim_ref': claim_ref.to_dict(), 'recovered_without_provider_call': recovered,
            'ledger_updated': False, 'execution_policy': 'only later eligible hourly opportunity; no retrospective fill'},
            kind='run_artifact', domain='finance', synthetic=inp.ctx.synthetic)
        result = succeeded_result(inp.ctx, inp.spec, solution_status=target.solution_status,
            artifacts=[before_ref, result_ref], performance={}, dataset_checksum=binding['source']['checksum'],
            instance_seconds=time.monotonic()-started)
        result['payload']['forward_jev'] = {'protocol': PROTOCOL, 'binding_checksum': checksum(binding),
            'result_ref': result_ref.to_dict(), 'ledger_updated': False, 'invocation_state': 'recovered_original_response' if recovered else 'invoked', 'usage': actual_usage()}
        return result
    except Exception as exc:
        reason = getattr(exc, 'details', {}).get('reason')
        # Only this worker's newly owned permanent claim proves its zero counter is
        # authoritative. Existing/unreadable claims may already have paid effects.
        definitely_no_call = original_no_call_verified or fresh_claim_owned and transport_calls == 0
        invocation_uncertain = not definitely_no_call and response_ref is None
        failure_state = 'not_invoked' if definitely_no_call else 'indeterminate' if invocation_uncertain else 'failed_invocation'
        failure_usage = actual_usage() if client is not None and 'actual_usage' in locals() else {**client.usage, 'transport_calls': transport_calls} if client is not None and fresh_claim_owned else {'calls': 0, 'transport_calls': 0} if definitely_no_call else None
        failure_ref = inp.artifacts.put_json({'protocol': PROTOCOL, 'kind': 'failed_forward_result',
            'managed_run_id': inp.spec['run_id'], 'binding_checksum': checksum(binding),
            'binding_ref': descriptor['binding_ref'], 'invocation_input_ref': before_ref.to_dict() if before_ref else None,
            'response_journal_ref': response_ref.to_dict() if response_ref else None,
            'status': 'failed', 'error_type': type(exc).__name__,
            'invocation_state': failure_state,
            'usage': failure_usage, 'worker_image_digest': inp.ctx.image_digest, 'worker_source_commit': descriptor['worker_source_commit'],
            'reason': getattr(exc, 'details', {}).get('reason', 'forward_integrity_or_invocation_failed'),
            'captured_at': inp.ctx.now_ts(), 'ledger_updated': False}, kind='run_artifact', domain='finance', synthetic=inp.ctx.synthetic)
        result = failed_job_result(inp.ctx, exc, run_id=inp.spec['run_id'])
        result['artifacts'] = ([before_ref.to_dict()] if before_ref else []) + [failure_ref.to_dict()]
        result.update(configuration_id=inp.spec['configuration_id'], input_snapshot_id=inp.spec['input_snapshot_id'])
        if fresh_claim_owned and definitely_no_call and before_ref is not None:
            inp.artifacts.create_once(canonical_json_bytes({'protocol': PROTOCOL, 'binding_checksum': checksum(binding),
                'managed_run_id': inp.spec['run_id'], 'invocation_state': 'not_invoked',
                'failure_ref': failure_ref.to_dict(), 'reason': reason, 'recorded_at': inp.ctx.now_ts()}),
                kind='run_artifact', artifact_id=journal_id(binding, 'no_call'))
        result['payload'] = {'forward_jev': {'protocol': PROTOCOL, 'binding_checksum': checksum(binding),
            'invocation_state': failure_state,
            'usage': failure_usage, 'response_journal_ref': response_ref.to_dict() if response_ref else None,
            'failure_ref': failure_ref.to_dict(), 'invocation_input_ref': before_ref.to_dict() if before_ref else None}}
        return result


def _native_client(inp):
    # Called ONLY inside the approved managed worker, never by the EC2 preflight/consumer.
    from finplan_model.benchmarks.jev import JevClient, SECRET
    from finplan_model.core.config import load_config
    import boto3
    config = load_config(inp.ctx.environment).raw['llm_benchmarks']['jev']
    value = boto3.Session(region_name='us-east-2').client('secretsmanager').get_secret_value(SecretId=SECRET)['SecretString']
    try:
        parsed = json.loads(value); value = parsed.get('api_key', parsed.get('key'))
    except ValueError:
        pass
    return JevClient(value, model=config['model'], max_calls=config['max_calls'], max_input_tokens=config['max_input_tokens'])


class _ReplayClient:
    def __init__(self, model, saved):
        self.model = model
        self.saved = saved
        self.cache = {}
        self.usage = saved['usage']

    def ask(self, payload, **kwargs):
        from finplan_model.benchmarks.jev import JevClient
        validation = JevClient('offline-response-validation', model=self.model,
            transport=lambda _: self.saved['public_response'], sleep=lambda _: None)
        result = validation.ask(payload, **kwargs)
        self.cache[checksum(payload)] = result
        return result


def _implementation_checksums():
    from pathlib import Path
    import finplan_model.benchmarks.jev as native
    import finplan_model.forward.public_preflight as preparation
    return {name: sha256_checksum(Path(path).read_bytes()) for name, path in
        (('forward_worker', __file__), ('native_jev', native.__file__), ('native_preparation', preparation.__file__))}


def _owned_context_reader(inp):
    import boto3
    from finplan_contracts import boundaries
    from finplan_model.control.store import DynamoRunStore
    session = boto3.Session(region_name='us-east-2')
    table = boundaries.resource_name(inp.ctx.environment, 'financemodel', 'job-control')
    return DynamoRunStore(session.client('dynamodb'), table).get_forward_context
