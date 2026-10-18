"""Typed beta backend producer. No public contract extension and no approval shortcut."""
from __future__ import annotations

from finplan_model.control.auth import Principal
from finplan_model.core.errors import ErrorCode, FinplanError, contract_version
from finplan_model.benchmarks.forward_jev import OBJECTIVE, checksum, validate_binding


def require_operator(cfg, principal):
    if cfg.env != 'beta' or principal.role_name not in cfg.dev_desktop_role_names:
        raise FinplanError(ErrorCode.FORBIDDEN, 'trusted forward producer is limited to the configured beta development role')


def authenticated_operator(session, cfg):
    """The executable derives identity from real AWS credentials; no caller ARN argument."""
    principal = Principal.from_arn(session.client('sts').get_caller_identity()['Arn'])
    require_operator(cfg, principal)
    return principal


def submission(binding, *, dry_run):
    validate_binding(binding)
    return {'domain': 'finance', 'domain_schema_version': '1.0', 'job_type': 'jev_backtest',
        'purpose': 'research', 'dry_run': dry_run, 'contract_version': contract_version(),
        'input_snapshot_id': binding['source']['input_snapshot_id'],
        'configuration': {'domain': 'finance', 'domain_schema_version': '1.0',
            'payload': {'strategy': 'jev', 'objective': OBJECTIVE,
                'universe': binding['source']['investable_universe'], 'rebalance_frequency': 'daily',
                'lookback_days': 60, 'constraints': {'long_only': True, 'max_weight': 1.}}},
        'idempotency_key': 'forward-jev-' + checksum(binding).split(':')[1],
        'max_runtime_seconds': 900}


def verified_binding(evidence, experiment, origin_run_id):
    """Resolve authoritative captured bytes and scoped heads, never caller provenance."""
    from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
    from finplan_model.benchmarks.forward_jev import make_binding, instant
    run = evidence.read_run_header(origin_run_id)
    if not run or run.get('experiment_id') != experiment or run.get('approach_id') != 'jev_systemone' or run.get('run_id') != origin_run_id:
        raise ValueError('original run belongs to another experiment/approach')
    captured = evidence.read_input(run['input_id'])
    metadata, payload = captured['metadata'], captured['payload']
    if metadata.get('input_id') != run['input_id'] or metadata.get('kind') != 'model_input' or metadata.get('experiment_id') != experiment or metadata.get('approach_id') != 'jev_systemone' or metadata.get('slot') != run['slot'] or payload.get('run_id') != origin_run_id or payload.get('input_id') != run['input_id'] or payload.get('approach_id') != 'jev_systemone' or payload.get('slot') != run['slot']:
        raise ValueError('original run/input association mismatch')
    shared_id = 'mi_' + sha256_checksum(canonical_json_bytes([experiment,'__shared__',run['slot'],'shared_source'])).split(':')[1]
    shared = evidence.read_input(shared_id)
    if shared['metadata'].get('experiment_id') != experiment or shared['metadata'].get('kind') != 'shared_source' or shared['metadata'].get('slot') != run['slot'] or shared['metadata']['checksum'] != payload['shared_source_checksum'] or sha256_checksum(canonical_json_bytes(shared['payload'])) != shared['metadata']['checksum']:
        raise ValueError('original shared source association/checksum mismatch')
    expected_ref = {'input_id': shared_id, 'checksum': shared['metadata']['checksum']}
    original_ref = metadata.get('shared_source_ref') or {}
    if original_ref.get('input_id') != expected_ref['input_id'] or original_ref.get('checksum') != expected_ref['checksum']:
        raise ValueError('captured input shared-source reference does not match authoritative source')
    if metadata.get('captured_at') and instant(metadata['captured_at']) < instant(payload['availability_cutoff']):
        raise ValueError('captured input cutoff exceeds original capture time')
    if run.get('started_at') and metadata.get('captured_at') and instant(run['started_at']) < instant(metadata['captured_at']):
        raise ValueError('original invocation started before captured input')
    book = evidence.get_portfolio(run['portfolio_id'])
    if book.get('experiment_id') != experiment or book.get('approach_id') != 'jev_systemone' or book.get('isolated_experiment') is not True or book['revision'] != payload['holdings']['revision'] or run['portfolio_id'] != payload['holdings']['portfolio_id']:
        raise ValueError('original/current isolated book binding mismatch')
    return make_binding(payload, shared['payload'], experiment_id=experiment, captured_checksum=metadata['checksum'])


def predecessor(service, binding):
    """Allow a fresh, separately approved request only after a safe permanent predecessor."""
    from datetime import timedelta
    from finplan_model.benchmarks.forward_jev import context_identity,managed_id,instant,read_named,journal_id,checksum
    head=service.store.get_forward_context(context_identity(binding))
    if head is None:return None
    if head['managed_run_id']==managed_id(binding):
        if head['binding_checksum']!=checksum(binding):raise ValueError('permanent context binding conflict')
        return head['managed_run_id']
    prior=service.store.get_run(head['managed_run_id'])
    if prior is None:raise ValueError('permanent context predecessor head is missing')
    old=service.d.artifacts.get_json(head['binding_ref'])
    if checksum(old)!=head['binding_checksum'] or context_identity(old)!=context_identity(binding):raise ValueError('permanent context predecessor binding mismatch')
    if service.now()<instant(old['availability_cutoff'])+timedelta(hours=1):
        raise FinplanError.precondition('original exact-context request is still eligible; use its original proposal',reason='forward_context_pending')
    section=(prior.get('result') or {}).get('payload',{}).get('forward_jev',{})
    usage=section.get('usage')
    if not prior.get('job_started') and int(prior.get('attempt',0))==0:
        return head['managed_run_id']
    if section.get('invocation_state')=='not_invoked' and isinstance(usage,dict) and usage.get('transport_calls',usage.get('calls'))==0:
        proof=service.d.artifacts.get_json(section['failure_ref'])
        if proof.get('protocol')!='finplan-jev-forward/1' or proof.get('kind')!='failed_forward_result' or proof.get('managed_run_id')!=prior['run_id'] or proof.get('binding_checksum')!=checksum(old) or proof.get('worker_image_digest')!=prior['forward_jev']['worker_image_digest'] or proof.get('invocation_state')!='not_invoked' or (proof.get('usage') or {}).get('transport_calls',(proof.get('usage') or {}).get('calls'))!=0:
            raise ValueError('predecessor no-call proof invalid')
        return head['managed_run_id']
    if prior['state'] not in ('succeeded','failed','cancelled','timed_out'):
        raise FinplanError.precondition('previous managed context still active',reason='forward_context_pending')
    try:_,claim=read_named(service.d.artifacts,journal_id(old,'claim'))
    except FinplanError as exc:
        if exc.code!='NOT_FOUND':raise
        raise FinplanError.precondition('missing original invocation artifacts cannot prove no paid call',reason='forward_context_indeterminate') from None
    if claim.get('binding_checksum')!=checksum(old):raise ValueError('predecessor claim mismatch')
    try:_,response=read_named(service.d.artifacts,journal_id(old,'response'))
    except FinplanError as exc:
        if exc.code!='NOT_FOUND':raise
        raise FinplanError.precondition('unknown prior paid outcome requires independent human resolution',reason='forward_context_indeterminate') from None
    if response.get('binding_checksum')!=checksum(old):raise ValueError('predecessor response mismatch')
    if service.now()<instant(response['received_at'])+timedelta(days=1):
        raise FinplanError.precondition('approved native daily signal already available or still under review',reason='forward_native_signal_available')
    return head['managed_run_id']
