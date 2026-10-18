"""Trusted beta Qwen producer; public submit cannot attach the private binding."""
from __future__ import annotations
from finplan_model.control.forward_jev import authenticated_operator, require_operator
from finplan_model.core.errors import contract_version, FinplanError
from finplan_model.benchmarks.forward_qwen import validate_binding, checksum, PROTOCOL


def submission(binding, *, dry_run):
    validate_binding(binding)
    return {'domain':'finance','domain_schema_version':'1.0','job_type':'swarm_mode_a',
        'purpose':'research','dry_run':dry_run,'contract_version':contract_version(),
        'input_snapshot_id':binding['source']['input_snapshot_id'],
        'configuration':{'domain':'finance','domain_schema_version':'1.0','payload':{
            'strategy':'qwen_swarm','objective':'forward_paper_decision','universe':binding['source']['investable_universe'],
            'rebalance_frequency':'daily','lookback_days':60,'constraints':{'long_only':True,'max_weight':1.}}},
        'idempotency_key':'forward-qwen-'+checksum(binding).split(':')[1],'max_runtime_seconds':900}


def verified_binding(evidence, experiment, origin_run_id):
    """Resolve authoritative captured bytes and scoped heads, never caller provenance."""
    from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
    from finplan_model.benchmarks.forward_qwen import make_binding, instant
    run = evidence.read_run_header(origin_run_id)
    if not run or run.get('experiment_id') != experiment or run.get('approach_id') != 'qwen_swarm_mode_a' or run.get('run_id') != origin_run_id:
        raise ValueError('original run belongs to another experiment/approach')
    captured = evidence.read_input(run['input_id'])
    metadata, payload = captured['metadata'], captured['payload']
    if metadata.get('input_id') != run['input_id'] or metadata.get('kind') != 'model_input' or metadata.get('experiment_id') != experiment or metadata.get('approach_id') != 'qwen_swarm_mode_a' or metadata.get('slot') != run['slot'] or payload.get('run_id') != origin_run_id or payload.get('input_id') != run['input_id'] or payload.get('approach_id') != 'qwen_swarm_mode_a' or payload.get('slot') != run['slot']:
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
    if book.get('experiment_id') != experiment or book.get('approach_id') != 'qwen_swarm_mode_a' or book.get('isolated_experiment') is not True or book['revision'] != payload['holdings']['revision'] or run['portfolio_id'] != payload['holdings']['portfolio_id']:
        raise ValueError('original/current isolated book binding mismatch')
    return make_binding(payload, shared['payload'], experiment_id=experiment, captured_checksum=metadata['checksum'])



def predecessor(service,binding):
    from datetime import timedelta
    from finplan_model.benchmarks.forward_qwen import context_identity,managed_id,instant,read_named,journal_id
    head=service.store.get_forward_context(context_identity(binding))
    if head is None:return None
    if head['managed_run_id']==managed_id(binding):
        if head['binding_checksum']!=checksum(binding):raise ValueError('permanent Qwen context binding conflict')
        return head['managed_run_id']
    prior=service.store.get_run(head['managed_run_id'])
    if prior is None:raise ValueError('permanent Qwen predecessor is missing')
    old=service.d.artifacts.get_json(head['binding_ref'])
    if checksum(old)!=head['binding_checksum'] or context_identity(old)!=context_identity(binding):raise ValueError('permanent Qwen context mismatch')
    if service.now()<instant(old['availability_cutoff'])+timedelta(hours=1):
        raise FinplanError.precondition('original Qwen context remains eligible',reason='forward_context_pending')
    if not prior.get('job_started') and int(prior.get('attempt',0))==0:
        return head['managed_run_id']
    # Missing/partial offline output can never prove no GPU model execution.
    if prior['state']=='succeeded':
        section=(prior.get('result') or {}).get('payload',{}).get('forward_qwen',{})
        output=service.d.artifacts.get_json(section['result_ref'])
        if output.get('protocol')!=PROTOCOL or output.get('managed_run_id')!=prior['run_id'] or output.get('binding_checksum')!=checksum(old):raise ValueError('Qwen predecessor output mismatch')
        roles=output.get('role_receipts') or []
        if not roles:raise ValueError('Qwen predecessor original response receipts absent')
        responses=[service.d.artifacts.get_json(role['response_ref']) for role in roles]
        if any(response.get('managed_run_id')!=prior['run_id'] or response.get('binding_checksum')!=checksum(old) for response in responses):raise ValueError('Qwen predecessor original response binding differs')
        signal_formed_at=max(instant(response['received_at']) for response in responses)
        if service.now()>=signal_formed_at+timedelta(days=1):return head['managed_run_id']
        raise FinplanError.precondition('approved Qwen native daily signal still available',reason='forward_native_signal_available')
    raise FinplanError.precondition('prior offline GPU outcome indeterminate; no automatic relaunch',reason='forward_context_indeterminate')
