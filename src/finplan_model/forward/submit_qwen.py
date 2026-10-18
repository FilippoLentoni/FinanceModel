"""Prepare/dry-run or submit ONE current-book Qwen request to human approval.

No model invocation, approval, or managed compute creation occurs in this command.
Default is dry-run. --submit records an awaiting_approval job and immutable inputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main(argv=None, *, session=None, evidence=None, service=None):
    from finplan_contracts import boundaries
    from finplan_model.core.config import load_config
    from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum, ArtifactRef
    from finplan_model.control.forward_qwen import authenticated_operator
    from finplan_model.benchmarks.forward_qwen import make_binding, checksum, managed_id, instant
    from finplan_model.forward.hourly import identifier
    from finplan_model.control.handlers import build_service
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--slot', required=True)
    parser.add_argument('--submit', action='store_true', help='record awaiting human approval; NEVER approves or launches')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if session is None:
        import boto3
        session = boto3.Session(region_name='us-east-2')
    cfg = load_config('beta')
    principal = authenticated_operator(session, cfg)
    slot = instant(args.slot).isoformat().replace('+00:00','Z')
    if evidence is None:
        from finplan_platform.core.modeling_evidence import ModelingEvidence
        evidence = ModelingEvidence.from_ec2(region='us-east-2')
    run = evidence.read_run(identifier('mr', args.experiment, 'qwen_swarm_mode_a', slot))
    captured = evidence.read_input(run['input_id'])
    shared = evidence.read_input(identifier('mi', args.experiment, '__shared__', slot, 'shared_source'))
    if captured['metadata']['experiment_id'] != args.experiment or captured['metadata']['approach_id'] != 'qwen_swarm_mode_a' or captured['metadata']['slot'] != slot or run['portfolio_id'] != captured['payload']['holdings']['portfolio_id']:
        raise ValueError('original run/input/experiment/book association mismatch')
    if captured['payload']['shared_source_checksum'] != shared['metadata']['checksum']:
        raise ValueError('original input/shared source checksum mismatch')
    book = evidence.get_portfolio(run['portfolio_id'])
    if book['revision'] != captured['payload']['holdings']['revision'] or book.get('experiment_id') != args.experiment or book.get('approach_id') != 'qwen_swarm_mode_a':
        raise ValueError('current isolated book no longer matches original prepared request')
    binding = make_binding(captured['payload'], shared['payload'], experiment_id=args.experiment, captured_checksum=captured['metadata']['checksum'])
    if service is None:
        service = build_service('beta', session=session, runs_table=boundaries.resource_name('beta','financemodel','job-control'))
    status, response = service.submit_forward_qwen(principal, binding, dry_run=not args.submit,
        correlation_id='forward-qwen-'+checksum(binding).split(':')[1][:32], origin_evidence=evidence)
    public_bytes = canonical_json_bytes(binding['request'])
    result = {'http_status': status, 'response': response, 'managed_run_id': managed_id(binding),
        'binding_checksum': checksum(binding), 'binding_persisted': args.submit,
        'binding_ref': {'artifact_id': 'run_artifact_'+checksum(binding).split(':')[1][:40], 'kind':'run_artifact', 'owner':'financemodel', 'checksum':checksum(binding),'content_type':'application/json'},
        'public_request': binding['request'], 'public_request_checksum':sha256_checksum(public_bytes),
        'origin_run_id': binding['origin_run_id'], 'origin_input_id':binding['origin_input_id'],
        'portfolio_id':binding['holdings']['portfolio_id'], 'holdings_revision':binding['holdings']['revision'],
        'model_invoked':False, 'compute_started':False, 'approval_granted':False,
        'caps':cfg.raw['llm_benchmarks']['qwen'],
        'approval_command':'python scripts/approve_run.py --env beta '+managed_id(binding),
        'execution_mode':'forward_one_decision', 'worker_capability_published':(service.d.settings.job_definition('swarm_mode_a') or {}).get('forward_protocol')=='finplan-qwen-forward/1', 'approval_command_cwd':'FinanceModel repository root',
        'warning':'Only a separate authorized human may approve. Each approved request is one approved offline GPU decision, not recurring vendor authorization.'}
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n');args.output.chmod(0o600)
    print(json.dumps(result,indent=2))
    return 0 if status in (200,202) else 1


if __name__=='__main__':
    raise SystemExit(main())
