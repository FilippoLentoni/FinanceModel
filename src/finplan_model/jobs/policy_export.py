"""One bounded export of an evaluated strategy; never trains a learner."""
from __future__ import annotations

from finplan_model.core.errors import FinplanError
from finplan_model.core.ids import configuration_id
from finplan_model.rl.inference import export_actor
from .results import succeeded_result


def run_prepare_policy(inp):
    source = inp.spec.get('policy_source') or {}
    if inp.spec.get('purpose') != 'research' or not source.get('strategy_id'):
        raise FinplanError.precondition('a frozen succeeded research run is required', reason='policy_source_missing')
    members=[]
    for member in source.get('members', []):
        data=inp.artifacts.get(member['artifact'])
        members.append({'seed':member['seed'],'source_checksum':member['artifact']['checksum'],'actor':export_actor(data, source['strategy_id'])})
    bundle={**{k:v for k,v in source.items() if k != 'members'},'members':members,'mode':'advisory_paper','format':'finplan-strategy-bundle/1'}
    ref=inp.artifacts.put_json(bundle,kind='policy_inference',domain='finance',synthetic=True if inp.ctx.synthetic else None)
    doc=succeeded_result(inp.ctx,inp.spec,solution_status='not_applicable',artifacts=[ref],performance={},dataset_checksum=inp.market.dataset_checksum)
    doc['payload']['policy_export']={'source_run_id':source['source_run_id'],'member_count':len(members),'seeds':[m['seed'] for m in members],'format':bundle['format'],'mode':'advisory_paper'}
    return doc


def freeze_policy_source(result, spec, source_run_id, strategy_id='ppo'):
    summary=result.get('payload',{}).get('model_selection') or {}
    choices=summary.get('selection',{}).get('chosen_hyperparameters',{})
    if result.get('completion_status') != 'succeeded' or strategy_id not in choices:
        raise FinplanError.precondition('the strategy must have been evaluated in a succeeded source run',reason='policy_source_invalid')
    selected=choices[strategy_id]
    protocol=spec['selection_protocol']
    common={'source_run_id':source_run_id,'strategy_id':strategy_id,'instruments':sorted(spec['universe']),
            'training_end':protocol['splits']['train']['end'],'available_after':protocol['splits']['validation']['end'],
            'constraints':spec['simulation']['constraints'],'constraint_policy':spec['simulation'].get('constraint_policy','project')}
    if strategy_id not in ('ppo', 'sac'):
        from finplan_model.strategies import build_strategy
        strategy=build_strategy(strategy_id, selected)
        return {**common,'parameters':dict(strategy.params),'configuration_id':configuration_id({**common,'parameters':dict(strategy.params)}),'members':[]}
    seeds=selected.get('seeds')
    if result.get('completion_status')!='succeeded' or not seeds or selected.get('policy_selection')!='ensemble':
        raise FinplanError.precondition('source must be a succeeded PPO ensemble experiment',reason='policy_source_invalid')
    rows=[r for r in summary.get('policies',[]) if r['algorithm']==strategy_id and r.get('reward')==selected.get('reward')]
    if sorted(r['seed'] for r in rows)!=sorted(seeds) or sorted(seeds) != sorted(protocol['rl']['seeds']):
        raise FinplanError.precondition('the complete selected seed ensemble is required',reason='policy_members_missing')
    refs={a['artifact_id']:a for a in result['artifacts']}
    members=[]
    for row in sorted(rows,key=lambda x:x['seed']):
        ref=refs.get(row['artifact_id'])
        if not ref or ref['kind']!='rl_policy' or ref['checksum']!=row['checksum']:
            raise FinplanError.precondition('policy metadata does not match the artifact references',reason='policy_checksum_mismatch')
        members.append({'seed':row['seed'],'artifact':ref})
    env=summary['rl'][strategy_id]['environment']
    env={k:v for k,v in env.items() if k!='reward_formula'}
    return {**common,'configuration_id':selected['configuration_id'],'environment':env,'members':members}
