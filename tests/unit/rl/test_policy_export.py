import copy

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.jobs.policy_export import freeze_policy_source


def source():
    refs = [{'artifact_id':f'policy_seed_{seed}', 'kind':'rl_policy', 'checksum':'sha256:'+str(seed)*64}
            for seed in range(5)]
    reward = {'risk_penalty':.5}
    result = {'completion_status':'succeeded', 'artifacts':refs,
              'payload':{'model_selection':{
                  'selection':{'chosen_hyperparameters':{
                      'ppo':{'configuration_id':'cfg_selected','seeds':list(range(5)), 'policy_selection':'ensemble', 'reward':reward},
                      'min_variance':{'lookback':60}}},
                  'policies':[{'algorithm':'ppo','seed':seed,'artifact_id':ref['artifact_id'], 'checksum':ref['checksum'], 'reward':reward}
                              for seed,ref in enumerate(refs)],
                  'rl':{'ppo':{'environment':{'window':60,'reward_formula':'training-only'}}}}}}
    spec = {'universe':['B','A'], 'simulation':{'constraints':{'max_weight':.6}},
            'selection_protocol':{'splits':{'train':{'end':'2025-12-31'},'validation':{'end':'2026-06-30'}},
                                  'rl':{'seeds':list(range(5))}}}
    return result, spec


def test_export_freezes_complete_ensemble_and_selection_availability():
    result, spec = source()
    frozen = freeze_policy_source(result, spec, 'run_source')
    assert frozen['instruments'] == ['A','B'] and len(frozen['members']) == 5
    assert frozen['available_after'] == '2026-06-30'
    assert frozen['configuration_id'] == 'cfg_selected'
    assert 'reward_formula' not in frozen['environment']
    assert frozen['constraints'] == spec['simulation']['constraints']


@pytest.mark.parametrize('failure', ['failed_run','missing_member','checksum','unknown_strategy','single_seed'])
def test_export_rejects_sources_that_were_not_evaluated_as_selected(failure):
    result, spec = source()
    strategy = 'ppo'
    if failure == 'failed_run': result['completion_status'] = 'failed'
    if failure == 'missing_member': result['payload']['model_selection']['policies'].pop()
    if failure == 'checksum': result['artifacts'][0]['checksum'] = 'sha256:'+'f'*64
    if failure == 'unknown_strategy': strategy = 'unknown'
    if failure == 'single_seed': result['payload']['model_selection']['selection']['chosen_hyperparameters']['ppo']['policy_selection'] = 'best_seed'
    with pytest.raises(FinplanError):
        freeze_policy_source(result, spec, 'run_source', strategy)


def test_baseline_freezes_evaluated_parameters_without_policy_members():
    result, spec = source()
    frozen = freeze_policy_source(result, spec, 'run_source', 'min_variance')
    assert frozen['strategy_id'] == 'min_variance'
    assert frozen['parameters']['lookback'] == 60 and frozen['members'] == []
    assert frozen['configuration_id'].startswith('cfg_')
    again = freeze_policy_source(copy.deepcopy(result), copy.deepcopy(spec), 'run_source', 'min_variance')
    assert frozen == again
