"""Immutable model identity, separate from actual data/latency/code metadata."""
import copy

import pytest

from finplan_model.forward.model_freeze import freeze_experiment_models


def approach(aid='sac', **changes):
    return {'approach_id':aid,'family':'reinforcement_learning','adapter':'actor','version':'1',
            'native_cadence':'completed_daily','news_layer':'not_consumed','strategy_name':None,
            'parameters':None,'input_capabilities':{'news':'not_consumed'},**changes}


def policy(cfg='cfg_controlled_1',weight=1.):
    return {'configuration_id':cfg,'strategy_id':'sac','instruments':['A'],
            'environment':{'window':20,'features':['log_return_window','current_weights']},
            'members':[{'seed':0,'actor':{'weight':weight}}]}


def initialize():
    return freeze_experiment_models(None,[approach()],{'sac':policy()})


def test_first_registry_derives_identity_from_actual_bytes_not_friendly_label():
    result=initialize();row=result['approaches'][0]
    assert row['version']=='cfg_controlled_1' and row['policy_checksum'].startswith('sha256:')
    assert result['blockers']=={}
    restored=freeze_experiment_models({'approaches':result['approaches']},result['approaches'],{'sac':policy()})
    assert restored['policies']==result['policies'] and restored['blockers']=={}


@pytest.mark.parametrize('changed',[policy(weight=2.),policy(cfg='cfg_controlled_2')])
def test_canonical_replacement_cannot_execute_or_relabel_original_book(changed):
    original=initialize();frozen=copy.deepcopy(original['approaches'])
    result=freeze_experiment_models({'approaches':frozen},[approach()],{'sac':changed})
    assert result['policies']=={} and result['approaches']==frozen
    block=result['blockers']['sac']
    assert block['reason']=='frozen_model_identity_changed' and block['invocation_state']=='not_invoked'
    assert block['expected_identity']['policy_checksum']!=block['observed_identity']['policy_checksum']
    assert original['approaches']==frozen


def test_newly_available_policy_does_not_silently_bind_previously_unavailable_variant():
    original={'approaches':[approach()]}
    result=freeze_experiment_models(original,[approach()],{'sac':policy()})
    assert result['blockers']['sac']['reason']=='frozen_model_identity_changed'
    assert result['policies']=={}


def test_missing_bound_artifact_remains_accounted_with_original_identity():
    original=initialize()
    result=freeze_experiment_models({'approaches':original['approaches']},original['approaches'],{})
    assert result['blockers']['sac']['reason']=='frozen_policy_artifact_unavailable'
    assert result['approaches']==original['approaches']


def test_new_inventory_keeps_every_original_slot_and_exposes_need_for_new_experiment():
    original=initialize()
    prospective=original['approaches']+[approach('sac_candidate')]
    result=freeze_experiment_models({'approaches':original['approaches']},prospective,{'sac':policy(),'sac_candidate':policy(cfg='cfg_candidate')})
    assert [r['approach_id'] for r in result['approaches']]==['sac']
    assert result['inventory_changes']['new_approach_ids']==['sac_candidate']
    assert result['inventory_changes']['unregistered_policy_ids']==['sac_candidate']
    assert result['blockers']=={} and list(result['policies'])==['sac']


def test_missing_inventory_entry_still_returns_registered_approach_for_hourly_failure():
    original=initialize()
    result=freeze_experiment_models({'approaches':original['approaches']},[],{})
    assert result['blockers']['sac']['reason']=='registered_approach_missing_from_inventory'
    assert result['approaches']==original['approaches']


def test_code_and_data_updates_do_not_replace_frozen_policy_identity():
    original=initialize();prospective=copy.deepcopy(original['approaches'])
    prospective[0].update(code_versions={'financemodel':'new_bugfix_commit'},trigger='systemd_scheduled',portfolio_id='existing_book')
    result=freeze_experiment_models({'approaches':original['approaches']},prospective,{'sac':policy()})
    assert result['blockers']=={} and result['policies']['sac']==policy()


def test_changed_solver_parameters_block_that_approach_without_blocking_unaffected_control():
    rows=[approach('optimizer',adapter='classical',family='optimization',parameters={'lookback':60}),approach('cash',adapter='baseline',family='control')]
    changed=copy.deepcopy(rows);changed[0]['parameters']['lookback']=120
    result=freeze_experiment_models({'approaches':rows},changed,{})
    assert list(result['blockers'])==['optimizer']
    assert result['blockers']['optimizer']['changed_fields']==['parameters']
    assert result['approaches']==rows


def test_duplicate_registry_rows_fail_before_any_model_invocation():
    with pytest.raises(ValueError,match='duplicate approach'):
        freeze_experiment_models(None,[approach(),approach()],{})
