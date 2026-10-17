"""Actual selected actor identity and fixed universe determine canonical assignment."""
import copy

from finplan_model.forward.hourly import assign_policy_variants
from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum


def bundle(algorithm='ppo',universe=None,seed=0,selected=None,news=False):
    row={'strategy_id':algorithm,'configuration_id':'cfg_risk_'+('2' if seed>=5 else '05'),
        'source_run_id':'controlled_run','instruments':universe or ['A','B'],
        'environment':{'window':20,'features':['log_return_window','current_weights']+(['gdelt_intraday_organization_tone_v1'] if news else [])},
        'members':[{'seed':seed%5,'actor':{'algorithm':algorithm,'controlled_weights':[float(seed)]}}]}
    if selected is not None:row['evaluated_variant']={'selected_in_original_run':selected,'seed':seed%5}
    return row


def test_selected_sac_seed_canonical_and_other_nine_candidates_keep_distinct_books_without_alias():
    primary=bundle();actors=[bundle('sac',seed=n,selected=n==0) for n in range(10)]
    original=copy.deepcopy(actors)
    result=assign_policy_variants(primary,actors+[copy.deepcopy(actors[0])])
    assert result['sac']==actors[0] and result['ppo']==primary
    assert len(result)==11
    assert len([key for key in result if key.startswith('sac_policy_')])==9
    assert sum(canonical_json_bytes(value)==canonical_json_bytes(actors[0]) for value in result.values())==1
    assert len({sha256_checksum(canonical_json_bytes(value)) for value in result.values()})==11
    assert actors==original


def test_compatible_eight_asset_news_is_honest_canonical_variant_with_original_shape():
    ids=['A','B','C','D','E','F','G','H'];primary=bundle(universe=ids)
    news=bundle(universe=ids,news=True)
    result=assign_policy_variants(primary,[news,copy.deepcopy(news)])
    assert set(result)=={'ppo','ppo_gdelt_intraday'}
    assert result['ppo_gdelt_intraday']['environment']==news['environment']
    assert result['ppo_gdelt_intraday']['instruments']==ids


def test_incompatible_universes_keep_separate_candidates_instead_of_overwriting_canonical():
    primary=bundle(universe=['A','B']);incompatible=bundle('sac',universe=['C','D'],selected=True)
    compatible=bundle('sac',universe=['A','B'],selected=True)
    result=assign_policy_variants(primary,[incompatible,compatible])
    assert result['sac']==compatible and result['ppo']==primary
    extras=[value for key,value in result.items() if key.startswith('sac_policy_')]
    assert extras==[incompatible]


def test_unselected_candidates_never_implicitly_claim_selected_canonical_identity():
    primary=bundle();candidate=bundle('sac',selected=False)
    result=assign_policy_variants(primary,[candidate])
    assert 'sac' not in result and len(result)==2
    assert any(key.startswith('sac_policy_') for key in result)


def test_eight_asset_primary_prefers_original_selected_ensemble_over_new_seed_exports():
    from finplan_model.forward.hourly import select_eight_asset_primary
    ids=list('ABCDEFGH')
    ensemble=bundle(universe=ids);ensemble['members']=[{'seed':n} for n in range(5)]
    ensemble['source_run_id']='original_run';ensemble['configuration_id']='cfg_z'
    seed=bundle(universe=ids,selected=True);seed['source_run_id']='newer_run';seed['configuration_id']='cfg_a'
    newer=copy.deepcopy(ensemble);newer['source_run_id']='z_newer_selected_run'
    news=bundle(universe=ids,news=True);news['members']=[{'seed':n} for n in range(5)]
    sac=bundle('sac',universe=ids);sac['members']=[{'seed':n} for n in range(5)]
    before=copy.deepcopy([seed,ensemble,newer,news,sac])
    primary=select_eight_asset_primary(before)
    assert primary==newer and before==[seed,ensemble,newer,news,sac]
    assigned=assign_policy_variants(primary,[seed,ensemble,newer,news,sac])
    assert assigned['ppo']==newer
    assert sum(canonical_json_bytes(p)==canonical_json_bytes(seed) for p in assigned.values())==1
    assert sum(canonical_json_bytes(p)==canonical_json_bytes(ensemble) for p in assigned.values())==1
