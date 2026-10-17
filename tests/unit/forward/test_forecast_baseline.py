"""Forecast baselines are actual captured book values, never model-chosen capital."""
import copy

import pytest

from finplan_model.forward.forecasts import freeze,reconcile

ISSUED='2026-10-19T14:05:00Z'
TARGET='2026-10-19T15:05:00Z'


def forecast(**changes):
    return {'status':'forecast','target_at':TARGET,'expected_value':1100.,'baseline_value':1000.,'units':'USD',
            'assumptions':['Controlled fixture; no claim of live performance'],**changes}


def freeze_at_actual_book(f,**changes):
    return freeze(f,decision_at=ISSUED,run_id='controlled_run',portfolio_id='controlled_book',baseline_value=1000.,**changes)


@pytest.mark.parametrize('claimed',[100.,999.,1000.01,100000.])
def test_model_cannot_invent_different_capital_baseline(claimed):
    with pytest.raises(ValueError,match='baseline differs'):
        freeze_at_actual_book(forecast(baseline_value=claimed))


def test_true_baseline_is_frozen_without_mutating_supplied_forecast():
    supplied=forecast();original=copy.deepcopy(supplied)
    frozen=freeze_at_actual_book(supplied)
    assert frozen['baseline_value']==1000. and frozen['status']=='pending'
    assert frozen['decision_at']==ISSUED and frozen['next_eligible_observation_at']==TARGET
    assert supplied==original
    supplied['assumptions'].append('Later mutation')
    assert frozen['assumptions']==original['assumptions']


def test_native_source_cutoff_cannot_backdate_forecast_issued_after_its_target():
    with pytest.raises(ValueError,match='target must follow actual decision'):
        freeze_at_actual_book(forecast(target_at='2026-10-19T14:03:00Z'))


def test_known_baseline_flow_adjustment_and_first_valid_target_alignment():
    frozen=freeze_at_actual_book(forecast());original=copy.deepcopy(frozen)
    def observation(oid,time,value,kind='valid',book='controlled_book'):
        return {'observation_id':oid,'portfolio_id':book,'observed_at':time,'value':value,
                'observation_kind':kind,'quote_status':'fresh' if kind=='valid' else 'stale',
                'net_external_cash_flows_since_decision':50.}
    rows=[observation('early','2026-10-19T15:04:59Z',9999.),
          observation('stale',TARGET,9999.,kind='stale'),
          observation('other',TARGET,9999.,book='other_book'),
          observation('first','2026-10-19T15:06:00Z',1150.),
          observation('later','2026-10-19T15:07:00Z',1300.)]
    assert reconcile(frozen,rows[:3]) is None
    matched=reconcile(frozen,rows)
    assert matched['observation_id']=='first' and matched['flow_adjusted_realized_value']==1100.
    assert matched['signed_error']==0. and matched['absolute_error']==0.
    assert frozen==original
    assert reconcile(frozen,rows+[observation('new_later','2026-10-19T15:08:00Z',900.)],previous_match=matched)==matched


def test_unavailable_forecast_remains_unavailable_without_inventing_baseline():
    result=freeze_at_actual_book({'status':'not_available','available':False})
    assert result['available'] is False and 'baseline_value' not in result


@pytest.mark.parametrize('submitted,actual',[
    (forecast(baseline_value=999.),1000.),
    (forecast(),None),
    (forecast(target_at='2026-10-19T14:04:59Z'),1000.),
])
def test_invalid_forecast_retains_original_public_output_and_rejection_evidence(submitted,actual):
    from finplan_model.forward.forecasts import freeze_outcome
    original={'status':'succeeded','target_weights':[{'instrument_id':'A','weight':.5}],'cash_weight':.5,
        'forecast':submitted,'provider_final_response_text':'controlled original public response',
        'reviewer_invocation':{'model':'controlled_fixture','tool_attempts':[]}}
    saved=copy.deepcopy(original)
    outcome=freeze_outcome(original,decision_at=ISSUED,run_id='controlled_run',portfolio_id='controlled_book',baseline_value=actual)
    assert outcome['status']=='failed' and outcome['reason'].startswith('forecast_validation_failed:')
    assert outcome['rejected_forecast']==submitted and outcome['forecast']['available'] is False
    assert outcome['provider_final_response_text']==saved['provider_final_response_text']
    assert outcome['reviewer_invocation']==saved['reviewer_invocation']
    assert outcome['target_weights']==saved['target_weights'] and original==saved


def test_one_invalid_forecast_does_not_drop_later_approach_slots_or_valuation_accounting(tmp_path):
    from finplan_model.forward.hourly import run_hourly,identifier
    from finplan_model.forward.registry import APPROACHES
    from tests.unit.forward.test_hourly_recovery import FakeEvidence,source
    backend=FakeEvidence();shared=source()
    shared['registry_approaches']=[a.to_dict() for a in APPROACHES if a.approach_id in ('direct_codex_news','hosted_langgraph')]
    supplied={'status':'forecast','target_at':'2026-07-11T01:00:00Z','expected_value':110000.,'baseline_value':999.,'units':'USD'}
    calls=[]
    def invalid(base,source,holdings,before):
        calls.append('invalid')
        before({'boundary':'controlled_forecast_fixture','actual_source':{'news':[]},'messages':[]})
        return {'weights':{'A':.5,'B':.5},'cash':0.,'forecast':supplied,'provider_final_response_text':'controlled invalid-baseline final'}
    def following(base,source,holdings,before):
        calls.append('following')
        before({'boundary':'controlled_following_fixture'})
        return {'weights':{'A':0.,'B':0.},'cash':1.,'forecast':{'available':False}}
    def execute():return run_hourly('2026-07-11T00:00:00Z','controlled-forecast-accounting',tmp_path,backend=backend,
        source=shared,policies={},research={},now='2026-07-11T00:05:00Z',external_adapters={'codex':invalid,'langgraph':following})
    result=execute()
    assert calls==['invalid','following']
    assert [row['status'] for row in result['approaches']]==['failed','succeeded']
    first=backend.read_run(identifier('mr',result['experiment_id'],'direct_codex_news',result['slot']))
    assert first['outcome']['rejected_forecast']==supplied and first['outcome']['execution_status']=='not_applicable'
    assert first['outcome']['provider_final_response_text']=='controlled invalid-baseline final'
    observations=[row for (table,_),row in backend.records.items() if table=='portfolio_observation']
    assert len(observations)==2 and len({row['approach_id'] for row in observations})==2
    execute()
    assert calls==['invalid','following']  # Permanent retry preserves both original attempts.
