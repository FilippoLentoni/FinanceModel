"""beta-competition-v1: twice-daily New York slots, same-run fills, runner deadline, handler."""
import copy
from datetime import datetime

import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.forward import schedule
from finplan_model.forward.competition import BETA_COMPETITION_V1, competition
from finplan_model.forward.hourly import identifier, run_hourly
from test_hourly_recovery import FakeEvidence, adapters, source

POLICY = BETA_COMPETITION_V1.comparison_policy
EXPERIMENT = 'controlled-competition'
MON_0945 = '2026-07-13T13:45:00Z'   # Monday 09:45 EDT
MON_1545 = '2026-07-13T19:45:00Z'   # Monday 15:45 EDT


def parse(value):return datetime.fromisoformat(value.replace('Z','+00:00'))


# ----------------------------------------------------------------------------- schedule
@pytest.mark.parametrize('slot,expected', [
    ('2026-07-13T13:45:00Z', '2026-07-13T13:45:00Z'),          # 09:45 EDT
    ('2026-07-13T19:45:00Z', '2026-07-13T19:45:00Z'),          # 15:45 EDT
    ('2026-12-14T14:45:00Z', '2026-12-14T14:45:00Z'),          # 09:45 EST (DST moves the UTC instant)
    ('2026-12-14T09:45:00-05:00', '2026-12-14T14:45:00Z'),     # equivalent offset
])
def test_twice_daily_accepts_exactly_the_new_york_decision_times(slot, expected):
    assert schedule.validate_slot(slot, schedule.TWICE_DAILY) == expected


@pytest.mark.parametrize('slot,match', [
    ('2026-12-14T13:45:00Z', 'exactly'),        # 08:45 EST: the summer UTC time is wrong in winter
    ('2026-07-13T14:00:00Z', 'exactly'),        # 10:00 EDT
    ('2026-07-13T13:45:30Z', 'exactly'),        # seconds
    ('2026-07-11T13:45:00Z', 'weekday'),        # Saturday
    ('2026-07-13T13:45:00', 'aware'),           # naive
    ('not-a-time', 'ISO'),
])
def test_twice_daily_rejects_every_other_instant(slot, match):
    with pytest.raises(ValueError, match=match):
        schedule.validate_slot(slot, schedule.TWICE_DAILY)


def test_previous_slots_and_research_slots_follow_new_york_weekdays():
    assert schedule.previous_slot(MON_1545, schedule.TWICE_DAILY) == MON_0945
    assert schedule.previous_slot(MON_0945, schedule.TWICE_DAILY) == '2026-07-10T19:45:00Z'   # Friday 15:45
    assert schedule.previous_slot('2026-07-13T13:00:00Z') == '2026-07-13T12:00:00Z'
    assert schedule.research_slots_between('2026-07-13T12:45:00Z', MON_1545) == ['2026-07-13T19:00:00Z', '2026-07-13T13:00:00Z']
    assert schedule.validate_research_slot('2026-07-13T13:00:00Z') == '2026-07-13T13:00:00Z'
    with pytest.raises(ValueError):
        schedule.validate_research_slot(MON_0945)
    assert schedule.horizon_hour(MON_0945) == '2026-07-13T13:00:00Z'


def test_competition_declares_its_fixed_books_and_policy():
    spec = competition('beta-competition-v1')
    assert spec.approach_ids == ('cash', 'buy_and_hold', 'equal_weight', 'classical_min_variance', 'classical_cvar', 'ppo',
                                 'agent_min_variance', 'agent_cvar', 'agent_ppo', 'direct_claude_news')
    assert spec.cohort == 'eight_assets'
    assert POLICY['decision_opportunity'] == 'market_twice_daily_0945_1545_NY'
    assert POLICY['execution_timing'] == 'same_run_regular_session_quote' and POLICY['execution_price_freshness_seconds'] == 900
    assert POLICY['fees'] == {'proportional_bps': 1.0, 'fixed_per_trade': 0.0}
    with pytest.raises(ValueError, match='unknown experiment'):
        competition('beta-forward-hourly-v1')


# ----------------------------------------------------------------------------- same-run point rule
def quote(i, *, quoted='2026-07-13T13:44:00Z', available='2026-07-13T13:44:30Z', price=100., market_open=True):
    return {'instrument_id': i, 'price': price, 'volume': 5_000_000, 'status': 'fresh', 'market_open': market_open,
            'quoted_at': quoted, 'bar_started_at': quoted.replace(':44:00', ':43:00'), 'available_at': available,
            'regular_session': {'start': '2026-07-13T13:30:00Z', 'end': '2026-07-13T20:00:00Z'}}


def point(quotes, *, decision='2026-07-13T13:49:00Z', recorded='2026-07-13T13:50:00Z', opportunity=MON_0945):
    from finplan_model.evaluate.evaluator import evaluate_point_rebalance
    from finplan_model.sim.config import SimulationConfig
    return evaluate_point_rebalance({'weights': {'A': .5, 'B': .5}, 'cash': 0.}, positions={}, cash=100000., quotes=quotes,
        config=SimulationConfig.from_dict({k: POLICY[k] for k in ('fees', 'spread', 'slippage', 'liquidity')}),
        decision_at=decision, opportunity_at=opportunity, recorded_at=recorded, universe=['A', 'B'],
        max_quote_age_seconds=900, execution_timing='same_run_regular_session_quote')


def test_same_run_fills_at_the_slot_quote_collected_before_the_decision():
    result = point({'A': quote('A'), 'B': quote('B', price=50.)})
    assert result['execution_rule'] == 'same_run_regular_session_quote' and result['identity']['execution_timing'] == 'same_run_regular_session_quote'
    assert {f['instrument_id'] for f in result['fills']} == {'A', 'B'}
    assert result['reference_prices'] == {'A': 100., 'B': 50.}
    assert result['execution_at'] == '2026-07-13T13:50:00Z' and result['before_value'] == pytest.approx(result['after_value'] + result['transaction_cost'])
    assert result['transaction_cost'] > 0      # fees/spread/slippage unchanged


@pytest.mark.parametrize('quotes,kwargs,reason', [
    ({'A': quote('A'), 'B': quote('B')}, {'recorded': '2026-07-13T14:00:01Z'}, 'no longer fresh'),           # > 900 s old at the fill
    ({'A': quote('A', quoted='2026-07-13T13:20:00Z', available='2026-07-13T13:20:30Z'), 'B': quote('B')}, {}, 'not captured for this slot'),
    ({'A': quote('A', available='2026-07-13T13:51:00Z'), 'B': quote('B')}, {}, 'not captured for this slot'),  # known only after the fill
    ({'A': quote('A', market_open=False), 'B': quote('B')}, {}, 'off-session'),
    ({'A': quote('A'), 'B': quote('B')}, {'decision': '2026-07-13T13:40:00Z'}, 'scheduled slot'),            # decision before its slot
    ({'A': quote('A'), 'B': quote('B')}, {'decision': '2026-07-13T13:51:00Z'}, 'scheduled slot'),            # fill before the decision
])
def test_same_run_rejects_stale_foreign_future_and_off_session_quotes(quotes, kwargs, reason):
    with pytest.raises(FinplanError, match=reason):
        point(quotes, **kwargs)


def test_same_run_rule_is_xnys_only_and_hourly_rule_unchanged():
    from finplan_model.evaluate.evaluator import evaluate_point_rebalance
    from finplan_model.sim.config import SimulationConfig
    with pytest.raises(FinplanError, match='unsupported execution timing'):
        evaluate_point_rebalance({'weights': {'BTC-USD': 1.}, 'cash': 0.}, positions={}, cash=1., quotes={}, config=SimulationConfig.from_dict({}),
            decision_at=MON_0945, opportunity_at=MON_0945, recorded_at=MON_0945, universe=['BTC-USD'], execution_calendar='CRYPTO_24X7',
            quote_protocol='native_completed_1h_coinbase_close/1', execution_timing='same_run_regular_session_quote')
    # Without execution_timing the next-quote rule still refuses a quote older than the decision.
    from finplan_model.evaluate.evaluator import evaluate_point_rebalance as rule
    with pytest.raises(FinplanError, match='eligible window|declared opportunity'):
        rule({'weights': {'A': 1.}, 'cash': 0.}, positions={}, cash=1000., quotes={'A': quote('A')}, config=SimulationConfig.from_dict({}),
             decision_at='2026-07-13T13:49:00Z', opportunity_at='2026-07-13T13:50:00Z', recorded_at='2026-07-13T13:50:00Z', universe=['A'])


# ----------------------------------------------------------------------------- run_hourly same-run path
class Ledger(FakeEvidence):
    def __init__(self):
        super().__init__();self.commits=[]
    def commit_paper_step(self, rid, expected_revision, state, fills, execution, at):
        row = self.records['research_run', rid];book = self.books[row['portfolio_id']]
        assert not row.get('execution_ref') and row['status'] in ('succeeded', 'hold', 'reused')
        assert row['portfolio_revision'] == expected_revision == book['revision']
        assert parse(row['started_at']) <= parse(execution['execution_at']) <= parse(at)
        book['paper_state'] = copy.deepcopy(state);book['revision'] += 1
        row.update(status='applied' if fills else 'no_op', execution_ref={'checksum': 'controlled'})
        self.commits.append((rid, fills, execution))
        return {'status': 'accepted'}


def store_quotes(e, slot, quotes, *, experiment=EXPERIMENT):
    at = max(q['available_at'] for q in quotes)
    e.capture_input(experiment, '__valuation_quotes__', slot, {'quotes': quotes, 'available_at': at, 'availability_cutoff': at}, None, at, kind='valuation_quotes')


def run(e, tmp_path, slot, now, approach_ids=('cash', 'equal_weight'), **kwargs):
    return run_hourly(slot, EXPERIMENT, tmp_path, backend=e, source=source(), policies={}, research={}, now=now,
                      external_adapters=adapters(), comparison_policy=POLICY, approach_ids=list(approach_ids), **kwargs)


def test_decision_fills_at_the_same_runs_quotes_and_values_the_book(tmp_path):
    e = Ledger();store_quotes(e, MON_0945, [quote('A'), quote('B', price=50.)])
    result = run(e, tmp_path, MON_0945, '2026-07-13T13:50:00Z')
    rows = {r['approach_id']: r for r in result['approaches']}
    assert result['status'] == 'complete'
    assert rows['equal_weight']['execution']['status'] == 'applied' and rows['cash']['execution']['status'] == 'no_op'
    (rid, fills, execution), = [c for c in e.commits if c[1]]
    assert execution['execution_rule'] == 'same_run_regular_session_quote' and execution['opportunity_at'] == MON_0945
    assert execution['quote_provenance']['A']['quoted_at'] == '2026-07-13T13:44:00Z'
    book = e.books[rows['equal_weight']['portfolio_id']]
    assert book['revision'] == 2 and {p['instrument_id'] for p in book['paper_state']['positions'] if p['quantity']} == {'A', 'B'}
    valuation = e.records['portfolio_observation', identifier('mo', EXPERIMENT, 'equal_weight', MON_0945, 'valuation')]['payload']
    assert valuation['observation_kind'] == 'valid' and valuation['quote_status'] == 'fresh'
    registry = e.registered[EXPERIMENT]['comparison_policy']
    assert registry['decision_opportunity'] == schedule.TWICE_DAILY
    assert registry['first_scheduled_slot'] == '2026-07-13T13:00:00Z' and registry['scheduled_start_at'] == MON_0945 and registry['cadence_seconds'] == 3600
    shared = e.read_input(result['shared_input_id'])['payload']
    assert shared['execution_reconciliation']['mode'] == 'same_run_regular_session_quote'
    # A retry of the same slot neither re-invokes nor re-fills.
    calls, commits = list(e.calls), len(e.commits)
    again = run(e, tmp_path, MON_0945, '2026-07-13T13:52:00Z')
    assert e.calls == calls and len(e.commits) == commits
    assert {r['approach_id']: r['execution']['status'] for r in again['approaches']} == {'cash': 'already_resolved', 'equal_weight': 'already_resolved'}


def test_stale_quote_is_refused_and_the_next_decision_supersedes_it(tmp_path):
    e = Ledger();store_quotes(e, MON_0945, [quote('A'), quote('B')])
    first = run(e, tmp_path, MON_0945, '2026-07-13T14:05:00Z')     # 21 minutes after the quote
    row = next(r for r in first['approaches'] if r['approach_id'] == 'equal_weight')
    assert row['execution']['status'] == 'not_filled' and 'fresh' in row['execution']['reason'] or 'missed' in row['execution']['reason']
    attempt = e.records['portfolio_observation', identifier('mo', EXPERIMENT, 'equal_weight', MON_0945, 'execution_attempt')]['payload']
    assert attempt['status'] == 'execution_pending' and attempt['actual_fills'] == []
    old = row['run_id']
    assert e.read_run_header(old)['status'] == 'succeeded'
    # Retrying the slot keeps the original refusal; it is never retried with a later clock.
    retry = run(e, tmp_path, MON_0945, '2026-07-13T14:06:00Z')
    assert next(r for r in retry['approaches'] if r['approach_id'] == 'equal_weight')['execution']['status'] == 'attempt_already_recorded'
    store_quotes(e, MON_1545, [quote('A', quoted='2026-07-13T19:44:00Z', available='2026-07-13T19:44:20Z'),
                               quote('B', quoted='2026-07-13T19:44:00Z', available='2026-07-13T19:44:20Z')])
    second = run(e, tmp_path, MON_1545, '2026-07-13T19:49:00Z')
    row = next(r for r in second['approaches'] if r['approach_id'] == 'equal_weight')
    assert row['execution']['status'] == 'applied' and row['execution']['superseded_run_ids'] == [old]
    assert e.read_run_header(old)['status'] == 'superseded'


def test_twice_daily_experiment_rejects_an_undeclared_slot(tmp_path):
    with pytest.raises(ValueError, match='exactly'):
        run(Ledger(), tmp_path, '2026-07-13T14:00:00Z', '2026-07-13T14:05:00Z')


def test_agent_books_are_unavailable_when_the_runner_deadline_is_near(tmp_path):
    e = Ledger();store_quotes(e, MON_0945, [quote('A'), quote('B')])
    called = []
    def agent(base, source, holdings, before):
        called.append(base['approach_id']);raise AssertionError('must not start a remote call')
    result = run_hourly(MON_0945, EXPERIMENT, tmp_path, backend=e, source=source(), policies={}, research={}, now='2026-07-13T13:50:00Z',
                        external_adapters={**adapters(), 'agent_tool': agent}, comparison_policy=POLICY,
                        approach_ids=['cash', 'agent_min_variance', 'agent_cvar'], remaining_seconds=lambda: 120.)
    rows = {r['approach_id']: r for r in result['approaches']}
    assert not called
    assert rows['agent_min_variance']['status'] == 'unavailable' and rows['agent_min_variance']['reason'] == 'runner_deadline'
    captured = e.read_input(rows['agent_cvar']['input_id'])['payload']
    assert captured['invocation_state'] == 'not_invoked' and captured['numeric']['prepared']['boundary'] == 'not_invoked_runner_deadline'
    assert rows['cash']['status'] == 'succeeded'


# ----------------------------------------------------------------------------- Lambda handler
def test_handler_run_validates_the_slot_and_returns_a_small_summary(monkeypatch):
    from finplan_model.forward import runner
    seen = {}
    def fake_run(slot, experiment, output, **kwargs):
        seen.update(slot=slot, experiment=experiment, **kwargs)
        return {'status': 'complete', 'slot': slot, 'shared_input_id': 'mi_x', 'approaches': [
            {'approach_id': 'cash', 'status': 'succeeded', 'execution': {'status': 'no_op'}},
            {'approach_id': 'agent_ppo', 'status': 'unavailable', 'reason': 'runner_deadline'}]}
    class Context:
        def get_remaining_time_in_millis(self):return 500_000
    out = runner.run_slot({'action': 'run', 'experiment': 'beta-competition-v1', 'slot': '2026-10-19T09:45:00-04:00'}, Context(), backend=object(), run=fake_run)
    assert seen['slot'] == '2026-10-19T13:45:00Z' and seen['approach_ids'] == list(BETA_COMPETITION_V1.approach_ids)
    assert seen['cohort'] == 'eight_assets' and seen['comparison_policy'] == POLICY and seen['remaining_seconds']() == 500.
    assert out == {'status': 'complete', 'experiment': 'beta-competition-v1', 'slot': '2026-10-19T13:45:00Z', 'shared_input_id': 'mi_x', 'approaches': [
        {'approach_id': 'cash', 'status': 'succeeded', 'reason': None, 'execution_status': 'no_op'},
        {'approach_id': 'agent_ppo', 'status': 'unavailable', 'reason': 'runner_deadline', 'execution_status': None}]}
    with pytest.raises(runner.EventError, match='exactly'):
        runner.run_slot({'action': 'run', 'experiment': 'beta-competition-v1', 'slot': '2026-10-19T14:00:00Z'}, backend=object(), run=fake_run)


def test_handler_rejects_unknown_actions_experiments_and_other_environments(monkeypatch):
    from finplan_model.forward import runner
    with pytest.raises(runner.EventError, match='unknown action'):
        runner.handler({'action': 'train'})
    with pytest.raises(ValueError, match='unknown experiment'):
        runner.handler({'action': 'run', 'experiment': 'other', 'slot': MON_0945})
    with pytest.raises(runner.EventError, match='research slot'):
        runner.prepare_prompt({'action': 'prepare_llm_prompt', 'experiment': 'beta-competition-v1', 'research_slot': MON_0945}, backend=object())
    monkeypatch.setenv('FINPLAN_ENVIRONMENT', 'gamma')
    with pytest.raises(runner.EventError, match='beta only'):
        runner.handler({'action': 'preflight'})


def test_preflight_imports_and_reads_only():
    from finplan_model.forward import runner
    class Missing(Exception):
        response = {'Error': {'Code': 'ParameterNotFound'}}
    class Ssm:
        def get_parameter(self, Name):
            if Name.endswith('modeling-runner-client'):raise Missing()
            return {'Parameter': {'Value': 'x'}}
    class Backend:
        mirror_content = True
        class repo:names = {'research_input': 't1', 'research_run': 't2'}
        def read_registry(self, experiment):return None
    out = runner.preflight(backend_factory=Backend, ssm=Ssm())
    assert out['status'] == 'ok' and 'finplan_platform.core.modeling_evidence' in out['modules']
    assert out['agent_parameters_missing'] == ['/finplan/beta/financeagent/secret-ref/modeling-runner-client']
    assert out['experiments'] == {'beta-competition-v1': {'registered': False, 'approaches': []}}
    class NoBucket(Ssm):
        def get_parameter(self, Name):
            if Name.endswith('bucket-reports'):raise Missing()
            return {'Parameter': {'Value': 'x'}}
    assert runner.preflight(backend_factory=Backend, ssm=NoBucket())['status'] == 'failed'
