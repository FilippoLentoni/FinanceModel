"""The same models run through the deployed agent: one named MCP tool per call, checked strictly."""
from finplan_model.forward.hourly import _agent_tool_adapter
from finplan_model.forward.registry import AGENT_TOOLS, APPROACHES

UNIVERSE = ['AAPL', 'MSFT']
PPO_CHECKSUM = 'sha256:' + 'a' * 64


def _base(model):
    return {'model': next(a for a in APPROACHES if a.approach_id == 'agent_' + model).to_dict(), 'decision_time': '2026-10-16T22:00:00+00:00'}


def _source(served=('AAPL', 'GOOGL')):
    # The hosted advisory policy (another universe) no longer gates the agent PPO tool.
    rows = [{'instrument_id': i, 'session_date': '2026-10-16', 'close': px, 'available_at': '2026-10-16T21:00:00+00:00'} for i, px in (('AAPL', 100.), ('MSFT', 200.))]
    return {'investable_universe': UNIVERSE, 'input_snapshot_id': 'snap_1', 'observations': rows,
            'served_advisory_policy': {'instruments': list(served)},
            'frozen_policy_refs': {'ppo': {'artifact': {'checksum': PPO_CHECKSUM}}}}


# Runner holdings (raw-close marks); positions/cash are what the direct classical book re-marks.
HOLDINGS = {'weights': [{'instrument_id': 'AAPL', 'weight': .0103}, {'instrument_id': 'MSFT', 'weight': 0.}], 'cash_weight': .9897,
            'portfolio_value': 100030., 'high_watermark': 100500., 'positions': [{'instrument_id': 'AAPL', 'quantity': 10.}], 'cash': 99000.}


def _reply(tool, weights, strategy, checksum=None):
    def invoke(name, arguments, prompt, before):
        before({'payload': {'tool_request': {'name': name, 'arguments': arguments}, 'prompt': prompt}})
        invoke.call = (name, arguments, prompt)
        out = {'status': 'completed', 'tools_called': [tool], 'target_weights': [{'instrument_id': k, 'weight': v} for k, v in weights.items()],
               'cash_weight': 1 - sum(weights.values()), 'served_strategy': strategy}
        if checksum:
            out['served_policy_artifact_checksum'] = checksum
        return out
    return invoke


def test_tool_names_fix_the_model():
    assert AGENT_TOOLS == {'min_variance': 'generate_portfolio_recommendation_min_variance',
                           'cvar': 'generate_portfolio_recommendation_cvar',
                           'ppo': 'generate_portfolio_recommendation_ppo'}


def test_agent_runs_the_requested_classical_model_on_the_cohort_universe():
    invoke = _reply('generate_portfolio_recommendation_cvar', {'AAPL': .6, 'MSFT': .4, 'NFLX': 0.}, 'cvar')
    out = _agent_tool_adapter(_base('cvar'), _source(), HOLDINGS, lambda p: None, invoke=invoke)
    name, arguments, prompt = invoke.call
    assert name == 'generate_portfolio_recommendation_cvar' and 'algorithm' not in arguments
    assert arguments['instruments'] == UNIVERSE
    # Feature-close marks, as the direct classical book's HoldingsView: 10 x 100 + 99000 cash.
    assert arguments['holdings'] == {'weights': [{'instrument_id': 'AAPL', 'weight': .01}, {'instrument_id': 'MSFT', 'weight': 0.}],
                                     'cash_weight': .99, 'portfolio_value': 100000., 'high_watermark': 100500.}
    assert arguments['input_snapshot_id'] == 'snap_1' and arguments['as_of'] == '2026-10-16'
    assert prompt == 'Give me a portfolio recommendation for AAPL, MSFT using model cvar.'
    assert out['target_weights'] == [{'instrument_id': 'AAPL', 'weight': .6}, {'instrument_id': 'MSFT', 'weight': .4}]
    assert out['status'] == 'completed'


def test_a_different_tool_or_model_or_universe_is_a_failure_not_a_recommendation():
    wrong_tool = _agent_tool_adapter(_base('min_variance'), _source(), HOLDINGS, lambda p: None, invoke=_reply('generate_portfolio_recommendation_cvar', {'AAPL': 1.}, 'cvar'))
    assert wrong_tool['status'] == 'failed' and wrong_tool['reason'] == 'agent_did_not_run_exactly_the_requested_tool'
    wrong_model = _agent_tool_adapter(_base('min_variance'), _source(), HOLDINGS, lambda p: None, invoke=_reply('generate_portfolio_recommendation_min_variance', {'AAPL': 1.}, 'mean_variance'))
    assert wrong_model['status'] == 'failed' and wrong_model['reason'] == 'agent_returned_a_different_model'
    outside = _agent_tool_adapter(_base('cvar'), _source(), HOLDINGS, lambda p: None, invoke=_reply('generate_portfolio_recommendation_cvar', {'AAPL': .5, 'NFLX': .5}, 'cvar'))
    assert outside['status'] == 'failed' and outside['outside_instruments'] == ['NFLX']


def test_ppo_tool_selects_by_universe_and_must_serve_the_direct_books_frozen_policy():
    invoke = _reply('generate_portfolio_recommendation_ppo', {'AAPL': .2}, 'ppo', PPO_CHECKSUM)
    ok = _agent_tool_adapter(_base('ppo'), _source(), HOLDINGS, lambda p: None, invoke=invoke)
    assert ok['status'] == 'completed' and invoke.call[1]['instruments'] == UNIVERSE
    assert invoke.call[1]['holdings'] == {k: HOLDINGS[k] for k in ('weights', 'cash_weight', 'portfolio_value', 'high_watermark')}  # the direct ppo actor consumes these runner holdings
    other = _agent_tool_adapter(_base('ppo'), _source(), HOLDINGS, lambda p: None,
                                invoke=_reply('generate_portfolio_recommendation_ppo', {'AAPL': .2}, 'ppo', 'sha256:' + 'b' * 64))
    assert other['status'] == 'failed' and other['reason'] == 'agent_tool_served_a_different_frozen_policy'
    assert other['expected_policy_checksum'] == PPO_CHECKSUM
    sac = _agent_tool_adapter(_base('ppo'), _source(), HOLDINGS, lambda p: None,
                              invoke=_reply('generate_portfolio_recommendation_ppo', {'AAPL': .2}, 'sac', PPO_CHECKSUM))
    assert sac['status'] == 'failed' and sac['reason'] == 'agent_returned_a_different_model'


def test_tool_errors_pass_through_unchanged():
    def invoke(name, arguments, prompt, before):
        return {'status': 'failed', 'error_code': 'HOSTED_HTTP_503'}
    assert _agent_tool_adapter(_base('ppo'), _source(), HOLDINGS, lambda p: None, invoke=invoke) == {'status': 'failed', 'error_code': 'HOSTED_HTTP_503'}
