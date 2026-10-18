"""The same models run through the deployed agent: one named MCP tool per call, checked strictly."""
from finplan_model.forward.hourly import _agent_tool_adapter
from finplan_model.forward.registry import APPROACHES

UNIVERSE = ['AAPL', 'MSFT']


def _base(model):
    return {'model': next(a for a in APPROACHES if a.approach_id == 'agent_' + model).to_dict()}


def _source(served=('AAPL', 'GOOGL')):
    return {'investable_universe': UNIVERSE, 'input_snapshot_id': 'snap_1', 'observations': [{'session_date': '2026-10-16'}],
            'served_advisory_policy': {'instruments': list(served)}}


HOLDINGS = {'weights': [{'instrument_id': i, 'weight': 0.} for i in UNIVERSE], 'cash_weight': 1., 'portfolio_value': 100000., 'high_watermark': 100000.}


def _reply(tool, weights, strategy):
    def invoke(name, arguments, prompt, before):
        before({'payload': {'tool_request': {'name': name, 'arguments': arguments}, 'prompt': prompt}})
        invoke.call = (name, arguments, prompt)
        return {'status': 'completed', 'tools_called': [tool], 'target_weights': [{'instrument_id': k, 'weight': v} for k, v in weights.items()],
                'cash_weight': 1 - sum(weights.values()), 'served_strategy': strategy}
    return invoke


def test_agent_runs_the_requested_classical_model_and_extra_zero_weights_are_dropped():
    invoke = _reply('recommend_classical_portfolio', {'AAPL': .6, 'MSFT': .4, 'NFLX': 0.}, 'cvar')
    out = _agent_tool_adapter(_base('cvar'), _source(), HOLDINGS, lambda p: None, invoke=invoke)
    assert invoke.call[0] == 'recommend_classical_portfolio' and invoke.call[1]['algorithm'] == 'cvar'
    assert invoke.call[2] == 'Give me a portfolio recommendation using model cvar.'
    assert out['target_weights'] == [{'instrument_id': 'AAPL', 'weight': .6}, {'instrument_id': 'MSFT', 'weight': .4}]
    assert out['status'] == 'completed'


def test_a_different_tool_or_model_or_universe_is_a_failure_not_a_recommendation():
    wrong_tool = _agent_tool_adapter(_base('min_variance'), _source(), HOLDINGS, lambda p: None, invoke=_reply('get_etf_research_inputs', {}, None))
    assert wrong_tool['status'] == 'failed' and wrong_tool['reason'] == 'agent_did_not_run_exactly_the_requested_tool'
    wrong_model = _agent_tool_adapter(_base('min_variance'), _source(), HOLDINGS, lambda p: None, invoke=_reply('recommend_classical_portfolio', {'AAPL': 1.}, 'mean_variance'))
    assert wrong_model['status'] == 'failed' and wrong_model['reason'] == 'agent_returned_a_different_model'
    outside = _agent_tool_adapter(_base('cvar'), _source(), HOLDINGS, lambda p: None, invoke=_reply('recommend_classical_portfolio', {'AAPL': .5, 'NFLX': .5}, 'cvar'))
    assert outside['status'] == 'failed' and outside['outside_instruments'] == ['NFLX']


def test_ppo_tool_is_not_invoked_when_it_serves_another_universe():
    captured = []
    def invoke(*a):
        raise AssertionError('must not invoke')
    out = _agent_tool_adapter(_base('ppo'), _source(), HOLDINGS, captured.append, invoke=invoke)
    assert out['status'] == 'unavailable' and out['reason'] == 'agent_tool_serves_a_different_fixed_universe'
    assert captured[0]['invocation_state'] == 'not_invoked'
    ok = _agent_tool_adapter(_base('ppo'), _source(served=UNIVERSE), HOLDINGS, lambda p: None, invoke=_reply('recommend_portfolio', {'AAPL': .2}, 'ppo'))
    assert ok['status'] == 'completed'
