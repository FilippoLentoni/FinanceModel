"""direct_claude_news: runner prompt -> dev-box Claude response -> decision (offline fakes)."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path

import pytest

from finplan_model.core.artifacts import sha256_checksum
from finplan_model.forward import llm_review
from finplan_model.forward.hourly import identifier
from test_competition import Ledger, MON_0945, POLICY, quote, store_quotes
from test_hourly_recovery import adapters, source

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = 'controlled-competition'
RESEARCH_0900 = '2026-07-13T13:00:00Z'
SCRIPT_SPEC = importlib.util.spec_from_file_location('claude_review', ROOT / 'ops' / 'llm-autoschedule' / 'claude_review.py')
script = importlib.util.module_from_spec(SCRIPT_SPEC);SCRIPT_SPEC.loader.exec_module(script)


def research(e, slot=RESEARCH_0900, **changes):
    payload = {'news': [{'news_id': 'n1', 'headline': 'Controlled synthetic headline', 'url': 'https://example.invalid/n1', 'published_at': '2026-07-13T12:30:00Z',
                         'available_at': '2026-07-13T12:40:00Z', 'symbols': ['A']}],
               'universe': {'source_url': 'https://example.invalid/constituents', 'members': []}, 'coverage': [], 'workers': [],
               'available_at': '2026-07-13T12:45:00Z', 'availability_cutoff': '2026-07-13T12:45:00Z', **changes}
    e.capture_input(EXPERIMENT + '-interim', '__research_source__', slot, payload, None, '2026-07-13T12:45:00Z', kind='research_source')


def prepare(e, slot=RESEARCH_0900, now='2026-07-13T13:02:00Z'):
    return llm_review.prepare_llm_prompt(e, EXPERIMENT, slot, comparison_policy=POLICY, load_inputs=lambda cohort=None: (source(), {}), now=now)


class Dynamo:
    """The research_input table as the dev-box script sees it (index rows, records, content items)."""
    def __init__(self, e):
        self.items = {}
        for (kind, rid), doc in e.records.items():
            if kind != 'research_input':continue
            raw = e.artifacts[rid];key = f'modeling-evidence/v1/inputs/{rid}/{doc["checksum"].split(":")[1]}.json'
            record = {**doc, 'artifact_key': key}
            self.items[rid] = {'pk': {'S': rid}, 'kind': {'S': doc['kind']}, 'doc': {'S': json.dumps(record)},
                               'approach_key': {'S': f'{doc["experiment_id"]}#{doc["approach_id"]}'}, 'time_key': {'S': doc['slot'].replace('Z', '+00:00') + '#' + rid}}
            self.items[f'content#{key}#0000'] = {'pk': {'S': f'content#{key}#0000'}, 'parts': {'N': '1'}, 'data': {'B': gzip.compress(raw, mtime=0)}}
    def query(self, **kw):
        rows = [i for i in self.items.values() if i.get('approach_key', {}).get('S') == kw['ExpressionAttributeValues'][':k']['S']]
        return {'Items': sorted(rows, key=lambda i: i['time_key']['S'], reverse=True)[:kw['Limit']]}
    def get_item(self, TableName, Key, ConsistentRead):
        item = self.items.get(Key['pk']['S']);return {'Item': copy.deepcopy(item)} if item else {}
    def put_item(self, TableName, Item, ConditionExpression):
        assert ConditionExpression == 'attribute_not_exists(pk)' and Item['pk']['S'] not in self.items
        self.items[Item['pk']['S']] = copy.deepcopy(Item)


def claude_answer(text, *, exit_code=0, responded='2026-07-13T13:20:00Z'):
    def run(prompt):
        assert prompt.startswith('You are the direct Codex financial research reviewer')
        return {'claude_version': '2.1.302 (Claude Code)', 'requested_at': '2026-07-13T13:10:00Z', 'responded_at': responded, 'exit_code': exit_code,
                'stdout': json.dumps({'type': 'result', 'subtype': 'success', 'is_error': exit_code != 0, 'num_turns': 1, 'result': text, 'total_cost_usd': 0.02,
                                      'modelUsage': {'claude-opus-5-5': {'inputTokens': 3, 'outputTokens': 9, 'costUSD': 0.02}}})}
    return run


GOOD = json.dumps({'status': 'recommended', 'target_weights': [{'instrument_id': 'A', 'weight': 0.6}, {'instrument_id': 'B', 'weight': 0.3}],
                   'cash_weight': 0.1, 'rationale': 'Controlled rationale', 'citations': ['https://example.invalid/n1'],
                   'forecast': {'status': 'not_available', 'target_at': None, 'value': None, 'units': None, 'baseline_value': None, 'assumptions': []}})


def answered(e, text=GOOD, **kwargs):
    research(e);out = prepare(e);table = Dynamo(e)
    result = script.review_once(table, table='t', experiment=EXPERIMENT, now=script.parse_time('2026-07-13T13:10:00Z'), run=claude_answer(text, **kwargs))
    return out, table, result


def test_prompt_is_the_exact_review_prompt_persisted_once_with_a_deterministic_id():
    e = Ledger();research(e);out = prepare(e)
    pid = llm_review.prompt_id(EXPERIMENT, RESEARCH_0900)
    assert out['status'] == 'prepared' and out['prompt_id'] == pid == identifier('mi', EXPERIMENT, 'direct_claude_news', RESEARCH_0900, 'llm_prompt')
    doc = e.records['research_input', pid];payload = e.read_input(pid)['payload']
    assert doc['kind'] == 'llm_prompt' and doc['approach_id'] == 'direct_claude_news' and doc['slot'] == RESEARCH_0900
    from finplan_model.forward.research import prepare_review
    assert payload['status'] == 'pending' and payload['prompt'].startswith('You are the direct Codex financial research reviewer')
    assert 'RESPONSE JSON SCHEMA' in payload['prompt'] and payload['prompt_checksum'] == sha256_checksum(payload['prompt'].encode())
    assert payload['review_request']['actual_source']['news'][0]['news_id'] == 'n1'
    assert payload['holdings']['portfolio_id'] == e.ensure_portfolio(EXPERIMENT, 'direct_claude_news', 100000., ['A', 'B'], RESEARCH_0900)['portfolio_id']
    assert payload['tool_policy']['tools'] == 'none' and payload['response_contract']['key'] == llm_review.response_key(pid)
    assert prepare(e, now='2026-07-13T13:05:00Z')['status'] == 'exists'


def test_prompt_needs_a_successful_research_source():
    assert prepare(Ledger())['status'] == 'unavailable'
    e = Ledger();research(e, status='failed', acquisition_failure='controlled outage')
    out = prepare(e);assert out['status'] == 'unavailable' and 'controlled outage' in out['reason']
    e = Ledger();research(e, slot='2026-07-10T19:00:00Z')      # Friday 15:00 is the newest research at Monday 09:00
    assert prepare(e)['research_source_slot'] == '2026-07-10T19:00:00Z'


def test_dev_box_script_answers_once_and_the_runner_decodes_its_item():
    e = Ledger();out, table, result = answered(e)
    assert result['status'] == 'completed' and result['prompt_id'] == out['prompt_id']
    item = table.items[llm_review.response_key(out['prompt_id'])]
    assert script.response_id(out['prompt_id']) == llm_review.response_id(out['prompt_id'])
    body = llm_review.decode_response_item(item)
    assert body['parsed']['cash_weight'] == 0.1 and body['claude_version'].startswith('2.1') and body['command'][:3] == ['claude', '-p', '--output-format']
    assert '--tools' in body['command'] and body['command'][body['command'].index('--tools') + 1] == ''
    assert len(json.dumps(item)) < 350_000
    again = script.review_once(table, table='t', experiment=EXPERIMENT, now=script.parse_time('2026-07-13T13:15:00Z'), run=claude_answer(GOOD))
    assert again == {'status': 'idle', 'reason': 'already answered', 'prompt_id': out['prompt_id']}
    stale = script.review_once(table, table='t', experiment=EXPERIMENT, now=script.parse_time('2026-07-13T23:00:00Z'), run=claude_answer(GOOD))
    assert stale['reason'] == 'newest prompt is stale'


def decide(e, table, *, now='2026-07-13T13:50:00Z', tmp_path):
    from finplan_model.forward.hourly import run_hourly
    store_quotes(e, MON_0945, [quote('A'), quote('B')])
    reader = lambda key: copy.deepcopy(table.items.get(key))  # noqa: E731
    result = run_hourly(MON_0945, EXPERIMENT, tmp_path, backend=e, source=source(), policies={}, research={}, now=now, external_adapters=adapters(),
                        comparison_policy=POLICY, approach_ids=['cash', 'direct_claude_news'], llm_response_reader=reader)
    return next(r for r in result['approaches'] if r['approach_id'] == 'direct_claude_news')


def test_valid_response_is_the_decision_with_exact_prompt_and_response_references(tmp_path):
    e = Ledger();out, table, _ = answered(e)
    row = decide(e, table, tmp_path=tmp_path)
    assert row['status'] == 'succeeded' and row['execution']['status'] == 'applied'
    run = e.read_run(row['run_id'])['outcome']
    assert run['target_weights'] == [{'instrument_id': 'A', 'weight': 0.6}, {'instrument_id': 'B', 'weight': 0.3}] and run['cash_weight'] == pytest.approx(0.1)
    assert run['reviewer_invocation']['prompt_id'] == out['prompt_id'] and run['provider_final_response_text'] == GOOD
    prepared = e.read_input(row['input_id'])['payload']
    refs = prepared['numeric']['prepared']
    assert refs['prompt_ref']['input_id'] == out['prompt_id'] and refs['prompt_ref']['checksum'] == out['checksum']
    assert refs['response_ref']['pk'] == llm_review.response_key(out['prompt_id'])
    assert refs['messages'][0]['content'] == e.read_input(out['prompt_id'])['payload']['prompt']
    assert prepared['news']['consumed'] is True and prepared['news']['items'][0]['news_id'] == 'n1'


@pytest.mark.parametrize('text,kwargs,reason', [
    (GOOD, {'responded': '2026-07-13T13:55:00Z'}, 'after the decision cutoff'),
    (GOOD.replace('"B"', '"TSLA"'), {}, 'outside the experiment universe'),
    (GOOD.replace('0.6', '1.6'), {}, '[0, 1]'),
    (GOOD.replace('"cash_weight": 0.1', '"cash_weight": 0.5'), {}, 'cash_weight'),
    (GOOD.replace('https://example.invalid/n1', 'https://elsewhere.invalid'), {}, 'not supplied'),
    ('Here is my view: buy A', {}, 'invalid_json'),
    (GOOD, {'exit_code': 1}, 'cli_failed'),
])
def test_stale_out_of_universe_and_malformed_responses_make_the_book_unavailable(tmp_path, text, kwargs, reason):
    e = Ledger();_, table, _ = answered(e, text, **kwargs)
    row = decide(e, table, tmp_path=tmp_path)
    assert row['status'] == 'unavailable' and reason in row['reason']
    assert e.read_input(row['input_id'])['payload']['invocation_state'] == 'not_invoked'


def test_no_response_in_the_window_and_tampered_items_are_unavailable(tmp_path):
    e = Ledger();out, table, _ = answered(e)
    row = decide(e, table, now='2026-07-13T13:50:00Z', tmp_path=tmp_path / 'a')
    assert row['status'] == 'succeeded'
    # Seven hours later the 09:00 response is outside the window (15:45 decision without a 15:00 prompt).
    read = lambda k: table.items.get(k)  # noqa: E731
    assert llm_review.select_response(e, EXPERIMENT, '2026-07-13T20:00:00Z', ['A', 'B'], read_item=read)['status'] == 'selected'
    assert llm_review.select_response(e, EXPERIMENT, '2026-07-13T20:00:01Z', ['A', 'B'], read_item=read)['status'] == 'unavailable'
    late = llm_review.select_response(e, EXPERIMENT, '2026-07-14T13:50:00Z', ['A', 'B'], read_item=lambda k: table.items.get(k))
    assert late['status'] == 'unavailable' and all(r['research_slot'] > RESEARCH_0900 for r in late['searched'])
    item = copy.deepcopy(table.items[llm_review.response_key(out['prompt_id'])])
    item['body']['S'] = item['body']['S'].replace('0.6', '0.7')
    with pytest.raises(ValueError, match='checksum'):
        llm_review.decode_response_item(item)


@pytest.mark.parametrize('parsed', [None, {'target_weights': 'all A'}, {'target_weights': [{'instrument_id': 'A', 'weight': True}], 'rationale': 'x'},
                                    {'target_weights': [{'instrument_id': 'A', 'weight': .5}, {'instrument_id': 'A', 'weight': .1}], 'rationale': 'x'},
                                    {'target_weights': [{'instrument_id': 'A', 'weight': .6}, {'instrument_id': 'B', 'weight': .6}], 'rationale': 'x'},
                                    {'target_weights': [], 'rationale': ''}])
def test_allocation_validation_is_strict(parsed):
    with pytest.raises(ValueError):
        llm_review.validate_allocation(parsed, ['A', 'B'])
    ok = llm_review.validate_allocation({'target_weights': [{'instrument_id': 'A', 'weight': .25}], 'rationale': 'x'}, ['A', 'B'])
    assert ok['cash_weight'] == .75 and ok['target_weights'][1] == {'instrument_id': 'B', 'weight': 0.0}
