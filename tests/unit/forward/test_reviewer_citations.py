"""Stubbed Codex protocol only: actual supplied context bounds citation acceptance."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from finplan_model.forward.research import ReviewFailure,review
from tests.unit.forward.test_runtime import source,holdings

NEWS='https://example.org/selected-headline'
CONSTITUENTS='https://example.org/published-constituents'
UNSUPPLIED='https://example.org/collected-but-not-supplied'


def invoke(tmp_path,monkeypatch,citation,*,allow_forecast=True,forecast_override=None):
    shared=source();shared.update(availability_cutoff='2026-07-11T00:00:00Z',scheduled_slot='2026-07-11T00:00:00Z',
        news=[{'news_id':'controlled-news','headline':'Controlled public title','url':NEWS,'provider':'controlled_public_fixture',
               'published_at':'2026-07-10T12:00:00Z','available_at':'2026-07-10T13:00:00Z'}],
        research_universe={'source_url':CONSTITUENTS,'members':[]},raw_news=[{'url':UNSUPPLIED}],research_coverage=[])
    final={'status':'hold','target_weights':[],'cash_weight':1.,'rationale':'Controlled fixture',
        'citations':[citation],'forecast':{'status':'not_available','target_at':None,'value':None,'units':None,'baseline_value':None,'assumptions':[]}}
    if forecast_override is not None:final['forecast']=forecast_override
    def codex(command,**kwargs):
        if command[-1]=='--version':return SimpleNamespace(stdout='controlled-codex-version',returncode=0)
        schema=json.loads(Path(command[command.index('--output-schema')+1]).read_text())
        if not allow_forecast:
            assert schema['properties']['forecast']['properties']['status']['enum']==['not_available']
            assert all(schema['properties']['forecast']['properties'][key]=={'type':'null'} for key in ('target_at','value','units','baseline_value'))
            assert 'interim research proposal, without an executed portfolio forecast' in kwargs['input']
        path=Path(command[command.index('--output-last-message')+1]);path.write_text(json.dumps(final))
        assert UNSUPPLIED not in kwargs['input']
        assert CONSTITUENTS in kwargs['input'] and NEWS in kwargs['input']
        return SimpleNamespace(stdout=json.dumps({'type':'turn.completed','usage':{'input_tokens':123,'output_tokens':12}}),stderr='',returncode=0)
    monkeypatch.setenv('FINPLAN_CODEX_MODEL','controlled_fixture_model')
    monkeypatch.setattr('finplan_model.forward.research.subprocess.run',codex)
    captured=[]
    return review(shared,holdings(),tmp_path/'final.json',capture=captured.append,allow_forecast=allow_forecast)


@pytest.mark.parametrize('citation',[NEWS,CONSTITUENTS])
def test_selected_news_and_actually_supplied_constituent_url_are_allowed(tmp_path,monkeypatch,citation):
    result=invoke(tmp_path,monkeypatch,citation)
    assert result['citations']==[citation]
    assert result['reviewer_invocation']['tool_attempts']==[]


@pytest.mark.parametrize('citation',[UNSUPPLIED,'https://example.org/invented'])
def test_unsupplied_raw_source_or_invented_citation_is_rejected_with_original_public_evidence(tmp_path,monkeypatch,citation):
    with pytest.raises(ReviewFailure,match='unprovided source') as caught:invoke(tmp_path,monkeypatch,citation)
    assert json.loads(caught.value.public_final_response)['citations']==[citation]
    assert caught.value.invocation['model']=='controlled_fixture_model'
    assert caught.value.invocation['input_capture_scope']=='public_prompt_only_no_tools'
    assert (tmp_path/'final.json.input.json').exists()
    assert (tmp_path/'final.json.raw-response.json').exists()


def test_interim_uses_exact_captured_restricted_schema_without_mutating_primary_schema(tmp_path,monkeypatch):
    from finplan_model.forward.research import POLICY_SCHEMA
    import copy
    original=copy.deepcopy(POLICY_SCHEMA)
    result=invoke(tmp_path,monkeypatch,NEWS,allow_forecast=False)
    captured=json.loads((tmp_path/'final.json.input.json').read_text())
    passed=json.loads((tmp_path/'final.json.schema.json').read_text())
    assert captured['schema']==passed and captured['forecast_scope']=='interim_nonexecuted_research_proposal'
    assert result['forecast']['status']=='not_available' and POLICY_SCHEMA==original
    invoke(tmp_path/'primary',monkeypatch,NEWS)
    primary=json.loads((tmp_path/'primary'/'final.json.input.json').read_text())
    assert primary['schema']==original and primary['forecast_scope']=='primary_experiment_book'


@pytest.mark.parametrize('submitted',[
    {'status':'forecast','target_at':'2026-07-12T00:00:00Z','value':101000.,'units':'USD','baseline_value':100000.,'assumptions':[]},
    {'status':'not_available','target_at':None,'value':101000.,'units':None,'baseline_value':None,'assumptions':[]},
])
def test_interim_unexpected_forecast_is_rejected_with_original_public_evidence(tmp_path,monkeypatch,submitted):
    with pytest.raises(ReviewFailure,match='forbidden executed-portfolio forecast') as caught:
        invoke(tmp_path,monkeypatch,NEWS,allow_forecast=False,forecast_override=submitted)
    assert json.loads(caught.value.public_final_response)['forecast']==submitted
    assert caught.value.invocation['input_capture_scope']=='public_prompt_only_no_tools'
    assert (tmp_path/'final.json.raw-response.json').exists()
