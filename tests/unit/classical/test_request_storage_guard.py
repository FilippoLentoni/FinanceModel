"""Direct producer boundary applies the existing semantic rule before dispatch.

The legacy frozen schema marker remains absent; these tests prove runtime
rejection, and do not waive or relabel the pre-existing producer schema failure.
"""
import copy
from types import SimpleNamespace
import pytest
from finplan_model.classical_api import handle
from finplan_model.core.errors import FinplanError


@pytest.mark.parametrize('field',['contract_version','decision_at'])
@pytest.mark.parametrize('location',['s3://example-source-object/data.json','arn:aws:s3:::example-source-object','/tmp/example-data.json','https://example.org/research'])
def test_locations_rejected_before_contract_validation_or_any_producer_call(field,location,monkeypatch):
    import finplan_model.classical_api as api
    import finplan_model.etf_news as etf
    def forbidden(*args,**kwargs):pytest.fail('no validator, producer or provider may run for storage-bearing request')
    monkeypatch.setattr(api,'require_valid',forbidden)
    monkeypatch.setattr(etf,'handle',forbidden)
    service=SimpleNamespace(env='beta',d=SimpleNamespace(platform=SimpleNamespace(get_research_news=forbidden)))
    request={'symbols':['SPY','VOO'],field:location}
    with pytest.raises(FinplanError) as error:
        handle({'environment':'beta','operation':'get_etf_research_inputs','request':request},service)
    assert error.value.code=='VALIDATION_FAILED'
    assert error.value.details['pointer']=='/'+field
    assert location not in str(error.value)


def test_valid_existing_request_is_unchanged_at_producer_dispatch(monkeypatch):
    import finplan_model.classical_api as api
    import finplan_model.etf_news as etf
    request={'symbols':['SPY','VOO'],'contract_version':'1.13.1','decision_at':'2026-01-02T14:00:00Z'}
    original=copy.deepcopy(request);calls=[];real_validator=api.require_valid
    def producer(service,body):calls.append(body);return {'controlled_producer_result':True}
    def validator(document,schema):
        if schema.endswith('-response'):return document
        return real_validator(document,schema)
    monkeypatch.setattr(etf,'handle',producer);monkeypatch.setattr(api,'require_valid',validator)
    result=handle({'environment':'beta','operation':'get_etf_research_inputs','request':request},SimpleNamespace(env='beta'))
    assert result=={'controlled_producer_result':True} and calls==[original] and request==original
