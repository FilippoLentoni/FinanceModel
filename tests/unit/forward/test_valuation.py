"""Actual quote time, not a supplied age label, determines valuation eligibility."""
import copy
from types import SimpleNamespace

import pytest

from finplan_model.forward.hourly import POLICY, record_slot_observation
from finplan_model.forward.crypto_hourly import CRYPTO_POLICY

AT='2026-10-19T14:05:00Z'


def observed(quote=None,*,cash_only=False,source_updates=None,comparison_policy=None,at=AT):
    rows=[]
    backend=SimpleNamespace(repo=SimpleNamespace(get=lambda *args:None),record_observation=lambda *args,**kw:rows.append(args[4]))
    holdings={'positions':[] if cash_only else [{'instrument_id':'A','quantity':10.}],
        'cash':1000.,'portfolio_value':1000. if cash_only else 2000.,'revision':3,'as_of':'2026-10-16'}
    source={'execution_observations':[{'instrument_id':'A','session_date':'2026-10-16','close':100.}],
        'current_quotes':{'quotes':[] if quote is None else [copy.deepcopy(quote)]},**(source_updates or {})}
    record_slot_observation(backend,'controlled_valuation','model','2026-10-19T14:00:00Z','run','input',
        {'portfolio_id':'paper_controlled','revision':3},holdings,source,at,{'status':'succeeded'},comparison_policy)
    return rows[0]


def quote(**changes):
    return {'instrument_id':'A','price':110.,'status':'fresh','quoted_at':'2026-10-19T14:04:00Z',
        'available_at':'2026-10-19T14:04:10Z','freshness_seconds':60.,**changes}


def test_forged_freshness_zero_cannot_make_an_old_price_rankable():
    row=observed(quote(quoted_at='2026-10-19T12:34:59Z',available_at='2026-10-19T12:35:00Z',freshness_seconds=0))
    assert row['observation_kind']!='valid' and row['quote_status']!='fresh'
    assert row['value']!=2100.


@pytest.mark.parametrize('changes',[
    {'quoted_at':'2026-10-19T14:06:00Z','available_at':'2026-10-19T14:06:01Z','freshness_seconds':0},
    {'available_at':'2026-10-19T14:05:01Z','freshness_seconds':0},
    {'quoted_at':'2026-10-19T10:06:00-04:00','available_at':'2026-10-19T10:06:01-04:00','freshness_seconds':0},
    {'available_at':'2026-10-19T14:03:00Z'},
    {'quoted_at':'2026-10-19T14:04:00'},
    {'price':float('nan')}, {'price':True}, {'price':0.},
])
def test_future_naive_invalid_or_unavailable_quotes_cannot_be_fresh(changes):
    row=observed(quote(**changes))
    assert row['observation_kind']!='valid' and row['quote_status']!='fresh'


def test_equivalent_offsets_are_eligible_by_actual_instants():
    row=observed(quote(quoted_at='2026-10-19T10:04:00-04:00',available_at='2026-10-19T10:04:10-04:00',freshness_seconds=999999))
    assert row['observation_kind']=='valid' and row['quote_status']=='fresh'
    assert row['value']==2100.
    assert row['quote_provenance']['A']['quoted_at']=='2026-10-19T10:04:00-04:00'


def test_cash_only_exact_value_needs_no_market_quote_but_missing_historical_slot_stays_missing():
    actual=observed(cash_only=True)
    assert actual['value']==1000. and actual['observation_kind']=='valid'
    assert actual['quote_provenance']['basis']=='authoritative_paper_cash_balance'
    missing=observed(cash_only=True,source_updates={'historical_slot_missing':True})
    assert missing['observation_kind']=='pending' and missing['quote_status']!='fresh'


def test_crypto_uses_declared_hourly_age_and_explicit_continuous_session_evidence():
    q=quote(quoted_at='2026-10-17T12:00:00Z',available_at='2026-10-17T12:05:00Z',freshness_seconds=0)
    source={'native_crypto_dataset':{'dataset_checksum':'sha256:'+'a'*64,'as_of':'2026-10-17T12:00:00Z'}}
    row=observed(q,at='2026-10-17T12:05:00Z',source_updates=source,comparison_policy=CRYPTO_POLICY)
    assert row['value']==2100. and row['observation_kind']=='valid'
    assert row['market_session']['status']=='open'
    assert row['market_session']['source']['calendar']=='CRYPTO_24X7'
    assert row['market_session']['source']['dataset_checksum']==source['native_crypto_dataset']['dataset_checksum']
    stale=observed(q,at='2026-10-17T12:20:01Z',source_updates=source,comparison_policy=CRYPTO_POLICY)
    assert stale['observation_kind']!='valid' and stale['quote_status']!='fresh'
