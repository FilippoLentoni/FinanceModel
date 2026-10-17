"""Operator helper for a separately identified USD100 saved paper study. No live funds."""
from __future__ import annotations
from datetime import date
from .core.artifacts import canonical_json_bytes,sha256_checksum
from .core.errors import FinplanError

DEFAULT_INSTRUMENTS=('VOO','GOOGL')
DAILY_INSTRUMENTS=('GOOGL','AMZN','NVDA','MSFT','AAPL','META','TSLA','VOO')


def profile_universe(instruments):
    names=list(instruments)
    if len(names)!=len(set(names)) or not any(set(names)==set(allowed) for allowed in (DEFAULT_INSTRUMENTS,DAILY_INSTRUMENTS)):
        raise ValueError('paper profile requires an explicitly supported research universe')
    return names


def identity(caller_id,profile_id):
    if not isinstance(caller_id,str) or not caller_id or not isinstance(profile_id,str) or not profile_id or len(caller_id)>256 or len(profile_id)>100:raise ValueError('explicit bounded caller and profile identity required')
    return sha256_checksum(canonical_json_bytes({'caller_id':caller_id,'profile_id':profile_id,'version':'saved-paper-100/1'}))


def initialize(client,*,caller_id,profile_id,as_of,receipt=None,persist=lambda _:None,protected_portfolio_ids=(),instruments=None,news_required=None):
    """client(method,path,body=None) must use authentic operator authorization.

    Persist intent BEFORE calling create, then the minted portfolio ID BEFORE state
    initialization. Retries reuse the exact request/date, and never update an existing
    unrelated book. Local identity is provenance, not a replacement for API auth.
    """
    date.fromisoformat(as_of)
    names=profile_universe(DEFAULT_INSTRUMENTS if instruments is None else instruments)
    if news_required is not None and not isinstance(news_required,bool):raise ValueError('news_required must be explicit boolean')
    fingerprint=identity(caller_id,profile_id);key=fingerprint.split(':')[1]
    receipt=dict(receipt or {})
    intent={'identity':fingerprint,'caller_id':caller_id,'profile_id':profile_id,'as_of':as_of,'initial_cash':100.,'instruments':names}
    if news_required is not None:intent['news_required']=news_required
    if receipt and any(receipt.get(k)!=v for k,v in intent.items()):raise ValueError('paper profile receipt identity or initialization conflicts')
    receipt.update(intent);persist(dict(receipt))
    protected={pid:sha256_checksum(canonical_json_bytes(client('GET',f'/v1/portfolios/{pid}/state'))) for pid in protected_portfolio_ids}
    profile={k:receipt[k] for k in ('identity','caller_id','profile_id','initial_cash','instruments')}
    if news_required is not None:profile['news_required']=news_required
    created=client('POST','/v1/portfolios',{'name':f'Paper USD100 {profile_id}'[:200],'base_currency':'USD','synthetic':True,
        'settings':{'paper_research_profile':profile},'idempotency_key':'paper-profile-create-'+key})
    pid=created['portfolio_id']
    if pid in protected or (receipt.get('portfolio_id') and receipt['portfolio_id']!=pid):raise ValueError('paper profile creation returned a conflicting existing portfolio')
    if not created.get('synthetic') or created.get('settings',{}).get('paper_research_profile')!=profile:raise ValueError('created portfolio metadata is not the isolated paper profile')
    receipt['portfolio_id']=pid;persist(dict(receipt))
    state={'positions':[{'instrument_id':s,'quantity':0.} for s in names],'cash_balance':100.,'high_watermark':100.,'as_of':as_of,
           'base_currency':'USD','mode':'paper','source':'paper_initialization'}
    initialized=client('PUT',f'/v1/portfolios/{pid}/state',{'paper_state':state,'expected_revision':0,'idempotency_key':'paper-profile-state-'+key})
    if initialized['portfolio_id']!=pid or initialized['paper_state']!=state or initialized['revision']!=1:raise ValueError('paper profile initialization did not preserve the expected initial state')
    after={old:sha256_checksum(canonical_json_bytes(client('GET',f'/v1/portfolios/{old}/state'))) for old in protected}
    if protected!=after:raise ValueError('protected portfolio state changed during initialization; inspect concurrent updates')
    receipt.update({'initialized':True,'state_checksum':sha256_checksum(canonical_json_bytes(initialized)),'protected_state_checksums':after})
    persist(dict(receipt));return receipt


def approved_profile_instruments(portfolio,*,portfolio_id,available_instruments):
    """Restrict only an explicit saved research profile, never infer from zero positions."""
    if portfolio.get('portfolio_id')!=portfolio_id:raise FinplanError.precondition('portfolio metadata identity mismatch',reason='paper_profile_identity')
    profile=portfolio.get('settings',{}).get('paper_research_profile')
    if profile is None:return sorted(available_instruments)
    expected=identity(profile.get('caller_id'),profile.get('profile_id'))
    try:names=profile_universe(profile.get('instruments',[]))
    except (TypeError,ValueError):raise FinplanError.precondition('invalid isolated paper profile universe',reason='paper_profile_identity') from None
    if not portfolio.get('synthetic') or profile.get('identity')!=expected or profile.get('initial_cash')!=100 or ('news_required' in profile and not isinstance(profile['news_required'],bool)):
        raise FinplanError.precondition('invalid isolated paper profile metadata',reason='paper_profile_identity')
    if not set(names)<=set(available_instruments):raise FinplanError.precondition('approved snapshot lacks the paper profile instruments',reason='paper_profile_price_coverage')
    return sorted(names)
