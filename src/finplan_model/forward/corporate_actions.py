"""Conservative eligibility guard; no dividend/split accounting is invented."""
from __future__ import annotations
from datetime import UTC,datetime
from zoneinfo import ZoneInfo
import copy


def _instant(value):
    parsed=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if parsed.tzinfo is None:raise ValueError('timezone required')
    return parsed.astimezone(UTC)


def unreconciled_actions(holdings,source,at,*,book_created_at=None):
    """Return original captured events relevant to current nonzero positions.

    A ledger state's as_of is a valuation/rebalance timestamp, not acquisition
    proof. Only explicit position acquisition or book creation can exclude older
    events. Unknown acquisition dates conservatively retain captured events.
    Date-only action/acquisition evidence is compared by NY calendar day; same
    day remains ambiguous and blocks. Future/unavailable evidence is not used.
    """
    cutoff=_instant(at);day=cutoff.astimezone(ZoneInfo('America/New_York')).date().isoformat()
    held={p['instrument_id']:p for p in holdings.get('positions',[]) if p.get('quantity')}
    found=[]
    def eligible(symbol,event,layer,when=None,available=None):
        if symbol not in held:return
        try:
            if available and _instant(available)>cutoff:return
        except (ValueError,TypeError):pass  # Unknown evidence timing cannot certify safety.
        acquisition=held[symbol].get('acquired_at') or held[symbol].get('acquisition_at') or book_created_at
        try:
            if isinstance(when,bool):raise ValueError('boolean is not an event timestamp')
            if not isinstance(when,(str,int,float)):raise ValueError('event timestamp is unknown')
            if isinstance(when,str) and 'T' not in when and len(when)!=10:raise ValueError('date must be exact')
            event_instant=datetime.fromtimestamp(float(when),UTC) if isinstance(when,(int,float)) else _instant(when) if 'T' in str(when) else None
            if event_instant and event_instant>cutoff:return
            event_day=event_instant.astimezone(ZoneInfo('America/New_York')).date().isoformat() if event_instant else str(when)[:10]
            if len(event_day)!=10:raise ValueError('event date missing')
            datetime.fromisoformat(event_day)
            if event_day>day:return
            if acquisition:
                acquired_day=str(acquisition)[:10] if len(str(acquisition))==10 else _instant(acquisition).astimezone(ZoneInfo('America/New_York')).date().isoformat()
                if acquired_day>day or (len(str(acquisition))>10 and _instant(acquisition)>cutoff):raise ValueError('future acquisition cannot establish current holdings')
                if event_day<acquired_day:return
        except (ValueError,TypeError,OverflowError):event_day=None
        found.append({'instrument_id':symbol,'layer':layer,'event_day':event_day,
                      'holding_since':acquisition,'holding_since_basis':'position_acquisition' if held[symbol].get('acquired_at') or held[symbol].get('acquisition_at') else 'book_creation_conservative' if book_created_at else 'unknown_conservative',
                      'original_evidence':copy.deepcopy(event),'status':'unreconciled_captured_action'})
    for row in source.get('execution_observations',[]):
        if row.get('dividend') or row.get('split_ratio') not in (None,0,1):
            eligible(row.get('instrument_id'),row,'captured_daily_action',row.get('session_date'),row.get('available_at'))
    quotes=(source.get('current_quotes') or {}).get('quotes',[])
    if isinstance(quotes,dict):quotes=[{'instrument_id':key,**value} for key,value in quotes.items()]
    for quote in quotes:
        if quote.get('instrument_id') not in held:continue
        actions=quote.get('corporate_actions')
        if not actions:continue
        if not isinstance(actions,dict):
            eligible(quote.get('instrument_id'),actions,'current_quote_unknown',available=quote.get('available_at'));continue
        for kind,events in actions.items():
            if not events:continue
            if not isinstance(events,dict):
                eligible(quote.get('instrument_id'),events,'current_quote_'+kind,available=quote.get('available_at'));continue
            for key,event in events.items():
                when=event.get('date',key) if isinstance(event,dict) else key
                if isinstance(when,str) and when.isdigit():when=int(when)
                eligible(quote.get('instrument_id'),event,'current_quote_'+kind,when,quote.get('available_at'))
    return found


def book_inception(backend,book,holdings):
    """Use immutable portfolio metadata, never the mutable paper state's as_of."""
    if not any(p.get('quantity') for p in holdings.get('positions',[])):return None
    if book.get('created_at'):return book['created_at']
    original=backend.repo.get('portfolio',book['portfolio_id'])
    return original.doc.get('created_at') if original else None
