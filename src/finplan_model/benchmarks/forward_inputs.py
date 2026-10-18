"""Pure native current-book preparation shared by operator preflight and offline GPU worker.

No AWS SDK or provider client is imported in this numerical boundary.
"""
from __future__ import annotations
import math
from datetime import datetime
import numpy as np
from finplan_model.core.errors import FinplanError
from finplan_model.sim.market import HoldingsView,PointInTimeView
from finplan_model.forward.runtime import input_market

def _view(source,holdings,cutoff):
    universe=list(source['investable_universe']);market=input_market(source,cutoff)
    view=PointInTimeView(market,max(market.sessions),datetime.fromisoformat(cutoff.replace('Z','+00:00')),tuple(universe))
    if len(set(universe))!=len(universe) or not universe:raise ValueError('distinct supported universe required')
    positions=holdings.get('positions',[]);ids=[p['instrument_id'] for p in positions]
    if len(ids)!=len(set(ids)) or set(ids)-set(universe):raise ValueError('holdings have unsupported/duplicate instrument')
    shares={p['instrument_id']:p['quantity'] for p in positions};cash=holdings.get('cash',holdings.get('cash_balance',0.))
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in [cash,*shares.values()]):raise ValueError('holdings require finite nonnegative cash/quantities')
    prices={i:view.latest(i) for i in universe}
    if any(v is None for v in prices.values()):raise ValueError('eligible native prices missing')
    value=cash+sum(shares.get(i,0.)*prices[i] for i in universe)
    if value<=0:raise ValueError('positive native holdings value required')
    hv=HoldingsView(shares,float(cash),prices,value)
    dates,returns=view.returns(60)
    if len(returns)<20:raise FinplanError.precondition('native preparation requires20 completed returns',reason='insufficient_history')
    private={'layer':'local_preparation_only; NOT_CONSUMED_BY_MODEL','supplied_to_model':False,
        'transformation_version':'benchmarks.jev.encode_state/1','definitions':{
        'recent_trend':'product(1+last20simple_returns)-1; edges[-.05,-.01,.01,.05]',
        'volatility':'population_std(last20simple_returns)*sqrt252; edges[.15,.30,.60]',
        'current_exposure':'native marked position value/equity; edges[.01,.10,.30,.60]'},
        'derived_statistics':{i:{'momentum20':float(np.prod(1.+returns[-20:,j])-1.),'annualized_population_volatility20':float(np.std(returns[-20:,j])*np.sqrt(252.)),'exposure':hv.weights.get(i,0.)} for j,i in enumerate(universe)},
        'price_basis':source.get('price_basis','supplied_native_close'),'return_dtype':str(returns.dtype),'return_units':'fractional simple return',
        'return_order':universe,'return_session_dates':[d.isoformat() for d in dates],'simple_return_values':returns.tolist(),
        'native_mark_prices':prices,'native_weights':hv.weights,'native_cash_weight':hv.cash_weight,
        'holdings_revision':holdings.get('revision'),'native_session':view.decision_session.isoformat()}
    raw=[v for i in universe for b in market.bars_of(i) for v in (b.open,b.high,b.low,b.close,b.volume) if v is not None]
    return market,view,hv,private,raw

