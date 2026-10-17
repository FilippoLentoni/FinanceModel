"""Names in the exact concatenation order of rl.spec.build_observation."""
from __future__ import annotations


def observation_names(spec,instruments):
    names=[]
    for feature in spec.features:
        if feature in ('log_return_window','recent_returns'):
            count=spec.window if feature=='log_return_window' else min(5,spec.window)
            names.extend(f'{feature}.lag{lag}.{symbol}' for lag in range(count,0,-1) for symbol in instruments)
        elif feature=='current_weights':names.extend(f'current_weight.{symbol}' for symbol in (*instruments,'cash'))
        elif feature=='market_summary':names.extend(f'market_summary.{stat}.{horizon}sessions.{symbol}' for horizon in (5,20,60) for stat in ('mean','std_population') for symbol in instruments)
        elif feature=='portfolio_drawdown':names.append('portfolio_drawdown.fraction')
        elif feature=='cross_asset_risk':
            names.extend(f'cross_asset_risk.correlation.{a}.{b}' for index,a in enumerate(instruments) for b in instruments[index+1:]);names.append('cross_asset_risk.portfolio_volatility_annual')
        elif feature=='market_regime':names.extend('market_regime.'+n for n in ('market_mean_20','market_mean_60','market_std_20','market_std_60','cross_section_mean_dispersion_20','positive_cross_section_fraction_20'))
        elif feature in ('gdelt_organization_tone_v1','gdelt_intraday_organization_tone_v1'):
            if 'intraday' in feature:from finplan_model.rl.gdelt_intraday import FEATURE_ORDER
            else:from finplan_model.rl.gdelt import FEATURE_ORDER
            names.extend(f'{feature}.{symbol}.{field}' for symbol in instruments for field in FEATURE_ORDER)
        elif feature=='etf_news_metadata_v1':names.extend(f'{feature}.{symbol}.{field}' for symbol in instruments for field in ('sentiment','event_count_clipped_div100','uncertainty','age_hours_clipped_div24','missing'))
        else:raise ValueError('unknown frozen observation feature name')
    return names
