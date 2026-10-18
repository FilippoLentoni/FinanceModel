"""Implemented decision adapters, never family-level aggregation of synthetic books."""
from __future__ import annotations
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Approach:
    approach_id: str
    display_name: str
    family: str
    adapter: str
    native_cadence: str = "completed_daily"
    news_layer: str = "not_consumed"
    requires: str | None = None
    version: str = "1"
    strategy_name: str | None = None
    parameters: dict | None = None
    input_capabilities: dict | None = None

    def to_dict(self):
        row=asdict(self)
        row['input_capabilities']=self.input_capabilities or {
            'news':self.news_layer,'prices':'native_hourly_BTC_ETH_candles' if self.adapter=='crypto_native' else 'bucketed_text_market_descriptors' if self.adapter=='codex' else 'captured_remote_tool_boundaries' if self.adapter in ('langgraph','agent_tool') else 'not_consumed' if self.approach_id=='cash' else 'eligibility_only' if self.approach_id in ('equal_weight','buy_and_hold') else 'completed_daily_history',
            'holdings':'not_consumed' if self.approach_id=='cash' else 'predecision_book',
            'numeric':'bucketed_market_descriptors_and_fixed_role_public_messages' if self.adapter in ('jev','qwen') else 'frozen_named_observation' if self.adapter=='actor' else 'solver_mean_covariance_or_scenarios' if self.family=='optimization' else 'exact_prompt_and_context' if self.adapter in ('codex','langgraph','agent_tool') else 'control_rule',
            'current_quotes':'valuation_only', 'forecast':'not_available_unless_explicitly_supplied'}
        if self.adapter=='crypto_native':row['native_investable_universe']=['BTC-USD','ETH-USD']
        return row


APPROACHES = (
    *(Approach(n, n.replace('_', ' ').title(), "control" if n in ('cash','buy_and_hold','equal_weight') else "optimization", "baseline")
      for n in ('cash','buy_and_hold','equal_weight','min_variance','mean_variance','scenario_cvar')),
    *(Approach('classical_'+n, 'MCP classical '+n.replace('_',' '), 'optimization', 'classical')
      for n in ('min_variance','mean_variance','cvar')),
    *(Approach(algorithm+suffix, algorithm.upper()+' '+label, 'reinforcement_learning', 'actor',
              news_layer=layer, requires='verified_frozen_policy_bundle')
      for algorithm in ('ppo','sac')
      for suffix,label,layer in (('', 'price policy','not_consumed'),('_etf_news','ETF search metadata policy','etf_news_metadata_v1'),
          ('_gdelt','GDELT organization tone policy','gdelt_organization_tone_v1'),
          ('_gdelt_intraday','GDELT intraday organization tone policy','gdelt_intraday_organization_tone_v1'))),
    Approach('direct_codex_news','Direct Codex financial news','language_model','codex',native_cadence='hourly',news_layer='headlines_and_selected_context',requires='codex_authentication'),
    Approach('hosted_langgraph','Deployed FinanceAgent LangGraph','agentic','langgraph',native_cadence='hourly',news_layer='actual_graph_context',requires='supported_authenticated_hosted_backend'),
    Approach('jev_systemone','TypeSafe System One / Jev','language_model','jev',news_layer='not_consumed',requires='approved_vendor_budget_and_credentials'),
    Approach('qwen_swarm_mode_a','Qwen3.6-27B fixed role swarm','agentic','qwen',news_layer='not_consumed',requires='approved_GPU_job_and_verified_checkpoint'),
)


# Existing crypto books have native BTC/ETH hourly candles, 24/7 sessions and their
# own cost/evaluator protocol. They remain explicit blocked entries in equity
# cohorts rather than being mislabeled as equivalent S&P500 strategies.
APPROACHES += tuple(Approach('crypto_'+name,'Crypto '+name.replace('_',' '),'optimization' if name in ('min_variance','mean_variance','cvar') else 'control',
    'crypto_native',native_cadence='hourly_crypto_24x7',requires='native_BTC_ETH_cohort_and_crypto_evaluator_adapter',version='crypto-hourly-paper-v1')
    for name in ('cash','btc_buy_hold','equal_buy_hold','min_variance','mean_variance','cvar'))
APPROACHES += tuple(Approach('classical_'+name+'_etf_news','MCP classical '+name+' ETF news sensitivity','optimization','classical_news',
    news_layer='derived_sentiment_uncertainty_mean_covariance_scenarios',requires='eligible_licensed_ETF_search_metadata',strategy_name=name,version='etf-search-metadata/1')
    for name in ('min_variance','mean_variance','cvar'))

# The same models run through the deployed agent: one prompt + explicit MCP tool request
# per model and hour, in books separate from the direct runs, so agent errors are visible.
AGENT_TOOLS = {
    'min_variance': ('recommend_classical_portfolio', {'algorithm': 'min_variance'}),
    'cvar': ('recommend_classical_portfolio', {'algorithm': 'cvar'}),
    'ppo': ('recommend_portfolio', {}),
}
APPROACHES += tuple(Approach('agent_'+model,'Agent via MCP tool: '+label,'agentic','agent_tool',native_cadence='hourly',
    requires='supported_authenticated_hosted_backend',strategy_name=model)
    for model,label in (('min_variance','min variance'),('cvar','CVaR'),('ppo','PPO')))

def evaluated_optimizer_variants():
    # Exact finite evaluated grids declared by selection.protocol and classical.benchmark.
    # Distinct parameter settings receive distinct books; default-equivalent settings
    # already have their canonical baseline ID and are not duplicated under aliases.
    variants=[]
    for lookback in (20,60,120):
        if lookback!=60:
            variants.append(Approach(f'min_variance_l{lookback}',f'Min variance lookback {lookback}','optimization','baseline',strategy_name='min_variance',parameters={'lookback':lookback}))
        for risk in (1.,5.,20.):
            if lookback==60 and risk==5.:continue
            variants.append(Approach(f'mean_variance_l{lookback}_r{int(risk)}',f'Mean variance lookback {lookback} risk {risk:g}','optimization','baseline',strategy_name='mean_variance',parameters={'lookback':lookback,'risk_aversion':risk}))
    for lookback in (60,120):
        for alpha in (.9,.95):
            if lookback==60 and alpha==.95:continue
            variants.append(Approach(f'scenario_cvar_l{lookback}_a{int(alpha*100)}',f'Scenario CVaR lookback {lookback} alpha {alpha}','optimization','baseline',strategy_name='scenario_cvar',parameters={'lookback':lookback,'alpha':alpha,'scenario_method':'bootstrap','n_scenarios':500,'seed':0}))
    for algorithm in ('min_variance','mean_variance','cvar'):
        for lookback in (20,120,252):
            variants.append(Approach(f'classical_{algorithm}_l{lookback}',f'MCP classical {algorithm} lookback {lookback}','optimization','classical',strategy_name=algorithm,parameters={'lookback':lookback}))
    return variants


def registry(extra_policies=()):
    """Persist actual additional frozen candidates as distinct variants, never overwrite IDs.

    Recursive research controllers propose candidates, and are not an investment policy.
    A discovered evaluated candidate must be passed with its own immutable identity.
    """
    rows=[a.to_dict() for a in (*APPROACHES,*evaluated_optimizer_variants())]+list(extra_policies)
    ids=[r['approach_id'] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError('duplicate approach identity')
    return rows
