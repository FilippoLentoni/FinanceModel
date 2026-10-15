ETF news feature research (beta)

get_etf_research_inputs takes symbols and an optional decision_at, then reads frozen central
platform news manifests. Only validated, licensed search metadata published before retrieval,
retrieved before platform availability, and available by the decision enters the 24-hour view.
Unknown rights, incomplete dates and missing data produce explicit missingness.

Protocol etf-search-metadata/1 orders each requested ETF's vector as sentiment, event_count,
uncertainty, age_hours, missing. Headline lexicon is deterministic and uncalibrated. PPO state
extension is exposed, but production checkpoints are unchanged; retraining requires historical
point-in-time coverage and research_and_training rights. Research-only rights are insufficient.
LLM context is cited metadata, treated as untrusted evidence. It is not a validated trading policy.

For a matched optimizer sensitivity, supply baseline_analysis_id with frozen ETF-only solver
inputs at the same decision time, OR input_snapshot_id of approved prices and optional algorithm.
The snapshot path filters approved requested ETFs from mixed data, requires 20 aligned sessions,
uses existing adjusted-price loading, 10% cash and 60% cap (100% for a single ETF), equal synthetic
starting weights, and one-session horizon. Requested symbols missing from prices yield ineligible.

The comparison keeps timing, prices, solver settings and current weights constant. It assumes a
maximum +/-5 basis points per-session return shift for sentiment and up to 25% risk scaling for
uncertainty. This is a sensitivity experiment, not a calibrated forecast. Outputs include input
checksums, weights, turnover and solver risk metrics; no accuracy or profitability claim is made.
Newly collected news cannot create a historical backtest. Matched forward net outcomes remain
necessary before claiming improvement. No new cloud training or automatic production activation.
