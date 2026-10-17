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

Opt-in PPO environment integration

EnvSpec features can explicitly include etf_news_metadata_v1. It adds five values per instrument,
in instrument order: sentiment, min(event_count,100)/100, uncertainty, min(age_hours,24)/24,
and missing. The feature name versions this normalization and changes configuration_id.
Existing feature defaults, observation bytes and default environment documents are unchanged.

build_observation requires news_dataset, instruments and decision_time when this group is selected.
PortfolioEnv and RLPolicyStrategy accept news_provider(instruments, decision_time), which returns
api/research-news-response. PriceArrays.from_market retains exact market decision timestamps;
legacy arrays without those timestamps cannot enable news. FrozenEtfNewsProvider freezes a corpus
and exposes corpus_checksum for experiment provenance; use the checksum with policy artifacts.
It preserves original publication/retrieval/availability dates at every historical request.

Training observations require research_and_training rights plus eligible 24-hour coverage for
EVERY instrument at EVERY decision. Research-only, future, missing or stale inputs fail training.
Evaluation can represent excluded data using explicit missingness, provided a source/provider was
supplied. No provider is inferred, no historic data is manufactured and no checkpoint is retrained
or activated by selecting the feature. Existing job protocols remain unchanged unless explicitly
configured and wired with a frozen provider. No measured accuracy improvement is claimed.

A realized-outcome matched experiment runner and separate VOO/GOOGL $100 paper-profile helper are
now available; see docs/matched-news-shadow.md. They preserve publication/snapshot vintage blockers.
Optimizer sensitivity alone remains distinct from this costed out-of-sample shadow study.

Strict actual-model completion (supersedes readiness-only interpretation)

The saved $100 VOO/GOOGL profile now requires licensed current news in actual min-variance,
mean-variance and CVaR solver inputs. It fails closed if either instrument lacks source coverage.
The numerical recommendation embeds consumed features, source receipt IDs, baseline/enriched input
checksums and uncalibrated assumptions in diagnostics.news_evidence. Ordinary existing portfolios
retain the price-only path. Profile policy formation uses actual issuance time, with completed
price bars; an explicit historical as_of keeps its historical cutoff and rejects later news.

News-aware PPO inference now dispatches to the correct observation implementation and requires
etf_news_metadata_v1 plus frozen training corpus identity, five compatible actor members and eligible
current source coverage. Export retains the training corpus checksum. No legacy actor is padded,
renamed or treated as compatible. GOOGL is supported by the versioned financial-news feature adapter.

Training jobs can use selection_protocol.rl.news_corpus={dataset:<frozen API corpus>,checksum:<SHA>}
when explicitly selecting the news feature. Corpus identity enters configuration_id. The low-level
trainer and selection job inject FrozenEtfNewsProvider into environments and common evaluation.
Preflight checks every possible train and validation decision for research_and_training rights and
24-hour publication/retrieval/availability coverage BEFORE creating a learner. A research-only
search receipt cannot train this checkpoint. Current forward headlines therefore do not meet the
PPO requirement, and this change's strict completion task remains OPEN.

Run .venv/bin/python scripts/check_news_exit.py /private/evidence.json. It exits 1 unless actual
central paid-delivery receipt checksums correlate with optimizer and trained PPO diagnostics and
verified hosted tool evidence. Readiness/ineligible/fixture actor output cannot satisfy completion.
Operator evidence must independently verify actor training and real market provenance; the checker
is not a substitute for authenticated API/artifact verification.

Bounded cost planning (read-only preflight, no jobs submitted): current configured ml.m5.xlarge
price was $0.23/hour, retrieved 2026-10-10T23:47:38.485292Z. Existing recursive_evaluate hard cap is
900 seconds, one CPU instance: $0.0575 compute plus $0.01 configured storage allowance=$0.0675/run.
Five such seed jobs plus a 600-second export upper bound total approximately $0.385834, excluding
other AWS activity and data licensing. The control plane must recheck price age, project budget and
actual job estimates before submission. Completion within the time cap is not guaranteed.

A window-2, two-asset actor has 17 input dimensions with default price/holding features plus news;
8 sessions are merely the mechanical minimum for disjoint tiny train/validation/test fixtures, not
credible financial evaluation. The existing 60-session study needs at least 62 training sessions
plus independent validation/test coverage. Current research-only news cannot meet even the rights
minimum. A licensed historical corpus, or separately audited public-domain archive with verifiable
historical content availability and explicit no-event coverage, is required. Do not backdate today's
retrieval or label an absent event feed complete. Such archive support needs a separate contract and
acquisition change before it can replace the current coverage gate.
