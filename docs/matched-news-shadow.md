Matched news shadow experiments and isolated $100 paper profile

Run locally (no AWS credentials, paid requests, training or orders):

  .venv/bin/python scripts/run_etf_shadow.py --study /private/study.json \
      --news /private/frozen-news.json --output /private/report.json

The study manifest declares symbols (default VOO,GOOGL), initial_cash=100, algorithm, fee_bps
(default10), slippage_bps(default5), market_data_synthetic, and steps. Each step contains:
- decision_at and snapshot_available_at (real point-in-time UTC timestamps)
- history: at least21, at most253 aligned observations with session_at, available_at and
  prices:{VOO:number,GOOGL:number}; both times must precede the decision
- snapshot_checksum: sha256 of canonical history JSON (sorted keys, compact UTF8)
- next_open_at, next_open:{symbol:price}, close_at, close:{symbol:price}; open strictly after
  policy formation and close no earlier than open; steps cannot overlap

News is an immutable api/research-news-response corpus. Each original publication, retrieval,
availability, receipt/payload hash, research rights and quality declaration is preserved. A recent
headline cannot be replayed before actual availability. Missing news or price vintage returns
status=ineligible, performance=null, dated blockers and a prospective_protocol. Today’s downloaded
price history also cannot claim it was frozen years ago. No forecasts become realized metrics.

Eligible runs solve both arms from identical price inputs and price-only current holdings, then
execute each arm’s targets in its own synthetic book at the same next observed opens. This common
policy holding state isolates news effects; realized book holdings/costs may diverge. Both use
10% cash, 60% asset cap and the same cost model. News coefficients remain hypotheses (+/-5bps mean,
0..25% uncertainty risk scaling). Reports include net price return, max drawdown, traded-notional
turnover, fees and slippage. Cash distributions/splits are not modeled; returns are price returns,
not total-return claims. Use corporate-action-free forward periods or extend the manifest/accounting
before a dividends-inclusive study. PPO/LLM performance remains ineligible without frozen compatible
policies and permitted corpus. A small favorable sample does not establish predictive accuracy.

Separate saved paper profile

finplan_model.paper_profile.initialize(client, caller_id=...,profile_id='etf-stock-100',as_of=...,
receipt=...,persist=...,protected_portfolio_ids=[...]) uses existing operator-authenticated APIs.
client(method,path,body=None) must return response JSON; authentication is not supplied by profile
metadata. Persist the receipt privately before/after every API call. Creation is idempotent for the
explicit caller/profile; initial cash is synthetic $100 and zero quantities VOO/GOOGL. Protected old
books are read before/after, never written. A changed caller/profile/date conflicts instead of
resetting capital. All returned IDs remain explicit. The existing $10,000 book is never inferred
as this study. Recommendations for this explicit profile use VOO/GOOGL from approved mixed data;
ordinary saved books keep the original approved universe. Profile initialization does not apply
recommendations, place orders or enable daily rebalance execution.
