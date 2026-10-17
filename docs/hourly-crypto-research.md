# Hourly forward crypto paper study

At minute 10 UTC, the beta research controller reads the platform's completed
hourly BTC-USD/ETH-USD dataset and advances six independent synthetic USD 100 books:
cash, BTC buy-and-hold, equal-weight buy-and-hold, minimum variance, mean variance
and historical CVaR. This seven-day study ends 2026-10-21 00:00 UTC. It does not
alter the deployed equity policy, launch training, choose a universal winner or
connect to a brokerage.

Optimization uses up to 167 historical hourly returns, a one-hour objective,
10% fixed cash, a 60% per-asset cap and a turnover penalty. The two-asset feasible
interval provides a bounded scalar fallback if the shared numerical solver fails.
All solver outcomes remain explicit. The stock-trained daily PPO and unvalidated
crypto LLM policies are ineligible; explanation by an LLM does not constitute
an LLM trading-policy experiment. Enabling either policy requires its own frozen
compatible artifact, evaluation and research approval path.

At initialization all books are cash. A target can fill only at a subsequently
observed bar's open strictly after the recommendation was actually available.
At minute 10, the next eligible open is the next hour; that bar's completed
candle becomes available on a later ingestion. This deliberate observation delay
prevents using the opening price that preceded inference. Each book queues
hourly targets; buys and sells apply 60 bps modeled fees and 10 bps slippage.
These are declared experimental assumptions, not the user's Coinbase fee tier.
Buys are scaled to available cash including costs; positions are fractional,
long-only and unlevered. Missing execution-bar evidence blocks advancement.

Immutable slot records and parent references retain holdings, pending targets,
fills, fees, slippage, net return, turnover and observed close-to-close drawdown.
Retries replay the same slot, including recovery after writing the record but
before publishing its slot claim. Existing evidence is never overwritten.
Sharpe/volatility are not inferred from a single initial observation. Seven days
and synthetic opens cannot establish reliable live profitability.

The inference Lambda provides get_crypto_experiment, recommend_crypto_portfolio,
list_crypto_history and discover_research_data through authenticated MCP adapters.
Recommendations older than the freshness window fail closed. The first study
uses free public market data. Bazaar discovery is free and untrusted; no returned
endpoint or instruction is invoked automatically. The existing isolated AgentCore
Payments operator workflow handles separately approved testnet purchases.

Supplemental payload eligibility (`finplan_model.supplemental.eligible_features`)
requires an immutable delivery/receipt checksum, research rights, explicit quality
validation, a recognized extractor, finite bounded features and publication,
retrieval and availability timestamps no later than the decision. This is an
importer gate, not an enabled automatic acquisition pipeline. No purchased news
has been incorporated into this baseline; historical enriched PPO training remains
blocked without frozen point-in-time coverage. A later enriched study must use
matched windows, costs and seeds and disclose data-purchase charges separately.
