# Add approval-gated current-book Jev bridge

## Why
The existing Jev managed worker only supports historical daily benchmarks. Hourly preflight correctly captures a public request without invoking a model, but an approved historical pilot cannot become the current isolated book's policy.

## What Changes
Add a typed beta backend producer, explicit forward-one-decision worker and strict read-only hourly reuse. Preserve existing public schemas, budget/manual-approval controls and paper execution policy. Add narrow paired beta managed-worker reads for isolated paper-book eligibility. Protect paid invocation with permanent binding, conditional claim and original-response recovery.

## Impact
FinanceModel control/runtime/artifact store/release references and FinancialPlanning beta portfolio GET authorization. Existing books/history, gamma/prod permissions and original hourly runs are preserved. No direct platform DynamoDB access, real brokerage activity or automatic approval.
