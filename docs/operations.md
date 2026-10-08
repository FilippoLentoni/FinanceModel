# Operations

Task 10.6 of `add-research-job-foundation`: prices, budget categories, approval, cancellation and
rollback. Design D3, D4 and the Migration Plan. Mechanics: [job-execution.md](job-execution.md),
[pipeline.md](pipeline.md), [bootstrap.md](bootstrap.md). This page contains no account identifier,
ARN, endpoint or price; the `leak-scan` build gate checks it.

## Configuration an operator writes (per environment)

Each value is SSM configuration under `/finplan/<env>/financemodel/config/`, never a repository
file. FinanceModel's deploy (the pipeline) writes none of these. The bootstrap writes only
`instance-prices`, and only where that is safe (absent or stale), from the AWS Price List API.

| Parameter | Value | Required? | Who writes it | Default when absent |
|---|---|---|---|---|
| `instance-prices` | `{"retrieved_at": "<RFC 3339>", "currency": "USD", "usd_per_hour": {"ml.m5.xlarge": <price>}, "source": "AWS Price List API", ...}`: on-demand SageMaker **Processing** hourly prices in the deployment region. Prices older than `cost.price_max_age_days` (30) are refused | Yes, for any paid job | The bootstrap (absent or stale only), then the operator every 30 days with `scripts/instance_prices.py --write` | **None.** Paid jobs fail with `PRECONDITION_FAILED` until it is set |
| `auto-approve-usd` | Estimates at or below it start without approval (FM-OQ-3) | No | Operator | `0`: every paid job waits for approval |
| `lease-limits` | `{"cpu": 1, "gpu": 0}` | No | Operator | Configuration defaults (one CPU slot; the `ml.m5.xlarge` processing quota is 1 for the whole account) |
| `production-candidate-principals` | JSON list of role names allowed to submit `production_candidate` runs (platform-side principals only, never tool roles) | No | Operator | Empty |
| `integration-snapshot-id` | (beta, gamma) An approved **synthetic** platform snapshot ID (`snap_...`, for example one from the platform's fixture ingestion) that the deployed suite's fixture job runs on. The suite cancels its run before it starts while `auto-approve-usd` is 0 | **Yes, before the first pipeline run reaches `IntegrationBetaTests`** (and `GammaTests`) | Operator only: the bootstrap reports whether it is set but never writes it, because only the operator knows which approved snapshot of that environment to use | **None.** The beta/gamma suite fails and names the parameter |

Instance prices (no price is ever written in this repository):

```sh
uv run python scripts/instance_prices.py                    # show what would be written (read-only)
uv run python scripts/instance_prices.py --write            # write absent or stale prices, all environments
uv run python scripts/instance_prices.py --write --force --env beta   # refresh a current value
```

The script asks the AWS Price List API (`AmazonSageMaker`, `component` `Processing`, the
deployment region, every instance type of a deployed job type) and refuses to write when a price is
missing. It writes as the `bootstrap` writer of the contract SSM rules.

Integration snapshot (beta and gamma; pick an **approved, synthetic** snapshot of that environment
from the platform, for example with its `GET /v1/snapshots` operation):

```sh
aws ssm put-parameter --name /finplan/beta/financemodel/config/integration-snapshot-id \
  --type String --value snap_<id> --overwrite
```

## Budget categories (shared, FinancialPlanning-owned)

| Category | Default (USD) | FinanceModel use |
|---|---|---|
| `cpu_research` | 7 | Every phase 1 job type (`prepare_dataset`, `run_backtest`, `run_benchmark`, `report`) and the pipeline's test runs |
| `gpu` | 25 | Reserved for the next change; always needs approval |
| `platform_infra`, `bedrock_explanations`, `reserve` | 8, 5, 5 | Not spent by FinanceModel jobs |

- **Where the caps live.** `/finplan/shared/financialplanning/config/budget-allocation`.
  FinanceModel reads it and never writes it; it is the same map for beta, gamma and prod.
- **Remaining budget.** The cap minus the upper-bound estimates of this environment's runs in that
  category. Actual billed cost reconciliation is not implemented yet.
- **Account-wide guard.** The FinancialPlanning AWS Budgets deny at 100% of USD 50. While
  `/finplan/shared/financialplanning/config/budget-state` is `enforced`, every paid submission,
  approval and start fails with `BUDGET_EXCEEDED`.
- **Enforced role names.** FinanceModel publishes its per-environment role names at
  `/finplan/<env>/financemodel/config/budget-enforced-role-names` (pipeline, every deploy) and its
  account-level pipeline and build role names at
  `/finplan/shared/financemodel/config/budget-enforced-role-names` (bootstrap; contracts 1.0.0,
  D16). Rerun the FinancialPlanning bootstrap after the first FinanceModel deploy so the deny
  action covers them.
- **Lifting the cap** is a human step, done in the FinancialPlanning runbook.

## Approving a paid run (FM-OQ-6 interim: CLI)

Paid runs above `auto-approve-usd`, and every GPU run, wait in `awaiting_approval`. Only
`finplan-<env>-financemodel-approver-role` may approve. No `finplan-*` service role can assume it,
and the API denies tool, agent, pipeline and FinanceModel service roles.

```sh
# assume the approver role (a profile that assumes finplan-beta-financemodel-approver-role)
AWS_PROFILE=<approver-profile> uv run python scripts/approve_run.py --env beta <run_id> --show   # inspect
AWS_PROFILE=<approver-profile> uv run python scripts/approve_run.py --env beta <run_id>          # confirm, approve
```

The CLI shows the estimate, the budget category, the remaining allocation and the price date
before it asks for confirmation. Unapproved runs end `cancelled` (`approval_expired`) after 72
hours.

**Status.** The job API deploys with every release since contracts 1.0.0. The approval command
has been verified offline only (`tests/unit/control/test_approve_cli.py`), not yet against beta.

## Cancelling a run

```sh
AWS_PROFILE=<approver-profile> uv run python - <<'EOF'
import boto3
from scripts.approve_run import signed_request
s = boto3.session.Session(region_name="us-east-2")
endpoint = s.client("ssm").get_parameter(Name="/finplan/beta/financemodel/api/job-endpoint")["Parameter"]["Value"]
print(signed_request("POST", f"{endpoint}/v1/jobs/<run_id>/cancel", region="us-east-2",
                     credentials=s.get_credentials(), body={"idempotency_key": "cancel-<run_id>"}))
EOF
```

Any principal the job API admits may cancel its runs (the approver role and the granted tool
roles).

The rules:

- runs in `awaiting_approval` or `queued` (and `starting` runs without a job) end `cancelled` at
  once;
- a started run goes to `stopping`, SageMaker is asked to stop it, and the run ends `cancelled`
  once SageMaker confirms;
- cancelling a terminal run returns it unchanged;
- nothing is staged for a cancelled run.

## Rollback (with in-flight cancellation)

1. **Cancel in-flight runs first.** List the environment's non-terminal runs (`GET /v1/jobs` with
   `state` set to each of `awaiting_approval`, `queued`, `starting`, `running`) and cancel them.
   A run must never start on job definitions that are about to change (design, Migration Plan
   step 4).
2. **Start the pipeline** with the variable `rollback_to_release_id=rel_...`. The Build stage
   re-emits the stored BuildOutput from `releases/<release_id>/` after re-verifying its digest.
   Nothing is rebuilt.
3. **The stages redeploy** that release. The job definitions point back to its image digest (the
   image repository keeps the newest 10 release images; older releases cannot be rolled back to),
   and every manifest records `rolled_back_from`.
4. **Data is never rolled back.** Run records, registry records and lineage are immutable and stay.

## Contracts pin

`finplan-contracts` **1.0.0** (the first stable release) is pinned by version and wheel digest
(`contracts-pin.json`, `pyproject.toml`, `uv.lock`) and allowed in beta, gamma and prod, so
`PromotionCheck` no longer stops gamma and prod (it still refuses a 0.x pin outside beta, per the
contract rule). The wheel is the FinancialPlanning platform's reproducible build, vendored byte for
byte; the platform publishes the same bytes to CodeArtifact once its bootstrap has been re-run. To
re-pin later:

1. vendor or fetch the new wheel;
2. update `contracts-pin.json`, `pyproject.toml` and `uv.lock` (`uv lock`, then `uv sync --locked`);
3. rerun the gates (`scripts/check_contracts_pin.py --env prod` and the pre and post gates).
