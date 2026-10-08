# Bootstrap runbook (FinanceModel)

Task 10.5 of `add-research-job-foundation`. Design: D10, Migration Plan step 1. Contracts: D6,
D11, D12 and the shared bootstrap runbook in the FinancialPlanning contract package, applied here
to FinanceModel. It mirrors the FinancialPlanning bootstrap that is already deployed
(`FinancialPlanning/docs/bootstrap.md`).

> **Do not run the bootstrap during implementation work.** A human runs it once, after this IaC
> is synthesized. Afterwards only the scoped roles it creates deploy anything.

## Approval status

- The user approved the one-time bootstrap **in principle on 2026-10-07** (design, "Observed
  facts"; tasks 10.5).
- The entry point refuses to start without a synthesized cloud assembly.
- Before deploying anything it prints the **exact stacks** with their resource types and a
  **monthly cost estimate** priced from the AWS Price List API at run time. No price is written
  in this repository. The operator then types `deploy`; anything else stops the bootstrap with
  nothing deployed.

## Prerequisites

1. **The FinancialPlanning bootstrap has run.** FinanceModel reuses what it created and never
   creates its own:
   - the permission boundaries `finplan-<env>-permission-boundary`,
     `finplan-<env>-research-permission-boundary` and `finplan-shared-permission-boundary`;
   - the project budget (USD 50), its alerts, its 100% deny action and the budget-state writer;
   - the default allocation `/finplan/shared/financialplanning/config/budget-allocation`
     (`cpu_research` 7, `gpu` 25);
   - the connection reference `/finplan/shared/financialplanning/config/codeconnection-ref`.
2. **AWS credentials** in `us-east-2` from the default credential chain. The DevDesktop
   instance role works as is; no `botocore[crt]` is needed (only `aws login` sessions need it,
   then run with `uv run --with "botocore[crt]" ...`). The user's existing credentials are
   acceptable (contracts OQ-11). A root caller is not refused, and the bootstrap prints the
   scoped, MFA-protected role recommendation.
3. **Local untracked configuration**, read the same way as the platform bootstrap reads it:
   `--config PATH`, else `$FINPLAN_BOOTSTRAP_CONFIG`, else the shared
   `~/.finplan/bootstrap.json` that the FinancialPlanning bootstrap already uses. Only its
   `account_id`, `primary_region` and `codeconnection_arn` are used; its platform-only keys
   (repository, pipeline name, budget notification address) are ignored and never printed.
   An optional overlay `~/.finplan/financemodel-bootstrap.json` can override those values, and
   `FINPLAN_ACCOUNT_ID`, `FINPLAN_PRIMARY_REGION` and `FINPLAN_CODECONNECTION_ARN` override both.
   The script refuses a configuration file inside the repository. `repo`, `github_repository`
   (`FilippoLentoni/FinanceModel`, from `config/shared.json`) and `pipeline_name` are always
   FinanceModel's.

   Without a `codeconnection_arn` the bootstrap reuses **the same existing CodeConnection as the
   platform**, read-only from `/finplan/shared/financialplanning/config/codeconnection-ref`.
4. **Toolchain:** `uv` (Python 3.12) and Node.js for `npx aws-cdk@2`.

## What the bootstrap deploys

Exactly two account-level stacks (environment `shared`). It never deploys an environment stack.

| Stack | Contents |
|---|---|
| `finplan-shared-financemodel-pipeline-store` | The pipeline store bucket `finplan-shared-financemodel-pipeline-store-<account-id>`: SSE-S3, TLS only, Block Public Access, versioned. It holds pipeline artifacts, the content-addressed CDK file assets (`assets/`), the release ledger (`releases/`, never expires) and the staged tooling template (`bootstrap/`). Retained when the stack is deleted. Legacy synthesizer: deployed inline, no CDK bootstrap role or bucket |
| `finplan-shared-financemodel-tooling` | The image repository `finplan-shared-financemodel-cpu-images` (immutable tags, scan on push, the newest 10 release images kept), the pipeline `finplan-shared-financemodel-pipeline` with its scoped roles, its four CodeBuild projects (build, and one stage project per environment) and their explicit `/aws/codebuild/<project>` log groups (30-day retention, deleted with the stack; matrix row `pipeline-financemodel`) ([pipeline.md](pipeline.md)). Its template exceeds the 51,200-byte inline limit, so the CLI stages it in the store under `bootstrap/` with the operator's credentials |

Neither stack needs the CDK bootstrap stack (`CDKToolkit`). The bootstrap refuses an assembly
that references `cdk-hnb659fds` roles or `cdk-*-assets` buckets, or that carries container-image
assets (a lesson of the platform bootstrap).

Scoped roles created:

| Role | Tags and boundary | Used by |
|---|---|---|
| `finplan-shared-financemodel-pipeline-role` | `shared`, shared boundary | CodePipeline |
| `finplan-shared-financemodel-pipeline-build-project-role` | `shared`, shared boundary | Build stage (gates, synth, image push, assets, ledger) |
| `finplan-shared-financemodel-deploy-role-<env>` | `<env>`, `<env>` boundary | Deploy actions |
| `finplan-shared-financemodel-deploy-role-<env>-exec` | `<env>`, `<env>` boundary | CloudFormation. Limited to `finplan-<env>-financemodel-*` resources. Creates only roles that carry the `<env>` or the `<env>` research boundary. Writes SSM only under `/finplan/<env>/financemodel/` |
| `finplan-<env>-financemodel-pipeline-stage-role` | `<env>`, `<env>` boundary | Promotion check, release publishing, registry seeding, environment tests |

## Steps

```sh
# 1. synthesize the cloud assembly (offline)
uv run python scripts/synth.py --out cdk.out

# 2. bootstrap (interactive; a human step)
AWS_REGION=us-east-2 uv run python scripts/bootstrap.py --assembly cdk.out   # instance role; no botocore[crt]
```

`scripts/bootstrap.py` drives the contract sequence
`finplan_contracts.bootstrap.run_bootstrap` and stops at the first failing step.

| # | Step | Stops when |
|---|---|---|
| 1 | Assembly: copy only the two tooling stacks into `cdk.out.bootstrap/` and check them | No assembly, missing tooling stacks, or CDK bootstrap references |
| 2 | Pre-run plan: exact stacks, resource types, estimate (Price List API) | n/a (read-only) |
| 3 | Caller: the STS account equals `account_id`. A root caller proceeds and gets the recommendation | The account differs |
| 4 | Region equals `primary_region` | It differs |
| 5 | Connection: the reused CodeConnection is `AVAILABLE` (read-only) | Any other status |
| 6 | Scoped roles: every deploy action uses a scoped deploy role with a boundary | A deploy action without one |
| 7 | Confirmation: the operator types `deploy` | Anything else. Nothing is deployed |
| 8 | Writes `/finplan/shared/financemodel/config/codeconnection-ref` | n/a |
| 9 | `npx aws-cdk@2 deploy --all` of `cdk.out.bootstrap/` with `SourceDryRunPassed`. First reports, read-only, whether the shared budget allocation exists; it never writes a budget parameter. After the deploy succeeds it writes, as the `bootstrap` writer: `/finplan/shared/financemodel/config/budget-enforced-role-names` (the pipeline and build role names; contracts 1.0.0, D16) and, **only where absent or stale**, `/finplan/<env>/financemodel/config/instance-prices` for beta, gamma and prod with on-demand SageMaker Processing prices fetched from the AWS Price List API at that moment (`scripts/instance_prices.py`). It then reports whether the operator has set `config/integration-snapshot-id` in beta and gamma (never written) | A CloudFormation failure (automatic rollback). A price the API does not return is a warning, not a stop: paid jobs stay refused until `scripts/instance_prices.py --write` succeeds |
| 10 | Source-stage dry run: the pipeline is created with the transition into Build disabled; one execution must fetch `main` of `FilippoLentoni/FinanceModel`; then the transition is enabled and the result recorded in `~/.finplan/financemodel-source-dry-run.json` | Source cannot fetch the repository: *extend the GitHub App installation of that connection to `FilippoLentoni/FinanceModel`, then rerun*. The deploy stages stay disabled |

## After the bootstrap

- **Operator parameters** ([operations.md](operations.md#configuration-an-operator-writes-per-environment)).
  Before the first pipeline run reaches `IntegrationBetaTests`, write
  `/finplan/beta/financemodel/config/integration-snapshot-id` (and the gamma one before
  `GammaTests`): an approved **synthetic** platform snapshot of that environment. The bootstrap
  prints `[ACTION]` lines for any that are missing. Instance prices expire after 30 days: refresh
  them with `uv run python scripts/instance_prices.py --write`.
- **Budget enforcement.** FinanceModel publishes two kinds of role-name lists for the platform's
  budget action (contracts D4, D16):
  - `/finplan/shared/financemodel/config/budget-enforced-role-names` (this bootstrap): the
    account-level `finplan-shared-financemodel-pipeline-role` and
    `finplan-shared-financemodel-pipeline-build-project-role`;
  - `/finplan/<env>/financemodel/config/budget-enforced-role-names` (every deploy,
    `PublishRelease`): the job API handler, dispatcher, state handler and job-execution roles and
    that environment's deploy, execution and stage roles.

  The platform's budget action reads the published names **only when the FinancialPlanning
  bootstrap runs**. **Rerun the FinancialPlanning bootstrap after the first FinanceModel beta
  deploy**, so the 100% deny also covers FinanceModel. FinanceModel never creates a second budget.
- **Contracts.** The pin is `finplan-contracts` 1.1.0, allowed in every environment, and its
  ownership matrix carries every FinanceModel row: the job API, the job-execution role, the
  registry bucket policy and the CodeBuild log groups all deploy.
- **Pipeline changes** (the tooling stack itself) deploy only by rerunning the bootstrap. The
  pipeline never updates itself.

## Teardown

Only when FinanceModel is retired. Delete the environment stacks first:

- `finplan-<env>-financemodel-control`;
- then `finplan-<env>-financemodel-storage`. Its buckets and the job table are retained; the prod
  table has deletion protection.

Then delete the two account-level stacks, tooling first. Both have termination protection:

```bash
for stack in finplan-shared-financemodel-tooling finplan-shared-financemodel-pipeline-store; do
  aws cloudformation update-termination-protection --region <region> --stack-name "$stack" --no-enable-termination-protection
  aws cloudformation delete-stack --region <region> --stack-name "$stack"
  aws cloudformation wait stack-delete-complete --region <region> --stack-name "$stack"
done
```

The store bucket, the image repository and the environments' buckets are retained. Empty every
object version before deleting a bucket, as described in the platform runbook.

## Tests (no AWS)

`tests/unit/infra/test_build_and_bootstrap.py` runs the whole sequence with fake STS,
CodeConnections, CodePipeline and pricing clients, moto SSM and a fake deploy runner. It checks:

- the reused platform connection;
- that only the two tooling stacks deploy;
- that no budget parameter is written, only the shared tooling role names (validated against the
  contract key);
- that instance prices come from the (faked) Price List API's Processing price and that
  `integration-snapshot-id` is reported, never written (`tests/unit/infra/test_instance_prices.py`
  also covers the keep-current, refresh-stale and missing-price cases);
- that a declined confirmation, a wrong account, an unavailable connection and a failed dry run
  each stop the bootstrap before anything is deployed;
- that the bootstrap refuses a missing assembly.

`tests/unit/infra/test_pipeline_and_tooling.py` checks that the bootstrap assembly holds no CDK
bootstrap reference and that its template is staged in the FinanceModel store.

## Never

- Commit the account ID, the connection ARN or any role ARN.
- Run the bootstrap before `scripts/synth.py` has produced the assembly.
- Deploy environment stacks with the bootstrap identity.
- Create a budget, a budget allocation or a permission boundary from FinanceModel.
