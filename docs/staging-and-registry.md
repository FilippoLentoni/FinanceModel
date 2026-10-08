# Run-output staging and the model registry

Task group 9 of `add-research-job-foundation`. Specs: run-output-staging (RST-01 to RST-05) and
model-registry (REG-01 to REG-04). Contracts: D1 (ownership), D10 (staging protocol: manifest
last, explicit platform accept call), the staged-output manifest schema and
`api/get-staged-output-response`. Platform side: `FinancialPlanning/platform/finplan_platform/core/staging.py`.

## Staging (`finplan_model.staging`)

| Rule | Implementation |
|---|---|
| Who stages (RST-01) | `staging_decision(purpose, result)`. Only a `production_candidate` run that ended `succeeded` with `solution_status` `optimal`, `feasible` or `no_effect`. Every other run writes only to research storage, and its result says why: `staging.status` is `not_staged`, with a `reason` such as `purpose_not_production_candidate` or `solution_status_infeasible` |
| What (RST-02) | `plan-content.json` (finance `plan-content`, built from `payload.proposed_allocation` and also inlined as `payload.plan_content`, which the platform compares), `metrics/summary.json`, and `manifest.json`. The manifest validates against the pinned `staged-output-manifest` and its finance payload. It carries `run_id`, `model_version`, `configuration_id`, `input_snapshot_id`, `plan_id`, `parent_plan_version_id`, `completion_status`, `solution_status`, `evaluator_version`, `contract_version`, and the SHA-256 and size of every file. It is validated **before anything is written**: a schema failure raises `VALIDATION_FAILED` and nothing is staged, so the job ends `failed` |
| Order (RST-03) | Data files first, `manifest.json` **last**, under `<run-staging-ref><run_id>/`. There is no marker file. A job that dies before the manifest leaves an incomplete bundle, and the platform answers its accept call with `PRECONDITION_FAILED` `staged_output_incomplete` |
| Write-only, write-once (RST-04) | `S3StagingStore(client, run_staging_ref)` only calls `PutObject` with `If-None-Match: *`; it never reads, lists or deletes. IAM: [pipeline.md](pipeline.md#iam-ws-03-ws-05-rst-04-env-03-to-env-05) |
| Lineage first | With a registry, `stage_run_output` records `run_id -> model_version` before writing, because the platform checks lineage before it commits |
| Result | `stage_run_output` returns the result's `staging` block: `{"status": "staged", "plan_id", "manifest_checksum", "files", "platform_validation": "pending"}` |

The job container calls it after the handler and before writing `job-result.json`:

```python
from finplan_model.staging import S3StagingStore, StagingTarget, stage_run_output

block = stage_run_output(
    S3StagingStore(s3, run_staging_ref),            # /finplan/<env>/financialplanning/config/run-staging-ref
    doc, purpose=spec["purpose"], target=StagingTarget.from_spec(spec),
    registry=registry, spec=spec, actor="finplan-<env>-financemodel-job-execution-role")
doc["staging"] = block
```

**CONTRACT GAP.** The pinned job submission has no `plan_id` or `parent_plan_version_id`.
`StagingTarget.from_spec` reads the run spec's `staging` block (`{"plan_id",
"parent_plan_version_id"}`). Without one, a production candidate stages nothing (`not_staged`,
`no_plan_target`).

### Platform outcome (RST-05; `finplan_model.staging.outcome`)

FinanceModel never calls the accept route. `platform_outcome_hook(platform_client)` is a
`ServiceDeps.result_hooks` entry. On `get_job_result` it reads `GET /v1/staged-outputs/{run_id}`
(with the job API handler role, which the platform grants that route) and folds the outcome into
`staging`:

| Platform outcome | `platform_validation` | `plan_version_id` |
|---|---|---|
| none yet (`NOT_FOUND`) | `pending` | no |
| `accepted` + `validated` / `pending_validation` | same | yes, the platform's |
| `accepted` + `invalid` | `invalid` | **no** |
| `rejected` | `rejected` (plus `rejection_kind`, `platform_error_code`) | no |
| `no_version` | `no_version` | no |

`staging.status` stays `staged`: the run is never reported as a plan version.

## Model registry (`finplan_model.registry`)

Records live in `finplan-<env>-financemodel-model-registry-<account-id>`. Every object is written
once (`If-None-Match: *`). Overwrites and deletes are denied in IAM.

| Key | Content |
|---|---|
| `identity/<sha256 of the canonical identity>.json` | The record minted for (strategy, image digest, parameter schema version, trained-artifact checksum). This is the uniqueness item |
| `versions/<model_version>.json` | The same record, by `model_version` |
| `events/<model_version>/<seq>.json` | Lifecycle events with actor and time: `registered` -> `candidate` -> `promoted`, and any status -> `retired` |
| `runs/<run_id>.json` | Run lineage: the `model_version` the run used |

- **REG-01.** `ModelRegistry.register(identity, actor=)` returns `(record, created)`. An identical
  identity returns the existing `model_version` and writes nothing; a new digest mints a new one.
  Concurrent registrations converge on one record.
- **REG-02.** `update(mv, changes)` with anything but `status` raises `IMMUTABLE_RECORD`.
  `transition(...)` appends an event, and invalid moves are refused.
- **REG-03.** `verify_lineage(run_id, model_version)` confirms the pair or raises `NOT_FOUND`, with
  `details.record_type` `model_version` or `run_lineage`. It is served by the `job-registry-lookup`
  Lambda behind the job API:
  - `GET /v1/registry/lineage/{run_id}?model_version=mv_...` returns `matches: true` plus the
    strategy, image digest, schema version and status;
  - `GET /v1/registry/model-versions/{mv}` returns the record and its status.

  `/finplan/<env>/financemodel/model/registry-ref` = `<job-endpoint>/v1/registry`. The platform's
  `SsmModelRegistry` lookup client calls it with SigV4 (it is admitted by the API resource policy).
- **REG-04.**
  - `model_version_resolver(registry, actor=)` is the control plane's
    `ServiceDeps.model_version_resolver`. It mints or finds the `model_version` at submission, the
    run spec carries it, and the job result includes it.
  - `lineage_result_hook(registry, actor=)` records lineage.
  - `seed_baselines(registry, image_digest, actor=)` registers the six baselines at every deploy
    (`PublishRelease`).

### Wiring in the deployed control plane

`finplan_model.control.handlers.build_service` (task group 7) should add:

```python
from finplan_model.registry import ModelRegistry, S3RegistryStore, lineage_result_hook, model_version_resolver
from finplan_model.staging import platform_outcome_hook

registry = ModelRegistry(S3RegistryStore(s3, ssm_value(cfg.ssm_name("config", "registry-storage-ref"))), clock=clock, ids=ids)
actor = f"finplan-{env}-financemodel-job-api-handler-role"
deps.model_version_resolver = model_version_resolver(registry, actor=actor)
deps.result_hooks += [lineage_result_hook(registry, actor=actor), platform_outcome_hook(platform)]
```

## Tests

| Test | File |
|---|---|
| RST-01, RST-02, RST-03, RST-04 (code), RST-05 | `tests/unit/staging/test_staging.py` |
| RST-02 (contract schema), REG-03, REG-04 (end to end through the control plane and the job container) | `tests/contract/test_staging_registry_contract.py` |
| REG-01, REG-02 (in-memory and moto S3 stores) | `tests/unit/registry/test_registry.py` |
| RST-04, WS-03, WS-05 (IAM policy simulation) | `tests/unit/infra/test_iam_simulation.py` |
