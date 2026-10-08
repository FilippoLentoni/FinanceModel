# FinanceModel foundation: layout, contract pin and shared interfaces

Task group 1 of `add-research-job-foundation` (scaffolding, contract pin, configuration) and the
shared core interfaces every other task group builds on.

## Layout

| Path | Contents |
|---|---|
| `pyproject.toml`, `uv.lock` | uv project, Python 3.12, package `finplan-model` (import name `finplan_model`) |
| `src/finplan_model/core/` | Run context, clock, ID minting, contract errors, artifact store, platform snapshot reads, configuration, contract outcomes |
| `src/finplan_model/sim/` | Paper execution simulator ([simulator.md](simulator.md)) |
| `src/finplan_model/evaluate/` | Common evaluator, comparison guard, replay |
| `src/finplan_model/{datasets,strategies,jobs,control,staging,registry,reporting}/` | Task groups 3 to 9 |
| `infra/app.py`, `infra/stacks/` | CDK app (one `Stage` per environment) and shared stack helpers (`infra/stacks/common.py`) |
| `config/{beta,gamma,prod,shared}.json` | Settings, defaults and SSM parameter names only |
| `container/` | CPU processing image (`financemodel-cpu`, task group 6) |
| `scripts/` | `check_contracts_pin.py`, `build_gates.py`; pipeline and release: `synth.py`, `lambda_bundle.py`, `container_image.py`, `build_stage.py`, `publish_assets.py`, `release.py`, `stage_runner.py`, `bootstrap.py` ([pipeline.md](pipeline.md), [bootstrap.md](bootstrap.md)) |
| `tests/{unit,contract,integration,smoke}/` | Tests; `tests/harness.py` is the offline harness, `tests/fakes/` holds shared fakes |
| `vendor/finplan-contracts/` | The pinned contract wheel (until CodeArtifact exists) |

`tasks.md` task 1.1 names the package `src/financemodel/{simulator,evaluator,...}` and
`containers/cpu/`. The implementation uses `src/finplan_model/{sim,evaluate,...}` and `container/`
(the names every task group of this implementation shares); the responsibilities map one to one
(`simulator` -> `sim`, `evaluator` -> `evaluate`, `api` -> `control`).

## Contract pin

`finplan-contracts` is pinned by exact version **and** by the SHA-256 of the exact wheel:

- `contracts-pin.json`: package, version `1.0.0`, artifact path, `sha256`,
  `served_environments: ["beta", "gamma", "prod"]`;
- `pyproject.toml`: `finplan-contracts==1.0.0` with `[tool.uv.sources]` pointing at the vendored
  wheel; `uv.lock` records the same hash and uv refuses a wheel whose hash differs;
- `scripts/check_contracts_pin.py` verifies all of them ("Digest mismatch" on any difference) and,
  with `--env gamma|prod`, refuses a 0.x pin.

Contracts **1.0.0** is the first stable release (FinancialPlanning design D16): allowed in beta,
gamma and prod, with the ownership rows FinanceModel needs. The vendored wheel is byte-for-byte the
FinancialPlanning platform's reproducible build of 1.0.0; the platform's build stage publishes the
same bytes to its CodeArtifact registry (`/finplan/shared/financialplanning/contract/registry-ref`)
once its bootstrap has been re-run, and refuses to publish different bytes under that version.
FinanceModel serves contract major 1 only (`served_contract_majors`); it was never deployed on 0.x.
No contract schema is ever copied into this repository; the `copied-id` gate enforces it (CS-01).

## Build gates

`uv run python scripts/build_gates.py --stage pre|post|all [--assembly cdk.out] [--env beta]`:

| Gate | Stage | Check |
|---|---|---|
| `contracts-pin` | pre | Pin file, wheel digest, pyproject, uv.lock, installed version |
| `config` | pre | `config/*.json` validate (CPU-only job types, budget categories, simulator defaults) |
| `leak-scan` | pre | Account IDs, ARNs with accounts, bucket names, endpoints, secrets (contract scanner) |
| `copied-id` | pre | No contract schema `$id` or schema copy in the repository |
| `conformance` | pre | Contract consumer conformance suite against the pinned version |
| `live-perm-scan` | pre | No live broker/exchange/wallet client in dependencies, Dockerfiles or source; no live-financial IAM permission in repository documents |
| `unit` | pre | `pytest tests/unit tests/contract` under the offline harness |
| `ownership`, `boundaries`, `live-perm-scan-synth` | post | Synthesized templates: ownership matrix, permission boundary on every role, no live permissions |

Other task groups append gates to `scripts.build_gates.GATES`.

## Offline test harness

`tests/harness.py` (task 1.3, DEP-02) runs for every test: no network, no real AWS call (botocore's
sender is blocked; moto still works), **no SageMaker call at all** (even moto), fake credentials
only. Control-plane tests inject `tests/fakes/sagemaker.py::FakeSageMaker`. Shared fixtures in
`tests/conftest.py`: `ctx`, `clock`, `artifact_store`, `platform`, `market`.

## Shared core interfaces

| Interface | Module | Use |
|---|---|---|
| `RunContext` | `core/context.py` | Environment, `purpose`, `run_id`, `correlation_id`, injected `clock` and `ids`, pinned `contract_version`, `evaluator_version` (`EVALUATOR_VERSION` = `1.0.0`), `image_digest`, `synthetic`, `model_version`; `log(event, **fields)` writes one JSON line with the correlation ID; `error_envelope(exc)`; `RunContext.for_tests(seed=...)` |
| `Clock`, `SystemClock`, `FrozenClock`, `utc_iso`, `parse_utc` | `core/clock.py` | Never call `datetime.now()` directly |
| `IdMinter` (`run_id`, `model_version`, `artifact_id`, `correlation_id`), `configuration_id`, `request_hash`, `require_id` | `core/ids.py` | `run_` / `mv_` + ULID; `configuration_id` and `request_hash` come from the contract package |
| `FinplanError`, `ErrorCode`, `as_finplan_error` | `core/errors.py` | Raise with a registered contract code; `to_envelope(correlation_id)` validates the envelope; factories `validation(pointer=...)`, `not_permitted`, `precondition(reason=...)`, `internal`, `dependency_unavailable` |
| `ArtifactRef`, `ArtifactStore` (`InMemoryArtifactStore`, `LocalArtifactStore`, `S3ArtifactStore`) | `core/artifacts.py` | `put`/`put_json` return trusted references (content-addressed IDs, `owner` `financemodel`); `get` re-verifies the checksum; S3 keys and bucket names never leave the store |
| `PlatformClient`, `SnapshotReader`, `FixturePlatformClient`, `HttpPlatformClient`, `build_synthetic_snapshot` | `core/platform.py` | `SnapshotReader(client).resolve(id)` refuses non-approved snapshots (`PRECONDITION_FAILED` `snapshot_not_approved`); `.load(id)` downloads manifest and payload through grants and verifies every SHA-256 (`snapshot_checksum_mismatch`); `.read_observations(...)` pages the observations route; `get_staged_output(run_id)` on the client |
| `EnvConfig`, `load_config`, `load_all`, `load_shared_config`, `is_gpu_instance_type` | `core/config.py` | Job types (runtime ceilings, instance types, budget category, deployed flag), lease, queue, approval, cost, idempotency, retention, simulation defaults, and every SSM name (`cfg.ssm[...]`) |
| `require_valid`, `failed_job_result` | `core/outcome.py` | Contract validation with contract codes; the `job-result` document of a failed run |
| `SimulationConfig`, `MarketData`, `Bar`, `PointInTimeView`, `HoldingsView`, `Strategy`, `TargetWeights`, `Simulator`, `synthetic_market` | `sim/` | See [simulator.md](simulator.md) |
| `evaluate`, `EvaluationResult`, `EvaluatorIdentity`, `assert_comparable`, `replay` | `evaluate/` | The only entry point that produces results |
| `ModelStack`, `StageContext`, `resource_name`, `model_role_name`, `research_boundary`, `tag_role`, `role_arn_pattern`, `stack_name` | `infra/stacks/common.py` | Base stack applies tags and the environment permission boundary to every role |

## CDK app hook contract

`infra/app.py` builds one `Stage` per environment (`-c envs=beta,gamma` selects) and calls, in
order, `add_to_stage(stage, ctx)` of each existing module in `ENV_MODULES`
(`infra.stacks.storage`, `tables`, `roles`, `jobs`, `control`, `registry`, `release`), then
`add_to_app(app, shared, stages)` of each existing module in `APP_MODULES`
(`infra.stacks.tooling`, `pipeline`). A missing module is skipped. While no module adds a stack
to an environment, that environment gets a resource-free placeholder stack
(`finplan-<env>-financemodel-skeleton`, only `AWS::CDK::Metadata`), so the skeleton synthesizes
offline with `npx aws-cdk@2 synth` (exit 0), passes the post-synth gates and `uvx cfn-lint` with
no errors. The placeholder disappears as soon as any module contributes a stack to the stage.
