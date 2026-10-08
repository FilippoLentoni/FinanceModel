"""Run-output staging (task group 9): production-candidate bundles for the platform staging area,
manifest last, write-once; platform outcome linking. See ``docs/staging-and-registry.md``."""

from .bundle import (
    COMMITTABLE_SOLUTIONS,
    MANIFEST_NAME,
    METRICS_FILE,
    PLAN_CONTENT_FILE,
    InMemoryStagingStore,
    S3StagingStore,
    StagedBundle,
    StagingStore,
    StagingTarget,
    build_bundle,
    parse_staging_ref,
    plan_content_from_result,
    proposed_allocation,
    stage_run_output,
    staging_decision,
)
from .outcome import link_platform_outcome, platform_outcome_hook, staging_view

__all__ = [
    "COMMITTABLE_SOLUTIONS",
    "MANIFEST_NAME",
    "METRICS_FILE",
    "PLAN_CONTENT_FILE",
    "InMemoryStagingStore",
    "S3StagingStore",
    "StagedBundle",
    "StagingStore",
    "StagingTarget",
    "build_bundle",
    "link_platform_outcome",
    "parse_staging_ref",
    "plan_content_from_result",
    "platform_outcome_hook",
    "proposed_allocation",
    "stage_run_output",
    "staging_decision",
    "staging_view",
]
