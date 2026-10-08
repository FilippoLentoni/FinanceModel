"""Model registry (task group 9): ``model_version`` minting, immutable records, lifecycle events,
run lineage and the platform's lineage lookup. See ``docs/staging-and-registry.md``."""

from .registry import LIFECYCLE, STATUSES, InMemoryRegistryStore, ModelIdentity, ModelRegistry, RegistryStore, S3RegistryStore
from .resolver import lineage_result_hook, model_version_resolver, record_spec_lineage, seed_baselines

__all__ = [
    "LIFECYCLE",
    "STATUSES",
    "InMemoryRegistryStore",
    "ModelIdentity",
    "ModelRegistry",
    "RegistryStore",
    "S3RegistryStore",
    "lineage_result_hook",
    "model_version_resolver",
    "record_spec_lineage",
    "seed_baselines",
]
