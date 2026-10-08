"""Research datasets: deterministic, content-addressed, point-in-time datasets prepared from
approved platform snapshots (spec research-datasets; task group 3). See ``docs/datasets.md``."""

from .calendar import SessionList, add_months, fixture_session_list, parse_calendar_version
from .catalog import DatasetCatalog, InMemoryDatasetCatalog, LocalDatasetCatalog, S3DatasetCatalog
from .config import FeatureSpec, Length, PreparationConfig
from .dataset import Dataset
from .features import DatasetBar, compute_features
from .fixtures import MockSnapshotProvider, fixture_calendar, synthetic_etf_observations, synthetic_etf_snapshot
from .holdout import FrozenCandidate, HoldoutAccessLog, HoldoutAccessor, InMemoryHoldoutAccessLog, check_prospective_snapshots, holdout_access_report
from .prepare import PreparedDataset, dataset_key, prepare_dataset
from .splits import Fold, ResolvedRange, ResolvedSplits, resolve_splits, walk_forward_folds

__all__ = [
    "Dataset",
    "DatasetBar",
    "DatasetCatalog",
    "FeatureSpec",
    "Fold",
    "FrozenCandidate",
    "HoldoutAccessLog",
    "HoldoutAccessor",
    "InMemoryDatasetCatalog",
    "InMemoryHoldoutAccessLog",
    "Length",
    "LocalDatasetCatalog",
    "MockSnapshotProvider",
    "PreparationConfig",
    "PreparedDataset",
    "ResolvedRange",
    "ResolvedSplits",
    "S3DatasetCatalog",
    "SessionList",
    "add_months",
    "check_prospective_snapshots",
    "compute_features",
    "dataset_key",
    "fixture_calendar",
    "fixture_session_list",
    "holdout_access_report",
    "parse_calendar_version",
    "prepare_dataset",
    "resolve_splits",
    "synthetic_etf_observations",
    "synthetic_etf_snapshot",
    "walk_forward_folds",
]
