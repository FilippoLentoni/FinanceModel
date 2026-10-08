"""Benchmark runs and reports (spec benchmark-reporting; task group 5). See ``docs/benchmarks.md``."""

from .benchmark import BenchmarkRequest, BenchmarkRun, ResultEntry, StrategySpec, run_benchmark
from .holdout_record import COMPARABILITY_FIELDS, build_holdout_record, store_holdout_record, validate_holdout_record
from .report import PERIOD_TYPES, PORTFOLIO_METRICS, BenchmarkReport, ReportConfig, aggregate_fold_metrics, build_report, report_from_stored, store_report, to_run_result_payload
from .storage import assert_aggregate_only, inside_source_repository, require_research_storage, store_research_artifact

__all__ = [
    "BenchmarkReport",
    "BenchmarkRequest",
    "BenchmarkRun",
    "COMPARABILITY_FIELDS",
    "PERIOD_TYPES",
    "PORTFOLIO_METRICS",
    "ReportConfig",
    "ResultEntry",
    "StrategySpec",
    "aggregate_fold_metrics",
    "assert_aggregate_only",
    "build_holdout_record",
    "build_report",
    "inside_source_repository",
    "report_from_stored",
    "require_research_storage",
    "run_benchmark",
    "store_holdout_record",
    "store_report",
    "store_research_artifact",
    "to_run_result_payload",
    "validate_holdout_record",
]
