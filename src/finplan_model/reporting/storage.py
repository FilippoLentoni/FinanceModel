"""Where results and reports may be written (DS-13 report scenario; task 3.9).

Retrieved market data (Yahoo Finance via the platform's yfinance ingestion) is for personal and
research use, and the FinanceModel repository is public. Therefore:

* run results, benchmark reports and holdout records are written only to **research storage**
  (an :class:`~finplan_model.core.artifacts.ArtifactStore`: S3 research storage in AWS, an
  in-memory store in tests, a local directory outside any source repository for offline runs) or,
  for production candidates, to the platform staging area (run-output staging);
* a store rooted inside a source repository working tree is refused with
  ``OPERATION_NOT_PERMITTED`` for anything derived from real (non-synthetic) data;
* a benchmark report carries **aggregate metrics only**: :func:`assert_aggregate_only` refuses a
  report that embeds a series (a list of more than :data:`MAX_NUMERIC_LIST` numbers) or a raw-data
  field (``nav``, ``prices``, ``bars``, ``observations``, ``fills`` ...), whatever the data source.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from finplan_model.core.artifacts import ArtifactRef, ArtifactStore, LocalArtifactStore
from finplan_model.core.errors import FinplanError

__all__ = ["MAX_NUMERIC_LIST", "RUN_RESULT_KIND", "assert_aggregate_only", "inside_source_repository", "require_research_storage", "store_research_artifact"]

RUN_RESULT_KIND = "run_artifact"
MAX_NUMERIC_LIST = 8
SERIES_KEYS = frozenset({"nav", "navs", "prices", "price_series", "bars", "observations", "fills", "closes", "close_series", "volumes", "ohlcv", "series", "returns_series"})
_REPO_MARKERS = (".git", "openspec")


def inside_source_repository(path: str | Path) -> bool:
    """True when ``path`` is inside a source repository working tree (``.git`` or ``openspec`` next
    to a ``pyproject.toml`` in one of its parents)."""
    p = Path(path).resolve()
    for d in (p, *p.parents):
        if (d / ".git").exists() or ((d / "openspec").is_dir() and (d / "pyproject.toml").is_file()):
            return True
    return False


def require_research_storage(store: ArtifactStore, *, synthetic: bool) -> None:
    root = getattr(store, "root", None) if isinstance(store, LocalArtifactStore) else None
    if root is not None and not synthetic and inside_source_repository(root):
        raise FinplanError.not_permitted("results and reports derived from real market data are written only to research storage or platform staging, never into a source repository", reason="real_data_outside_research_storage")


def assert_aggregate_only(doc: Any, path: str = "") -> None:
    if isinstance(doc, Mapping):
        for k, v in doc.items():
            if isinstance(k, str) and k.lower() in SERIES_KEYS:
                raise FinplanError.validation("reports carry aggregate metrics only, never raw or per-session series", pointer=f"{path}/{k}", reason="report_embeds_series")
            assert_aggregate_only(v, f"{path}/{k}")
    elif isinstance(doc, (list, tuple)):
        if sum(1 for v in doc if isinstance(v, (int, float)) and not isinstance(v, bool)) > MAX_NUMERIC_LIST:
            raise FinplanError.validation("reports carry aggregate metrics only, never raw or per-session series", pointer=path, reason="report_embeds_series")
        for i, v in enumerate(doc):
            assert_aggregate_only(v, f"{path}/{i}")


def store_research_artifact(doc: Mapping[str, Any], store: ArtifactStore, *, kind: str, synthetic: bool) -> ArtifactRef:
    require_research_storage(store, synthetic=synthetic)
    return store.put_json(dict(doc), kind=kind, synthetic=True if synthetic else None, domain="finance")  # type: ignore[attr-defined]
