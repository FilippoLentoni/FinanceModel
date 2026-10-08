"""Build-stage provider guard and synthetic-fixture check (DS-12, DS-13; task 3.7).

FinanceModel obtains market data only by reading approved platform snapshots. The platform's
ingestion owns the ``yfinance`` adapter; FinanceModel never calls a market-data provider and never
depends on a provider client. The public repository never contains retrieved market data.

:func:`provider_problems` fails the build when

* a market-data provider client (``yfinance`` or any package in :data:`MARKET_DATA_PACKAGES`)
  appears in ``pyproject.toml`` (any dependency group), ``uv.lock``, a ``requirements*`` file or a
  ``Dockerfile*`` (images);
* non-test Python source imports one of them; or
* source, image or configuration files reference a provider endpoint (:data:`PROVIDER_ENDPOINTS`).

:func:`fixture_problems` fails the build when a data file that may hold market data lacks the
``synthetic: true`` marker:

* every data file under ``tests/``, or under a directory named ``fixtures`` or ``data``: JSON must
  carry ``"synthetic": true`` at the top level (for a JSON array, on every object), CSV/TSV files
  must start with the line ``# synthetic: true``, and binary data formats (parquet, feather,
  pickle, Excel, HDF5, NumPy) are refused there because they cannot carry the marker;
* anywhere in the repository, a JSON or CSV file that looks like a price series (date plus
  close/open/adjusted-close columns or fields) must carry the same marker.

Both are wired into ``scripts/build_gates.py`` as the ``provider-guard`` and ``fixture-check``
gates. Test code is scanned for fixtures but not for imports (guard tests plant violations in
temporary directories only).
"""

from __future__ import annotations

import csv
import io
import json
import re
import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

__all__ = ["MARKET_DATA_PACKAGES", "PROVIDER_ENDPOINTS", "fixture_problems", "provider_problems"]

#: Market-data provider clients FinanceModel must never depend on or import (normalized names).
MARKET_DATA_PACKAGES = frozenset(
    {
        "yfinance",
        "yahoo-finance",
        "yahooquery",
        "yahoo-fin",
        "yfinance-cache",
        "pandas-datareader",
        "alpha-vantage",
        "alpha-vantage-api",
        "polygon-api-client",
        "polygon",
        "finnhub-python",
        "tiingo",
        "quandl",
        "nasdaq-data-link",
        "iexfinance",
        "twelvedata",
        "eodhd",
        "eod",
        "openbb",
        "investpy",
        "tvdatafeed",
        "stooq",
        "fredapi",
        "intrinio-sdk",
        "databento",
    }
)
#: Import names that differ from distribution names.
_IMPORT_ALIASES = {"pandas_datareader": "pandas-datareader", "alpha_vantage": "alpha-vantage", "yahoo_fin": "yahoo-fin", "polygon": "polygon", "finnhub": "finnhub-python", "nasdaqdatalink": "nasdaq-data-link", "intrinio_sdk": "intrinio-sdk"}
#: Provider endpoint host patterns (regular expressions, so this file never matches itself).
PROVIDER_ENDPOINTS = (
    r"query[12]\.finance\.yahoo\.com",
    r"(?<![a-z])finance\.yahoo\.com",
    r"(?<![a-z])fc\.yahoo\.com",
    r"(?<![a-z])stooq\.(?:com|pl)",
    r"(?<![a-z])www\.alphavantage\.co",
    r"(?<![a-z])api\.polygon\.io",
    r"(?<![a-z])finnhub\.io",
    r"(?<![a-z])api\.tiingo\.com",
    r"(?<![a-z])cloud\.iexapis\.com",
    r"(?<![a-z])api\.twelvedata\.com",
    r"(?<![a-z])eodhd\.com",
    r"(?<![a-z])data\.nasdaq\.com",
)
_ENDPOINT_RE = re.compile("|".join(PROVIDER_ENDPOINTS), re.IGNORECASE)
_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_DYNAMIC_IMPORT_RE = re.compile(r"(?:__import__|import_module)\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)")
DEFAULT_EXCLUDES = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".ruff_cache", "cdk.out", "cdk.out.bootstrap", "build-output", ".build", "vendor", "openspec", ".claude"})
_DATA_SUFFIXES = {".json", ".jsonl", ".ndjson", ".csv", ".tsv"}
_BINARY_DATA_SUFFIXES = {".parquet", ".feather", ".arrow", ".pkl", ".pickle", ".xlsx", ".xls", ".h5", ".hdf5", ".npz", ".npy"}
_ENDPOINT_SCAN_SUFFIXES = {".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".txt", ".env"}
_PRICE_COLS = {"close", "adj close", "adj_close", "adjclose", "open", "high", "low"}
_DATE_COLS = {"date", "session_date", "datetime", "timestamp", "time"}
#: Repository JSON that is configuration or tooling, never market data.
_NON_DATA_NAMES = {"package.json", "package-lock.json", "tsconfig.json", "cdk.json", "contracts-pin.json", "cdk.context.json", "manifest.json", "tree.json"}


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


_FORBIDDEN = {_norm(p) for p in MARKET_DATA_PACKAGES}


def _files(root: Path, excludes: Iterable[str]) -> Iterable[Path]:
    ex = set(excludes)
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).parts
        if any(part in ex for part in rel):
            continue
        if path.is_file():
            yield path


def _dep_name(spec: str) -> str:
    return _norm(re.split(r"[\s\[<>=!~;@]", spec.strip(), maxsplit=1)[0])


def provider_problems(root: Path, *, excludes: Iterable[str] = DEFAULT_EXCLUDES) -> list[str]:
    root = Path(root)
    problems: list[str] = []
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        deps = list(data.get("project", {}).get("dependencies", []))
        for group in data.get("project", {}).get("optional-dependencies", {}).values():
            deps += list(group)
        for group in data.get("dependency-groups", {}).values():
            deps += [d for d in group if isinstance(d, str)]
        for dep in deps:
            if _dep_name(dep) in _FORBIDDEN:
                problems.append(f"pyproject.toml declares market-data provider client {_dep_name(dep)} (FinanceModel reads approved platform snapshots only)")
    lock = root / "uv.lock"
    if lock.is_file():
        for pkg in tomllib.loads(lock.read_text(encoding="utf-8")).get("package", []):
            if _norm(str(pkg.get("name", ""))) in _FORBIDDEN:
                problems.append(f"uv.lock resolves market-data provider client {pkg['name']}")
    for path in _files(root, excludes):
        rel = path.relative_to(root)
        name = path.name
        text: str | None = None
        if name.startswith("Dockerfile") or name.startswith("requirements") or name.endswith(".in") and "requirements" in name:
            text = path.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.strip().startswith("#"):
                    continue
                for tok in re.findall(r"[A-Za-z0-9_.-]+", line):
                    if _norm(tok) in _FORBIDDEN:
                        problems.append(f"{rel} installs market-data provider client {tok}")
        if path.suffix == ".py" and "tests" not in rel.parts:
            text = text or path.read_text(encoding="utf-8", errors="replace")
            for mod in _IMPORT_RE.findall(text) + _DYNAMIC_IMPORT_RE.findall(text):
                dist = _IMPORT_ALIASES.get(mod, mod)
                if _norm(dist) in _FORBIDDEN:
                    problems.append(f"{rel} imports market-data provider client {mod}")
        if (path.suffix in _ENDPOINT_SCAN_SUFFIXES or name.startswith("Dockerfile")) and "tests" not in rel.parts and path.resolve() != Path(__file__).resolve():
            text = text or path.read_text(encoding="utf-8", errors="replace")
            m = _ENDPOINT_RE.search(text)
            if m:
                problems.append(f"{rel} references market-data provider endpoint {m.group(0)}")
    return problems


def _in_fixture_scope(rel: Path) -> bool:
    return rel.parts[0] == "tests" or any(p in ("fixtures", "data") for p in rel.parts[:-1])


def _json_synthetic(doc: Any) -> bool:
    if isinstance(doc, dict):
        return doc.get("synthetic") is True
    if isinstance(doc, list):
        return bool(doc) and all(isinstance(x, dict) and x.get("synthetic") is True for x in doc)
    return False


def _json_looks_like_prices(doc: Any) -> bool:
    rows: list[Any] = []
    stack = [doc]
    while stack and len(rows) < 10_000:
        node = stack.pop()
        if isinstance(node, dict):
            keys = {str(k).lower() for k in node}
            if keys & _PRICE_COLS and keys & _DATE_COLS:
                rows.append(node)
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return len(rows) >= 5


def _csv_header(text: str) -> list[str]:
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            delim = "\t" if line.count("\t") > line.count(",") else ","
            return [c.strip().lower() for c in next(csv.reader(io.StringIO(line), delimiter=delim))]
    return []


def fixture_problems(root: Path, *, excludes: Iterable[str] = DEFAULT_EXCLUDES) -> list[str]:
    root = Path(root)
    problems: list[str] = []
    for path in _files(root, excludes):
        rel = path.relative_to(root)
        suffix = path.suffix.lower()
        in_scope = _in_fixture_scope(rel)
        if suffix in _BINARY_DATA_SUFFIXES:
            if in_scope:
                problems.append(f"{rel}: binary data files cannot carry the synthetic: true marker; fixtures must be synthetic JSON or CSV")
            continue
        if suffix not in _DATA_SUFFIXES or path.name in _NON_DATA_NAMES:
            continue
        if rel.parts[0] == "config" and suffix == ".json":
            continue  # environment configuration, validated by the config gate
        text = path.read_text(encoding="utf-8", errors="replace")
        if suffix in (".csv", ".tsv"):
            marked = text.lstrip().lower().startswith("# synthetic: true")
            header = set(_csv_header(text))
            if not marked and (in_scope or (header & _PRICE_COLS and header & _DATE_COLS)):
                problems.append(f"{rel}: data file without the '# synthetic: true' marker (retrieved market data must never be committed)")
            continue
        try:
            if suffix in (".jsonl", ".ndjson"):
                docs = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
                marked = bool(docs) and all(_json_synthetic(d) for d in docs)
                looks = any(_json_looks_like_prices(d) for d in docs) or len(docs) >= 5 and _json_looks_like_prices(docs)
            else:
                doc = json.loads(text)
                marked = _json_synthetic(doc)
                looks = _json_looks_like_prices(doc)
        except json.JSONDecodeError:
            if in_scope:
                problems.append(f"{rel}: fixture is not valid JSON, so its synthetic marker cannot be checked")
            continue
        if not marked and (in_scope or looks):
            problems.append(f"{rel}: data file without the synthetic: true marker (retrieved market data must never be committed)")
    return problems
