"""Dataset preparation configuration (spec research-datasets; design D6).

A :class:`PreparationConfig` holds everything that can change a prepared dataset: the instrument
definition (the ETF ticker is part of it, so it is part of the ``configuration_id``), the data
source, the point-in-time availability rule, the decision time, the feature pipeline and its
horizons, the chronological splits and embargo, the walk-forward fold generator, the quality-flag
policy and the revision policy. Its canonical form gives the preparation ``configuration_id``
(contract RFC 8785 canonicalization, ``cfg_`` + SHA-256).

Validation errors are ``VALIDATION_FAILED`` with a JSON pointer. There is no shuffle option:
evaluation splits are chronological only (an unknown key such as ``shuffle`` is refused).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, time
from typing import Any

from finplan_model.core.errors import FinplanError
from finplan_model.core.ids import configuration_id

__all__ = [
    "DATA_SOURCES",
    "FeatureSpec",
    "Length",
    "PREPARATION_VERSION",
    "PreparationConfig",
    "SplitRange",
    "WalkForwardConfig",
]

#: Version of the preparation code's output format; part of the dataset identity.
PREPARATION_VERSION = "dataset-prep-v1"
DATA_SOURCES = ("fixture", "platform_snapshots")
AVAILABILITY_MODES = ("retrieved_at", "session_close")
QUALITY_POLICIES = ("exclude", "fail")
REVISION_POLICIES = ("first_seen", "fail")
FEATURE_KINDS = ("trailing_return", "trailing_volatility", "close")
ALIGNMENTS = ("asof", "session")
#: Platform quality flags that mark dates whose provider response was empty or partial (or stale).
DEFAULT_QUALITY_FLAGS = ("empty_response", "partial_response", "missing_sessions", "stale_source")
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{0,31}\Z")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}\Z")
_KIND_RE = re.compile(r"^[a-z0-9-]{1,40}\Z")


def _ptr(*parts: Any) -> str:
    return "".join("/" + str(p) for p in parts)


def _no_unknown(d: Mapping[str, Any], allowed: set[str], *ptr: Any) -> None:
    for k in d:
        if k != "$comment" and k not in allowed:
            raise FinplanError.validation(f"unknown preparation setting {k}", pointer=_ptr(*ptr, k))


def _int(d: Mapping[str, Any], key: str, *ptr: Any, lo: int = 0, default: int | None = None) -> int:
    if key not in d:
        if default is None:
            raise FinplanError.validation(f"missing preparation setting {key}", pointer=_ptr(*ptr, key))
        return default
    v = d[key]
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise FinplanError.validation(f"{key} must be an integer >= {lo}", pointer=_ptr(*ptr, key))
    return v


def _choice(d: Mapping[str, Any], key: str, choices: tuple[str, ...], default: str, *ptr: Any) -> str:
    v = d.get(key, default)
    if v not in choices:
        raise FinplanError.validation(f"{key} must be one of {', '.join(choices)}", pointer=_ptr(*ptr, key))
    return str(v)


def _date(v: Any, pointer: str) -> date:
    try:
        return date.fromisoformat(str(v))
    except ValueError:
        raise FinplanError.validation("not an ISO date", pointer=pointer) from None


def _hhmm(v: Any, pointer: str) -> time:
    if not isinstance(v, str) or not re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", v):
        raise FinplanError.validation("time must be HH:MM (UTC)", pointer=pointer)
    try:
        return time.fromisoformat(v)
    except ValueError:
        raise FinplanError.validation("time must be HH:MM (UTC)", pointer=pointer) from None


@dataclass(frozen=True)
class Length:
    """A window length in sessions or calendar months (years are stored as 12 months)."""

    unit: str  # sessions | months
    value: int

    @classmethod
    def parse(cls, v: Any, pointer: str) -> "Length":
        if not isinstance(v, Mapping) or len(v) != 1:
            raise FinplanError.validation("a length is one of {sessions: n}, {months: n} or {years: n}", pointer=pointer)
        (unit, n), = v.items()
        if unit not in ("sessions", "months", "years") or isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise FinplanError.validation("a length is one of {sessions: n}, {months: n} or {years: n} with n >= 1", pointer=pointer)
        return cls("sessions", n) if unit == "sessions" else cls("months", n * (12 if unit == "years" else 1))

    @property
    def calendar(self) -> bool:
        return self.unit == "months"

    def to_dict(self) -> dict[str, int]:
        return {self.unit: self.value}


@dataclass(frozen=True)
class SplitRange:
    start: date
    end: date

    def to_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True)
class WalkForwardConfig:
    window: str  # expanding | rolling
    train_length: Length
    test_length: Length
    step: Length
    embargo_sessions: int
    min_folds: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {"window": self.window, "train_length": self.train_length.to_dict(), "test_length": self.test_length.to_dict(), "step": self.step.to_dict(), "embargo_sessions": self.embargo_sessions, "min_folds": self.min_folds}


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: str
    lookback: int = 1
    alignment: str = "asof"
    #: session offset of the referenced bar relative to the decision session (alignment ``session``)
    offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "lookback": self.lookback, "alignment": self.alignment, "offset": self.offset}


@dataclass(frozen=True)
class PreparationConfig:
    tickers: tuple[str, ...] = ("SPY",)
    dataset_kind: str = "etf-daily"
    asset_class: str = "etf"
    calendar: str = "XNYS"
    currency: str = "USD"
    granularity: str = "daily"
    data_source: str = "fixture"
    availability_mode: str = "retrieved_at"
    assumed_close_utc: time = time(21, 0)
    publication_lag_minutes: int = 30
    decision_time_utc: time = time(22, 0)
    lookback_sessions: int = 60
    label_horizon_sessions: int = 20
    feature_horizon_sessions: int = 0
    features: tuple[FeatureSpec, ...] = ()
    splits: dict[str, SplitRange] | None = None
    split_embargo_sessions: int = 20
    walk_forward: WalkForwardConfig | None = None
    quality_policy: str = "exclude"
    quality_flags: tuple[str, ...] = DEFAULT_QUALITY_FLAGS
    revisions: str = "first_seen"
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ parsing
    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PreparationConfig":
        d = dict(d)
        _no_unknown(d, {"instrument", "data_source", "availability", "decision_time_utc", "features", "splits", "walk_forward", "quality", "revisions"})
        inst = dict(d.get("instrument") or {})
        _no_unknown(inst, {"tickers", "ticker", "dataset_kind", "asset_class", "calendar", "currency", "granularity"}, "instrument")
        tickers_raw = inst.get("tickers", [inst["ticker"]] if "ticker" in inst else ["SPY"])
        if not isinstance(tickers_raw, list) or not tickers_raw or any(not isinstance(t, str) or not _TICKER_RE.match(t) for t in tickers_raw) or len(set(tickers_raw)) != len(tickers_raw):
            raise FinplanError.validation("instrument.tickers must be a non-empty list of distinct instrument ids", pointer="/instrument/tickers")
        dataset_kind = str(inst.get("dataset_kind", "etf-daily"))
        if not _KIND_RE.match(dataset_kind):
            raise FinplanError.validation("instrument.dataset_kind is not a dataset kind", pointer="/instrument/dataset_kind")
        if dataset_kind != "etf-daily" or inst.get("asset_class", "etf") != "etf":
            # Initial instrument definition: S&P 500 exposure through a tracking ETF daily series.
            # The index level and the constituent universe are different datasets (DS-09).
            raise FinplanError.validation("phase 1 datasets are the ETF daily series (dataset_kind etf-daily, asset_class etf); the index level and constituent universe are distinct datasets", pointer="/instrument/dataset_kind")
        if inst.get("granularity", "daily") != "daily":
            raise FinplanError.validation("only completed daily observations are used in phases 1 and 2", pointer="/instrument/granularity")
        cal = str(inst.get("calendar", "XNYS"))
        if not re.fullmatch(r"[A-Z]{4}", cal):
            raise FinplanError.validation("instrument.calendar must be an exchange MIC", pointer="/instrument/calendar")
        currency = str(inst.get("currency", "USD"))
        if not re.fullmatch(r"[A-Z]{3}", currency):
            raise FinplanError.validation("instrument.currency must be an ISO currency code", pointer="/instrument/currency")

        av = dict(d.get("availability") or {})
        _no_unknown(av, {"mode", "assumed_close_utc", "publication_lag_minutes"}, "availability")
        feats = dict(d.get("features") or {})
        _no_unknown(feats, {"lookback_sessions", "label_horizon_sessions", "feature_horizon_sessions", "pipeline"}, "features")
        lookback = _int(feats, "lookback_sessions", "features", lo=1, default=60)
        label_h = _int(feats, "label_horizon_sessions", "features", lo=0, default=20)
        feat_h = _int(feats, "feature_horizon_sessions", "features", lo=0, default=0)
        pipeline_raw = feats.get("pipeline")
        if pipeline_raw is None:
            pipeline = (FeatureSpec(f"trailing_return_{lookback}", "trailing_return", lookback), FeatureSpec(f"trailing_volatility_{lookback}", "trailing_volatility", lookback))
        else:
            if not isinstance(pipeline_raw, list):
                raise FinplanError.validation("features.pipeline must be a list", pointer="/features/pipeline")
            specs = []
            for n, f in enumerate(pipeline_raw):
                p = ("features", "pipeline", n)
                if not isinstance(f, Mapping):
                    raise FinplanError.validation("a feature is an object", pointer=_ptr(*p))
                _no_unknown(f, {"name", "kind", "lookback", "alignment", "offset"}, *p)
                name = f.get("name")
                if not isinstance(name, str) or not _NAME_RE.match(name):
                    raise FinplanError.validation("feature name must be snake case", pointer=_ptr(*p, "name"))
                kind = _choice(f, "kind", FEATURE_KINDS, "", *p)
                off = f.get("offset", 0)
                if isinstance(off, bool) or not isinstance(off, int):
                    raise FinplanError.validation("offset must be an integer", pointer=_ptr(*p, "offset"))
                specs.append(FeatureSpec(name, kind, _int(f, "lookback", *p, lo=1, default=1), _choice(f, "alignment", ALIGNMENTS, "asof", *p), off))
            if len({s.name for s in specs}) != len(specs):
                raise FinplanError.validation("feature names must be unique", pointer="/features/pipeline")
            pipeline = tuple(specs)

        sp = d.get("splits")
        splits: dict[str, SplitRange] | None = None
        split_embargo = 0
        if sp is not None:
            if not isinstance(sp, Mapping):
                raise FinplanError.validation("splits must be an object", pointer="/splits")
            _no_unknown(sp, {"train", "validation", "holdout", "embargo_sessions"}, "splits")
            splits = {}
            for name in ("train", "validation", "holdout"):
                r = sp.get(name)
                if not isinstance(r, Mapping) or set(r) - {"start", "end", "$comment"} or "start" not in r or "end" not in r:
                    raise FinplanError.validation(f"splits.{name} must be {{start, end}}", pointer=f"/splits/{name}")
                rng = SplitRange(_date(r["start"], f"/splits/{name}/start"), _date(r["end"], f"/splits/{name}/end"))
                if rng.end < rng.start:
                    raise FinplanError.validation(f"splits.{name} ends before it starts", pointer=f"/splits/{name}/end")
                splits[name] = rng
            # Chronological, non-overlapping, in the order train < validation < holdout (DS-04).
            if splits["validation"].start <= splits["train"].end:
                raise FinplanError.validation("validation range must start after the end of the training range", pointer="/splits/validation/start", train_end=splits["train"].end.isoformat(), validation_start=splits["validation"].start.isoformat())
            if splits["holdout"].start <= splits["validation"].end:
                raise FinplanError.validation("holdout range must start after the end of the validation range", pointer="/splits/holdout/start", validation_end=splits["validation"].end.isoformat(), holdout_start=splits["holdout"].start.isoformat())
            split_embargo = _int(sp, "embargo_sessions", "splits", lo=0)
            required = max(label_h, feat_h)
            if split_embargo < required:
                raise FinplanError.validation("splits.embargo_sessions must be at least the longest label or feature horizon", pointer="/splits/embargo_sessions", required_sessions=required, configured_sessions=split_embargo)

        wf_raw = d.get("walk_forward")
        wf: WalkForwardConfig | None = None
        if wf_raw is not None:
            if not isinstance(wf_raw, Mapping):
                raise FinplanError.validation("walk_forward must be an object", pointer="/walk_forward")
            _no_unknown(wf_raw, {"window", "train_length", "test_length", "step", "embargo_sessions", "min_folds"}, "walk_forward")
            tr = Length.parse(wf_raw.get("train_length"), "/walk_forward/train_length")
            te = Length.parse(wf_raw.get("test_length"), "/walk_forward/test_length")
            st = Length.parse(wf_raw.get("step"), "/walk_forward/step")
            if tr.calendar != st.calendar:
                raise FinplanError.validation("walk_forward train_length and step must use the same unit (sessions, or months/years)", pointer="/walk_forward/step")
            emb = _int(wf_raw, "embargo_sessions", "walk_forward", lo=0)
            required = max(label_h, feat_h)
            if emb < required:
                raise FinplanError.validation("walk_forward.embargo_sessions must be at least the longest label or feature horizon", pointer="/walk_forward/embargo_sessions", required_sessions=required, configured_sessions=emb)
            wf = WalkForwardConfig(_choice(wf_raw, "window", ("expanding", "rolling"), "expanding", "walk_forward"), tr, te, st, emb, _int(wf_raw, "min_folds", "walk_forward", lo=1, default=1))

        q = dict(d.get("quality") or {})
        _no_unknown(q, {"policy", "flags"}, "quality")
        flags = q.get("flags", list(DEFAULT_QUALITY_FLAGS))
        if not isinstance(flags, list) or any(not isinstance(f, str) or not _NAME_RE.match(f) for f in flags):
            raise FinplanError.validation("quality.flags must be a list of quality flag names", pointer="/quality/flags")
        lag = _int(av, "publication_lag_minutes", "availability", lo=0, default=30)
        if lag > 24 * 60:
            raise FinplanError.validation("publication_lag_minutes must be at most one day", pointer="/availability/publication_lag_minutes")
        cfg = cls(
            tickers=tuple(tickers_raw),
            dataset_kind=dataset_kind,
            asset_class="etf",
            calendar=cal,
            currency=currency,
            granularity="daily",
            data_source=_choice(d, "data_source", DATA_SOURCES, "fixture"),
            availability_mode=_choice(av, "mode", AVAILABILITY_MODES, "retrieved_at", "availability"),
            assumed_close_utc=_hhmm(av.get("assumed_close_utc", "21:00"), "/availability/assumed_close_utc"),
            publication_lag_minutes=lag,
            decision_time_utc=_hhmm(d.get("decision_time_utc", "22:00"), "/decision_time_utc"),
            lookback_sessions=lookback,
            label_horizon_sessions=label_h,
            feature_horizon_sessions=feat_h,
            features=pipeline,
            splits=splits,
            split_embargo_sessions=split_embargo,
            walk_forward=wf,
            quality_policy=_choice(q, "policy", QUALITY_POLICIES, "exclude", "quality"),
            quality_flags=tuple(sorted(set(flags))),
            revisions=_choice(d, "revisions", REVISION_POLICIES, "first_seen"),
            raw=d,
        )
        return cfg

    @classmethod
    def for_environment(cls, instrument: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> "PreparationConfig":
        """Defaults from ``config/<env>.json`` ``instrument`` plus run overrides."""
        base: dict[str, Any] = {
            "instrument": {"tickers": [str(instrument.get("ticker", "SPY"))], "dataset_kind": str(instrument.get("dataset_kind", "etf-daily")), "calendar": str(instrument.get("calendar", "XNYS")), "currency": str(instrument.get("currency", "USD")), "granularity": str(instrument.get("granularity", "daily"))},
            "data_source": str(instrument.get("data_source", "fixture")),
        }
        for k, v in dict(overrides or {}).items():
            if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
        return cls.from_dict(base)

    # ------------------------------------------------------------------ identity
    def to_dict(self) -> dict[str, Any]:
        return {
            "preparation_version": PREPARATION_VERSION,
            "instrument": {"tickers": list(self.tickers), "dataset_kind": self.dataset_kind, "asset_class": self.asset_class, "calendar": self.calendar, "currency": self.currency, "granularity": self.granularity},
            "data_source": self.data_source,
            "availability": {"mode": self.availability_mode, "assumed_close_utc": self.assumed_close_utc.strftime("%H:%M"), "publication_lag_minutes": self.publication_lag_minutes},
            "decision_time_utc": self.decision_time_utc.strftime("%H:%M"),
            "features": {"lookback_sessions": self.lookback_sessions, "label_horizon_sessions": self.label_horizon_sessions, "feature_horizon_sessions": self.feature_horizon_sessions, "pipeline": [f.to_dict() for f in self.features]},
            "splits": None if self.splits is None else {**{k: v.to_dict() for k, v in self.splits.items()}, "embargo_sessions": self.split_embargo_sessions},
            "walk_forward": None if self.walk_forward is None else self.walk_forward.to_dict(),
            "quality": {"policy": self.quality_policy, "flags": list(self.quality_flags)},
            "revisions": self.revisions,
            "price_basis": "unadjusted",
        }

    @property
    def configuration_id(self) -> str:
        return configuration_id(self.to_dict())

    @property
    def required_embargo_sessions(self) -> int:
        return max(self.label_horizon_sessions, self.feature_horizon_sessions)

    def expected_dataset_ids(self) -> dict[str, str]:
        """``finance/<dataset_kind>/<ticker>`` per configured ticker (platform dataset identity)."""
        return {f"finance/{self.dataset_kind}/{t}": t for t in self.tickers}


def finite(x: float) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))
