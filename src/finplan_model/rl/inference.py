"""Dispatch frozen actors to their versioned, checksum-preserved implementation."""
from .legacy_v1.inference import actor_action, export_actor, portfolio_state, allocation_response, recommend_baseline, FORMAT
from .legacy_v1 import inference as legacy
from . import inference_v2 as enriched

ENRICHED = frozenset(("cross_asset_risk", "market_regime", "recent_returns"))

def implementation_for(bundle=None):
    features = (bundle or {}).get("environment", {}).get("features", [])
    return enriched if ENRICHED.intersection(features) else legacy

def implementation_identity(bundle=None):
    return implementation_for(bundle).implementation_identity()

def recommend(bundle, prices, holdings):
    return implementation_for(bundle).recommend(bundle, prices, holdings)
