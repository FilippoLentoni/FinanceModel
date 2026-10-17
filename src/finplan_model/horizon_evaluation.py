"""Dispatch horizon replay by its recorded implementation identity."""
from . import legacy_horizon_evaluation as legacy
from . import horizon_evaluation_v2 as enriched
from .rl.inference import implementation_for, legacy as legacy_inference
from .horizon_evaluation_v2 import evaluation_contract, _Control, _FrozenStrategy, VERSION, MAX_SESSIONS

def evaluation_implementation(bundle=None):
    module = legacy if implementation_for(bundle) is legacy_inference else enriched
    return module.evaluation_implementation()

def evaluate_horizons(service, doc, market, raw, content, end):
    recorded = doc.get("provenance", {}).get("evaluation_implementation")
    module = legacy if recorded in (None, legacy.evaluation_implementation()) else enriched
    return module.evaluate_horizons(service, doc, market, raw, content, end)
