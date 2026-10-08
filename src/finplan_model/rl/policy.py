"""A trained RL policy as a common-evaluator strategy (spec rl-strategies, "Evaluator parity").

:class:`RLPolicyStrategy` implements the simulator's strategy interface: at each rebalance decision it
builds the declared observation from the point-in-time view only (the last ``window + 1`` visible
closes and the current holdings), asks the policy for a **deterministic** action
(``deterministic=True``) and maps it to target weights with the declared softmax transform. Without
``window + 1`` visible closes it holds (``no_effect``), as the optimizers do while they lack history.
Its portfolio metrics therefore come from the same evaluator version and simulation configuration as
the controls and optimizers in the same comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from finplan_model.sim.market import HoldingsView, PointInTimeView
from finplan_model.sim.strategy import TargetWeights

from .spec import EnvSpec, build_observation, softmax_weights

__all__ = ["RLPolicyStrategy"]


class RLPolicyStrategy:
    family = "reinforcement_learning"
    makes_predictions = False
    has_randomness = False  # deterministic evaluation; training randomness is recorded as the seed

    def __init__(self, name: str, predict: Callable[[np.ndarray], Any], spec: EnvSpec, *, seed: int, configuration_id: str, details: Mapping[str, Any] | None = None) -> None:
        self.name = name
        self._predict = predict
        self.spec = spec
        self.seed = int(seed)
        self.configuration_id = configuration_id
        self.details = dict(details or {})

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights:
        instruments = view.instruments
        _, px = view.price_matrix("close", lookback=self.spec.window + 1)
        w_now = holdings.weights
        if px.shape[0] < self.spec.window + 1:
            return TargetWeights({i: float(w_now.get(i, 0.0)) for i in instruments}, float(holdings.cash_weight), "no_effect", {"reason": "insufficient_history", "observations": int(px.shape[0]), "window": self.spec.window})
        cur = np.array([float(w_now.get(i, 0.0)) for i in instruments])
        obs = build_observation(self.spec, px, cur, float(holdings.cash_weight))
        weights, cash = softmax_weights(self._predict(obs), len(instruments), self.spec.action_scale)
        return TargetWeights({i: float(weights[k]) for k, i in enumerate(instruments)}, cash, "feasible", {"policy_seed": self.seed, "configuration_id": self.configuration_id, "transform": "softmax"})
