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

from .spec import EnvSpec, allocation_action, build_observation

__all__ = ["RLPolicyStrategy", "EnsemblePolicyStrategy"]


class EnsemblePolicyStrategy:
    """Equal-weight portfolio targets from every seed; never average neural-network parameters."""
    family = "reinforcement_learning"
    makes_predictions = False
    has_randomness = False

    def __init__(self, name: str, members: list[RLPolicyStrategy]) -> None:
        if not members:
            raise ValueError("an ensemble needs at least one policy")
        self.name, self.members = name, members

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights:
        targets = [member.decide(view, holdings) for member in self.members]
        weights = {i: float(np.mean([t.weights.get(i, 0.0) for t in targets])) for i in view.instruments}
        cash = float(np.mean([t.cash for t in targets]))
        return TargetWeights(weights, cash, targets[0].solution_status, {"policy_seeds": [m.seed for m in self.members], "aggregation": "mean_target_weights"})


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
        self._peak = 0.0

    def decide(self, view: PointInTimeView, holdings: HoldingsView) -> TargetWeights:
        instruments = view.instruments
        self._peak = max(self._peak, holdings.value)
        drawdown = max(0.0, min(1.0, 1 - holdings.value / self._peak)) if self._peak > 0 else 0.0
        _, px = view.price_matrix("close", lookback=self.spec.window + 1)
        w_now = holdings.weights
        if px.shape[0] < self.spec.window + 1:
            return TargetWeights({i: float(w_now.get(i, 0.0)) for i in instruments}, float(holdings.cash_weight), "no_effect", {"reason": "insufficient_history", "observations": int(px.shape[0]), "window": self.spec.window})
        cur = np.array([float(w_now.get(i, 0.0)) for i in instruments])
        obs = build_observation(self.spec, px, cur, float(holdings.cash_weight), drawdown=drawdown)
        weights, cash = allocation_action(self.spec, self._predict(obs), cur, float(holdings.cash_weight))
        return TargetWeights({i: float(weights[k]) for k, i in enumerate(instruments)}, cash, "feasible", {"policy_seed": self.seed, "configuration_id": self.configuration_id, "transform": "softmax"})
