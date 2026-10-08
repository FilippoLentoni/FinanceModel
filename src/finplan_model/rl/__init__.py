"""Reinforcement-learning allocation strategies (spec rl-strategies; change add-learning-and-llm-strategies).

* :mod:`finplan_model.rl.spec` - the versioned environment specification (state, action transform,
  reward terms and coefficients, episode, decision frequency) and its ``configuration_id``; pure
  Python, importable without the learning libraries (the control-plane Lambda imports nothing here).
* :mod:`finplan_model.rl.arrays` - aligned price arrays and decision schedules from
  :class:`~finplan_model.sim.market.MarketData` (numpy only).
* :mod:`finplan_model.rl.env` - the gymnasium portfolio environment (needs ``gymnasium``).
* :mod:`finplan_model.rl.policy` - a trained policy as a common-evaluator strategy.
* :mod:`finplan_model.rl.train` - PPO and SAC training with validation checkpoint selection
  (needs ``stable-baselines3`` and CPU ``torch``; the ``rl`` extra, installed in the job image only).

Nothing here imports torch at package import time.
"""

RL_ALGORITHMS = ("ppo", "sac")

__all__ = ["RL_ALGORITHMS"]
