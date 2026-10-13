import copy

import numpy as np
import pytest

from finplan_model.core.errors import FinplanError
from finplan_model.evaluate import evaluate
from finplan_model.rl.arrays import PriceArrays
from finplan_model.rl.env import PortfolioEnv
from finplan_model.rl.policy import RLPolicyStrategy
from finplan_model.rl.spec import EnvSpec, build_observation
from finplan_model.selection.recursive_profiles import STATE_PROFILES, profile_protocol
from tests.unit.rl.support import market, tiny_protocol
from tests.unit.rl.test_rl_stability import daily_config


@pytest.mark.parametrize("profile,dimension", list(zip(STATE_PROFILES, [37, 48, 43, 62, 79])))
def test_state_dimensions_finite_on_constant_prices_and_profile_isolation(profile, dimension):
    base = tiny_protocol()
    before = copy.deepcopy(base)
    p = profile_protocol(base, profile)
    spec = EnvSpec.from_dict(p["rl"]["env"])
    obs = build_observation(spec, np.ones((61, 5)), np.full(5, .1), .5, drawdown=.1)
    assert obs.shape == (dimension,) == (spec.observation_size(5),)
    assert np.isfinite(obs).all()
    assert base == before
    reference = profile_protocol(base, "recursive_ppo_state_baseline")
    p["rl"]["env"]["features"] = reference["rl"]["env"]["features"]
    assert p == reference  # Only state differs, including all training/reward/action settings.


@pytest.mark.parametrize("profile", STATE_PROFILES)
def test_enriched_observations_match_training_and_common_evaluator(profile):
    p = profile_protocol(tiny_protocol(), profile)
    spec = EnvSpec.from_dict(p["rl"]["env"])
    mkt, cfg = market(), daily_config()
    arrays = PriceArrays.from_market(mkt)
    observations = []
    def predict(obs):
        observations.append(obs.copy())
        return np.tanh(obs[-6:])
    common = evaluate(RLPolicyStrategy("ppo", predict, spec, seed=0, configuration_id="cfg_synthetic"), mkt, cfg,
                      start=mkt.sessions[70], end=mkt.sessions[120])
    expected = np.array(observations)
    observations.clear()
    trained = PortfolioEnv(arrays, spec, cfg, i0=70, i1=120, mode="eval").evaluate(predict)
    np.testing.assert_allclose(expected, observations, atol=1e-6)
    np.testing.assert_allclose([r["value"] for r in common.simulation.nav], trained["values"], atol=1e-6)


def test_sequence_preserves_order_and_rejects_invalid_history():
    spec = EnvSpec.from_dict({"window": 5, "features": ["recent_returns"]})
    returns = np.array([.01, -.02, .03, -.01, .02])
    prices = np.exp(np.r_[0., returns.cumsum()])[:, None]
    obs = build_observation(spec, prices, np.ones(1), 0.)
    np.testing.assert_allclose(obs, returns * spec.return_scale)
    with pytest.raises(FinplanError):
        build_observation(spec, prices * 0., np.ones(1), 0.)
    with pytest.raises(FinplanError):
        EnvSpec.from_dict({"window": 2, "features": ["recent_returns"]})


def test_frozen_beta_implementation_identity_survives_enrichment():
    from finplan_model.rl.inference import implementation_identity
    from finplan_model.horizon_evaluation import evaluation_implementation
    old = implementation_identity()
    assert old["source_checksums"]["observation_and_actions"] == "sha256:425e06f8a8af645e6351600d49f7943a13ee2de9b652f3cb1569c824c4878bee"
    assert old["source_checksums"]["inference"] == "sha256:8a49bbe54f0e80610fd071981f23302d87e1839ef8eb29e8b646c90ec280fec7"
    assert evaluation_implementation()["source_checksums"]["horizon_evaluation"] == "sha256:7efbdd5b85bd61cd518f398aa941a458307d4ecf6766f7bcc1d35fda956c513f"
    bundle = {"environment": {"features": ["market_summary", "cross_asset_risk"]}}
    assert implementation_identity(bundle)["version"] == "finplan-policy-inference/2"
    assert evaluation_implementation(bundle) != evaluation_implementation()
