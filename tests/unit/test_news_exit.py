from finplan_model.news_exit import check


def test_readiness_or_fixture_cannot_close_actual_model_exit():
    result=check({'models':{'ppo':{'eligible':False,'state_extension':[0]*10}},'actor_training_verified':False,'market_data_synthetic':True})
    assert not result['passed']
    assert 'ppo:news_not_consumed' in result['failures']
    assert 'real_market_and_trained_artifact_verification_missing' in result['failures']
