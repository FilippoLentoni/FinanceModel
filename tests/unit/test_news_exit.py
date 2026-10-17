from finplan_model.news_exit import check


def test_readiness_or_fixture_cannot_close_actual_model_exit():
    result=check({'models':{'ppo':{'eligible':False,'state_extension':[0]*10}},'actor_training_verified':False,'market_data_synthetic':True})
    assert not result['passed']
    assert 'ppo:news_not_consumed' in result['failures']
    assert 'real_market_and_trained_artifact_verification_missing' in result['failures']


def test_each_hosted_optimizer_requires_its_own_model_evidence():
    result=check({'hosted':{'optimization':{'http_status':200,'claim_check':'passed','tool_evidence_correlated':True}}})
    for algorithm in ('min_variance','mean_variance','cvar'):
        assert 'hosted_'+algorithm+':actual_model_evidence_missing' in result['failures']
