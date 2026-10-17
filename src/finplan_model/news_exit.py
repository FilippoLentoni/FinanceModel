"""Strict evidence check: readiness, synthetic fixtures and incompatible actors cannot pass."""

def check(evidence):
    failures=[]
    deliveries=evidence.get('central_deliveries',[])
    receipt_hashes={d.get('receipt_checksum') for d in deliveries if d.get('eligible') and 'synthetic' not in str(d.get('provider','')).lower() and d.get('quality_status')=='validated'}
    if not receipt_hashes:failures.append('no_verified_paid_central_delivery')
    def consumed(rec,kind):
        diagnostics=(rec or {}).get('diagnostics',{})
        news=diagnostics.get('news_evidence',{})
        if news.get('consumed') is not True or not news.get('corpus_checksum'):failures.append(kind+':news_not_consumed');return
        if not news.get('features') or any(f.get('missing',1) for f in news['features'].values()):failures.append(kind+':eligible_features_missing')
        provenance=news.get('provenance',[])
        if not provenance or not any(p.get('receipt_checksum') in receipt_hashes and p.get('eligible') for p in provenance):failures.append(kind+':paid_source_not_correlated')
        if kind=='ppo':
            if 'etf_news_metadata_v1' not in diagnostics.get('input_features',[]) or not diagnostics.get('observation_checksum'):failures.append('ppo:news_observation_missing')
            if not news.get('training_news_corpus_checksum') or len(rec.get('policy_seeds',[]))!=5 or not rec.get('policy_source_run_id'):failures.append('ppo:compatible_trained_ensemble_missing')
    optimizers=evidence.get('optimizer_recommendations',{})
    for algorithm in ('min_variance','mean_variance','cvar'):consumed(optimizers.get(algorithm),algorithm)
    consumed(evidence.get('ppo_recommendation'),'ppo')
    for name in ('min_variance','mean_variance','cvar','ppo'):
        hosted=evidence.get('hosted',{}).get(name,{})
        if hosted.get('http_status')!=200 or hosted.get('claim_check')!='passed' or not hosted.get('tool_evidence_correlated'):
            failures.append('hosted_'+name+':actual_model_evidence_missing')
    if evidence.get('market_data_synthetic') is not False or evidence.get('actor_training_verified') is not True:
        failures.append('real_market_and_trained_artifact_verification_missing')
    return {'passed':not failures,'failures':failures,'criterion':'purchased_news_to_central_storage_to_actual_optimizers_and_trained_ppo_to_hosted_langgraph'}
