import copy
from datetime import UTC,datetime,timedelta
from types import SimpleNamespace
import pytest
from finplan_model.rl.checkpoint import freeze_corpus,prepare,execute
from finplan_model.etf_news import extract
from tests.unit.test_etf_news import data


def corpus():
    d=data();d['deliveries']=[]
    start=datetime(2026,10,16,20,15,tzinfo=UTC)
    for i in range(14):
        r=copy.deepcopy(data()['deliveries'][0]);t=start+timedelta(days=i,hours=1)
        r.update(record_id='sha256:'+format(i,'064x'),available_at=t.isoformat(),retrieved_at=(t-timedelta(minutes=1)).isoformat())
        r['records'][0].update(published_at=(t-timedelta(minutes=2)).isoformat(),url=f'https://example.invalid/{i}')
        d['deliveries'].append(r)
    return d


def bundle():
    first=datetime(2026,7,1,20,tzinfo=UTC)
    times=[];t=first
    while t<=datetime(2026,10,30,20,tzinfo=UTC):
        if t.weekday()<5:times.append(t.isoformat())
        t+=timedelta(days=1)
    return {'instruments':['SPY','VOO'],'decision_times':times,'close':[[100,100] for _ in times], 'exec_price':[[100,100] for _ in times],'volume':[[1000,1000] for _ in times],'has_bar':[[True,True] for _ in times], 'news_start':'2026-10-16T20:15:00Z','checkpoint_at':'2026-10-30T20:15:00Z','snapshot_status':'approved','input_snapshot_id':'synthetic-test-id','snapshot_checksum':'sha256:'+'a'*64,'synthetic':False,'simulation':{}}


def test_overlapping_calendar_windows_union_without_boundary_gaps_or_backdating():
    d=corpus();calls=[]
    def read(symbols,cutoff):
        calls.append(cutoff);out=copy.deepcopy(d)
        # A shared boundary record is permitted; immutable identity deduplicates it.
        out['deliveries']=(d['deliveries'][:7] if len(calls)==1 else d['deliveries'][6:13] if len(calls)==2 else d['deliveries'][8:])
        return out
    frozen,m=freeze_corpus(SimpleNamespace(get_research_news=read),['SPY','VOO'],'2026-10-30T20:15:00Z')
    assert len(calls)==3 and len(frozen['deliveries'])==14
    assert m['checksum'].startswith('sha256:')
    assert frozen['deliveries'][0]['available_at']==d['deliveries'][0]['available_at']
    features,_,_=extract(frozen,symbols=['SPY'],decision_time='2026-10-16T20:15:00Z')
    assert features['SPY']['missing']==1 # not fabricated no-news observation


def test_research_only_blocks_before_learner_or_budget_access(monkeypatch):
    d=corpus();p=bundle();now=datetime(2026,10,30,21,tzinfo=UTC)
    plan=prepare(d,p,now=now)
    assert plan['sessions']==10 and 'insufficient_price_warmup' not in plan['blockers']
    assert 'rights_review_pending' in plan['blockers']
    called=[];monkeypatch.setattr('finplan_model.rl.train.train_policy',lambda *a,**k:called.append(True))
    result=execute(plan,d,p,{},'unused',now=now)
    assert result['jobs_submitted']==0 and result['actors_exported']==0 and not called


def test_training_coverage_is_separate_from_price_warmup_and_empty_is_missing():
    d=corpus();p=bundle();now=datetime(2026,10,30,21,tzinfo=UTC)
    for r in d['deliveries']:r['rights']['status']='research_and_training'
    plan=prepare(d,p,now=now)
    assert plan['status']=='ready' and plan['window']==60
    assert len(plan['coverage'])==10 and len(plan['seeds'])==5
    d['deliveries']=[];plan=prepare(d,p,now=now)
    assert 'forward_news_coverage_missing' in plan['blockers']
    assert all(all(r['inference_missing'].values()) for r in plan['coverage'])


def test_saturated_or_conflicting_windows_fail_closed():
    d=corpus();d['deliveries']=[d['deliveries'][0]]*168
    with pytest.raises(ValueError,match='saturated'):freeze_corpus(SimpleNamespace(get_research_news=lambda *a:d),['SPY'],'2026-10-30T20:15:00Z')


def test_future_prices_rejected():
    with pytest.raises(ValueError,match='future price'):prepare(corpus(),bundle(),now=datetime(2026,10,17,tzinfo=UTC))


def test_execute_pairs_five_seeds_existing_preflight_train_export_no_promotion(monkeypatch,tmp_path):
    from tests.unit.rl.support import sim_config
    d=corpus();p=bundle();p['simulation']=sim_config().to_dict()
    for r in d['deliveries']:r['rights']['status']='research_and_training'
    now=datetime(2026,10,30,21,tzinfo=UTC);plan=prepare(d,p,now=now);calls=[];exported=[]
    def train(arrays,spec,config,**kwargs):
        calls.append((spec.features,kwargs['seed'],kwargs['train_window'],kwargs['validation_window'],kwargs['deadline'],kwargs['news_provider']))
        return SimpleNamespace(model_bytes=b'synthetic-test-only',summary=lambda:{'seed':kwargs['seed']})
    monkeypatch.setattr('finplan_model.rl.train.train_policy',train)
    monkeypatch.setattr('finplan_model.rl.inference_v2.export_actor',lambda *args:exported.append(args) or {'synthetic':True})
    budget={'checked_at':now.isoformat(),'estimated_total_usd_upper_bound':.1,'estimate_basis':'synthetic test quote','project_spent_usd':1,'project_reserved_usd':0,'weekly_spent_usd':0,'weekly_reserved_usd':0,'exclusive_research_lease':True}
    result=execute(plan,d,p,budget,tmp_path,now=now)
    assert len(calls)==len(exported)==10 and result['actors_exported']==10
    assert [c[1] for c in calls]==[0,1,2,3,4]*2
    assert len({c[4] for c in calls})==1 # one combined runtime budget
    assert all(c[5] is None for c in calls[:5]) and all(c[5] for c in calls[5:])
    assert result['serving_promoted'] is False and result['test_status'].startswith('reserved_unopened')
    budget['weekly_spent_usd']=.45
    with pytest.raises(ValueError,match='budget reservation'):execute(plan,d,p,budget,tmp_path,now=now)
    assert len(calls)==10


def test_capture_only_cli_persists_missing_price_report_without_learners(tmp_path,monkeypatch):
    import sys,json
    from scripts.ppo_forward_checkpoint import main
    news=tmp_path/'source.json';news.write_text(json.dumps(corpus()));output=tmp_path/'out'
    monkeypatch.setattr(sys,'argv',['checkpoint','--capture-only','--news',str(news),'--output',str(output)])
    assert main()==2
    report=json.loads((output/'report.json').read_text())
    assert report['blockers']==['missing_price_bundle','rights_review_pending']
    assert report['jobs_submitted']==report['actors_exported']==0
    assert (output/'corpus.json').exists()
