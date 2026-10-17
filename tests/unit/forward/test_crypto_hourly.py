"""Cross-repository authoritative ledger acceptance with moto, never live AWS.

The platform runtime package is provided for this EC2 integration gate. The
standalone model unit suite skips it when that owned package is not installed.
"""
import copy
import json
from datetime import timedelta

import boto3
import pytest
from moto import mock_aws

from finplan_model.forward import crypto_hourly,hourly
from finplan_model.forward.crypto_hourly import crypto_registry
from tests.unit.evaluate.test_crypto_point import END,dataset,iso,source


@pytest.fixture
def evidence():
    repository=pytest.importorskip('finplan_platform.core.repository')
    artifact_module=pytest.importorskip('finplan_platform.core.artifacts')
    evidence_module=pytest.importorskip('finplan_platform.core.modeling_evidence')
    with mock_aws():
        db=boto3.client('dynamodb',region_name='us-east-2')
        repository.create_tables(db,'beta');repo=repository.MetadataRepository(db,'beta')
        s3=boto3.client('s3',region_name='us-east-2');bucket='example-crypto-controlled-reports'
        s3.create_bucket(Bucket=bucket,CreateBucketConfiguration={'LocationConstraint':'us-east-2'})
        kms=boto3.client('kms',region_name='us-east-2').create_key()['KeyMetadata']['Arn']
        yield evidence_module.ModelingEvidence(repo,artifact_module.ArtifactStore(s3,{'reports':bucket},kms_key_id=kms))


def run(e,tmp_path,end,monkeypatch):
    at=iso(end+timedelta(minutes=5));monkeypatch.setattr(hourly,'stamp',lambda:at)
    return crypto_hourly.run(iso(end),'controlled-crypto-hourly',tmp_path,backend=e,source=source(end),now=at)


def test_six_authoritative_books_two_slots_and_duplicate_recovery(evidence,tmp_path,monkeypatch):
    e=evidence
    first=run(e,tmp_path,END,monkeypatch)
    assert len(first['approaches'])==6
    assert all(r['status']=='succeeded' for r in first['approaches']),first
    registry=e.read_registry(first['experiment_id']);rows=registry['approaches']
    assert len({r['portfolio_id'] for r in rows})==6
    assert registry['comparison_policy']['initial_capital']==100000.
    for row in rows:
        book=e.get_portfolio(row['portfolio_id'])
        assert book['revision']==1 and book['paper_state']['cash_balance']==100000.
        assert not book['paper_state']['positions']
        run_record=e.read_run(hourly.identifier('mr',first['experiment_id'],row['approach_id'],first['slot']))
        assert not run_record.get('execution_ref')
    original_inputs={r['input_id']:e.read_input_bytes(r['input_id']) for r in first['approaches']}
    duplicate=run(e,tmp_path,END,monkeypatch)
    assert all(r['retry']=='existing_terminal' for r in duplicate['approaches'])
    assert original_inputs=={iid:e.read_input_bytes(iid) for iid in original_inputs}
    second=run(e,tmp_path,END+timedelta(hours=1),monkeypatch)
    assert len(second['approaches'])==6
    observed_day='2026-01-17'
    observations,_=e.query_day(first['experiment_id'],observed_day,kind='observations')
    valuations=[r for r in observations if r['kind']=='valuation']
    events=[r for r in observations if r['kind']=='paper_execution']
    assert len(valuations)==12 and len(events)==6, [(r['kind'],r['approach_id'],r.get('payload')) for r in observations if r['kind']!='valuation']
    assert len({r['observation_id'] for r in observations})==len(observations)
    for row in rows:
        book=e.get_portfolio(row['portfolio_id'])
        assert book['revision']==2 and book['paper_state']['cash_balance']>=0
        prior=e.read_run(hourly.identifier('mr',first['experiment_id'],row['approach_id'],first['slot']))
        receipt=prior['execution'];execution=receipt['execution']
        assert execution['execution_at']==iso(END+timedelta(hours=1,minutes=5))
        assert execution['after_value']+execution['transaction_cost']==pytest.approx(execution['before_value'])
        assert execution['identity']['execution_calendar']=='CRYPTO_24X7'
        assert all(q['frequency']=='1h' for q in execution['quote_provenance'].values())
        assert bool(receipt['simulated_fills'])==(row['approach_id']!='crypto_cash')
        current=e.read_run(hourly.identifier('mr',first['experiment_id'],row['approach_id'],second['slot']))
        captured=e.read_input(current['input_id'])['payload']
        prepared=captured['numeric']['prepared']
        assert captured['holdings']['revision']==2 and not captured['news']['consumed']
        if row['approach_id'] in ('crypto_btc_buy_hold','crypto_equal_buy_hold'):
            assert current['outcome']['status']=='hold' and current['outcome']['solution_status']=='no_effect'
        if row['family']=='optimization':
            assert prepared['native_source']['as_of']==second['slot']
            assert len(prepared['scenarios'])==47 and len(prepared['covariance'])==2
    revisions={r['portfolio_id']:e.get_portfolio(r['portfolio_id'])['revision'] for r in rows}
    run(e,tmp_path,END+timedelta(hours=1),monkeypatch)
    assert revisions=={r['portfolio_id']:e.get_portfolio(r['portfolio_id'])['revision'] for r in rows}
    again,_=e.query_day(first['experiment_id'],observed_day,kind='observations')
    assert len([r for r in again if r['kind']=='paper_execution'])==6
    # A terminal run whose valuation write was interrupted can repair that missing
    # observation without replacing its exact original input or executing again.
    lost=next(r for r in valuations if r['slot']==second['slot'] and r['approach_id']=='crypto_min_variance')
    original_payload=copy.deepcopy(lost['payload'])
    e.repo.client.delete_item(TableName=e.repo.names['portfolio_observation'],Key={'pk':{'S':lost['observation_id']}})
    run(e,tmp_path,END+timedelta(hours=1),monkeypatch)
    repaired,_=e.query_day(first['experiment_id'],observed_day,kind='observations')
    assert next(r for r in repaired if r['observation_id']==lost['observation_id'])['payload']==original_payload
    assert revisions=={r['portfolio_id']:e.get_portfolio(r['portfolio_id'])['revision'] for r in rows}


def test_source_outage_accounts_all_six_and_keeps_cash_control_available(evidence,tmp_path,monkeypatch):
    at=iso(END+timedelta(minutes=5));monkeypatch.setattr(hourly,'stamp',lambda:at)
    supplied={'investable_universe':['BTC-USD','ETH-USD'],'registry_approaches':crypto_registry(),
              'observations':[],'execution_observations':[],'acquisition_failure':'controlled provider outage'}
    result=crypto_hourly.run(iso(END),'controlled-crypto-outage',tmp_path,backend=evidence,source=supplied,now=at)
    statuses={r['approach_id']:r['status'] for r in result['approaches']}
    assert statuses['crypto_cash']=='succeeded'
    assert all(status=='unavailable' for aid,status in statuses.items() if aid!='crypto_cash')
    stored,_=evidence.query_day(result['experiment_id'],'2026-01-17')
    valuations,_=evidence.query_day(result['experiment_id'],'2026-01-17',kind='observations')
    assert len(stored)==len(valuations)==6
    assert all(evidence.read_input(r['input_id'])['payload'] for r in stored)


def test_timezone_alias_reuses_permanent_slot_without_reinvocation(evidence,tmp_path,monkeypatch):
    first=run(evidence,tmp_path,END,monkeypatch)
    at=iso(END+timedelta(minutes=5))
    alias=crypto_hourly.run(END.isoformat(),first['experiment_id'],tmp_path,backend=evidence,source=source(END),now=at)
    assert alias['slot']==first['slot']
    assert all(r['retry']=='existing_terminal' for r in alias['approaches'])
    rows,_=evidence.query_day(first['experiment_id'],'2026-01-17')
    assert len(rows)==6


def test_interrupted_invocation_recovers_original_input_without_repeating_solver(evidence,tmp_path,monkeypatch):
    original_finish=evidence.finish_run
    state={'failed':False}
    def crash_once(run_id,outcome,at):
        row=evidence.repo.require('research_run',run_id).doc
        if row['approach_id']=='crypto_min_variance' and not state['failed']:
            state['failed']=True
            raise RuntimeError('controlled crash after captured solver before final output')
        return original_finish(run_id,outcome,at)
    monkeypatch.setattr(evidence,'finish_run',crash_once)
    with pytest.raises(RuntimeError,match='controlled crash'):run(evidence,tmp_path,END,monkeypatch)
    rid=hourly.identifier('mr','controlled-crypto-hourly','crypto_min_variance',iso(END))
    interrupted=evidence.read_run(rid);raw=evidence.read_input_bytes(interrupted['input_id'])
    monkeypatch.setattr(evidence,'finish_run',original_finish)
    recovered_at=iso(END+timedelta(minutes=16));monkeypatch.setattr(hourly,'stamp',lambda:recovered_at)
    result=crypto_hourly.run(iso(END),'controlled-crypto-hourly',tmp_path,backend=evidence,source=source(),now=recovered_at)
    recovered=evidence.read_run(rid)
    assert recovered['status']=='failed'
    assert recovered['outcome']['reason']=='recovered_interrupted_invocation_original_input_preserved'
    assert evidence.read_input_bytes(interrupted['input_id'])==raw
    assert len(result['approaches'])==6
    assert all(evidence.get_portfolio(r['portfolio_id'])['revision']==1 for r in evidence.read_registry(result['experiment_id'])['approaches'])
