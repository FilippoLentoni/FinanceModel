#!/usr/bin/env python3
"""Operator checkpoint: read central data, freeze locally, optionally train local candidates."""
import argparse
import json
from pathlib import Path
from finplan_model.rl.checkpoint import freeze_corpus,prepare,execute
from finplan_model.core.platform import HttpPlatformClient


def main():
    p=argparse.ArgumentParser()
    source=p.add_mutually_exclusive_group()
    source.add_argument('--prices',type=Path,help='operator-frozen approved snapshot price bundle')
    source.add_argument('--snapshot-id',help='read and verify an approved real price snapshot through the platform API')
    p.add_argument('--simulation',type=Path,help='frozen simulation JSON for --snapshot-id (defaults to repository beta configuration)')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--endpoint',help='central platform HTTPS endpoint; reads only')
    p.add_argument('--region',default='us-east-2')
    p.add_argument('--news',type=Path,help='already frozen corpus instead of central reads')
    p.add_argument('--budget',type=Path,help='fresh whole-matched-run budget reservation; required for execute')
    p.add_argument('--capture-only',action='store_true',help='freeze corpus and report missing_price_bundle; no learners')
    p.add_argument('--checkpoint-at',default='2026-10-30T20:15:00Z')
    p.add_argument('--news-start',default='2026-10-16T20:15:00Z')
    p.add_argument('--symbols',default='VOO,GOOGL')
    p.add_argument('--execute',action='store_true',help='local CPU training and export only, never auto promotion')
    a=p.parse_args()
    if a.capture_only and a.execute:p.error('--capture-only cannot execute learners')
    if not a.capture_only and not (a.prices or a.snapshot_id):p.error('--prices or --snapshot-id required unless --capture-only')
    if a.snapshot_id and not a.endpoint:p.error('--snapshot-id requires --endpoint')
    prices=json.loads(a.prices.read_text()) if a.prices else {'instruments':a.symbols.split(','),'checkpoint_at':a.checkpoint_at,'news_start':a.news_start}
    a.output.mkdir(parents=True,exist_ok=True)
    client=None
    if a.endpoint:
        import boto3
        client=HttpPlatformClient(a.endpoint,region=a.region,credentials=boto3.Session().get_credentials().get_frozen_credentials())
    if a.snapshot_id:
        from finplan_model.rl.checkpoint_prices import freeze_prices
        simulation=json.loads(a.simulation.read_text()) if a.simulation else json.loads((Path(__file__).resolve().parents[1]/'config/beta.json').read_text())['simulation_defaults']
        prices=freeze_prices(client,a.snapshot_id,prices['instruments'],simulation,news_start=a.news_start,checkpoint_at=a.checkpoint_at)
        (a.output/'prices.json').write_text(json.dumps(prices,indent=2,allow_nan=False)+'\n')
    if a.news:
        dataset=json.loads(a.news.read_text());manifest={'source':'operator_frozen_corpus'}
    else:
        if not a.endpoint:p.error('--endpoint or --news required')
        try:
            dataset,manifest=freeze_corpus(client,prices['instruments'],prices['checkpoint_at'])
        except Exception as exc:
            report={'status':'blocked','blockers':['central_corpus_read_failed'],'error':str(exc),'jobs_submitted':0,'actors_exported':0}
            (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report));return 2
    for name,doc in (('corpus',dataset),('manifest',manifest)):
        (a.output/f'{name}.json').write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
    if a.capture_only:
        from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
        from finplan_model.rl.etf_news import validate_corpus
        merged=validate_corpus(dataset)
        report={'status':'blocked','blockers':['missing_price_bundle'],'checkpoint_at':prices['checkpoint_at'],'news_start':prices['news_start'],'corpus_checksum':sha256_checksum(canonical_json_bytes(dataset)),'delivery_count':len(merged['deliveries']),'rights_statuses':sorted({r['rights']['status'] for r in merged['deliveries']}),'jobs_submitted':0,'actors_exported':0,'serving_promoted':False}
        if 'research_only' in report['rights_statuses']:report['blockers'].append('rights_review_pending')
    else:
        report=prepare(dataset,prices)
    (a.output/'request.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    try:
        if a.execute:
            if not a.budget:p.error('--execute requires --budget')
            report=execute(report,dataset,prices,json.loads(a.budget.read_text()),a.output/'candidates')
    except Exception as exc:
        report={**report,'status':'blocked','blockers':sorted(set(report['blockers']+['execution_failed'])), 'error':str(exc),'serving_promoted':False}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report['status'],'blockers':report['blockers'],'output':str(a.output)}))
    return 0 if report['status']!='blocked' else 2

if __name__=='__main__':raise SystemExit(main())
