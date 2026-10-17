#!/usr/bin/env python3
"""Local frozen evidence runner: no credentials, paid calls, training or orders."""
import argparse,json
from pathlib import Path
from finplan_model.etf_shadow import run_study

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--news',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run_study(json.loads(args.study.read_text()),json.loads(args.news.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'study_id':result['study_id'],'blockers':result['blockers'],'output':str(args.output)}))

if __name__=='__main__':main()
