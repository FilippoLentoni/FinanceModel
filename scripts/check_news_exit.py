#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from finplan_model.news_exit import check
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);args=p.parse_args()
result=check(json.loads(args.evidence.read_text()));print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
