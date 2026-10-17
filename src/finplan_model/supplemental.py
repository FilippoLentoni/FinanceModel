"""Eligibility gate for externally acquired data; a receipt is not a quality certificate."""
from datetime import UTC, datetime
import hashlib
import json
import math


def timestamp(value):
    t=datetime.fromisoformat(value.replace('Z','+00:00'))
    if t.tzinfo is None:raise ValueError('supplemental timestamps must include timezone')
    return t.astimezone(UTC)


def eligible_features(record, *, decision_time, payload):
    """Called by research importers before any optimization/PPO/LLM feature use.

    Returns frozen numeric features only. Text and catalog instructions never enter
    the policy context as executable instructions. Historical coverage is separate.
    """
    required={'provider','payload_checksum','payment_receipt_checksum','delivery_status','published_at','retrieved_at','available_at','extractor_version','research_rights','quality_status','features'}
    if not isinstance(record,dict) or required-set(record):raise ValueError('missing supplemental provenance')
    if record['delivery_status']!='received' or record['quality_status']!='validated':raise ValueError('unvalidated supplemental delivery')
    if record['research_rights'] not in ('research_and_training','research_only'):raise ValueError('supplemental research rights not established')
    checksum='sha256:'+hashlib.sha256(payload).hexdigest()
    if checksum!=record['payload_checksum']:raise ValueError('supplemental payload integrity failure')
    receipt=record['payment_receipt_checksum']
    if not isinstance(receipt,str) or len(receipt)!=71 or not receipt.startswith('sha256:'):raise ValueError('missing immutable payment receipt')
    published,retrieved,available=[timestamp(record[k]) for k in ['published_at','retrieved_at','available_at']]
    if not published<=retrieved<=available<=decision_time:raise ValueError('supplemental information unavailable at decision time')
    if record['extractor_version']!='crypto-research-features-v1':raise ValueError('unrecognized supplemental extractor')
    features=record['features']
    if not isinstance(features,dict) or set(features)-{'sentiment','event_count','uncertainty'}:raise ValueError('unrecognized supplemental feature')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in features.values()):raise ValueError('nonfinite supplemental feature')
    if not -1<=features.get('sentiment',0)<=1 or not 0<=features.get('uncertainty',0)<=1 or not 0<=features.get('event_count',0)<=1000:raise ValueError('out-of-range supplemental feature')
    return dict(features)
