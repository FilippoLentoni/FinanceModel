"""Frozen current-day premarket machine metadata; partial coverage stays explicit."""
from __future__ import annotations
import copy,json,math
from datetime import datetime,time,timedelta
from zoneinfo import ZoneInfo
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.core.outcome import require_valid
from finplan_model.etf_news import dt
from .gdelt import MODES
FEATURE_VERSION='gdelt-gkg-v2-premarket-organization-tone-v1'
STATE_FEATURE='gdelt_intraday_organization_tone_v1'
FEATURE_ORDER=('tone_scaled','article_count_log_scaled','cluster_count_log_scaled','window_age_scaled','source_coverage_fraction')
MAX_WINDOW_AGE_HOURS=1.5

def formation_cutoff(now):
    local=dt(now).astimezone(ZoneInfo('America/New_York'))
    cutoff=datetime.combine(local.date(),time(9),tzinfo=local.tzinfo)
    if local<cutoff:raise FinplanError.precondition('today morning portfolio formation is not due before 09:00 New York',reason='morning_formation_not_due')
    return cutoff.astimezone(dt(now).tzinfo).isoformat()

def validate_document(wrapper):
    d=wrapper['document'];require_valid(d,'gdelt-intraday-document')
    encoded=wrapper.get('canonical_document_json',json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False))
    if json.loads(encoded)!=d or sha256_checksum(encoded.encode())!=wrapper['checksum']:
        raise FinplanError.precondition('intraday source checksum mismatch',reason='gdelt_checksum_mismatch')
    if d['feature_version']!=FEATURE_VERSION or d['rejected_rows']!=0:
        raise FinplanError.precondition('intraday source version or row quality invalid',reason='gdelt_source_invalid')
    lineage=d['lineage'];rights=lineage['rights']
    if rights.get('rights_status')!='research_and_training' or not rights.get('evidence_checksum') or not lineage.get('attribution'):
        raise FinplanError.precondition('intraday metadata requires captured training rights',reason='gdelt_rights_missing')
    start,end,cutoff=map(dt,(d['window_start'],d['window_end'],d['decision_cutoff']))
    local=cutoff.astimezone(ZoneInfo('America/New_York'))
    if local.date().isoformat()!=d['source_day'] or local.hour!=9 or local.minute!=0 or local.second!=0 or (end-start).total_seconds()!=3600 or (cutoff-end).total_seconds()!=900:
        raise FinplanError.precondition('intraday window differs from frozen 09:00 daily formation',reason='gdelt_window_invalid')
    n=d['captured_batches'];expected=d['expected_batches']
    if expected!=4 or n!=len(lineage['sources']) or len(d['missing_batches'])!=4-n or len({s['batch_at'] for s in lineage['sources']})!=n:
        raise FinplanError.precondition('intraday coverage evidence contradicts observed batches',reason='gdelt_coverage_invalid')
    if d['quality_status']!=('complete' if n==4 else 'partial' if n else 'unavailable'):
        raise FinplanError.precondition('intraday coverage quality is inconsistent',reason='gdelt_coverage_invalid')
    for source in lineage['sources']:
        if not source['archive_checksum'] or dt(source['source_available_at'])<max(dt(source['source_last_modified']),dt(source['batch_at'])+timedelta(minutes=15)) or dt(source['retrieved_at'])<dt(source['source_last_modified']):
            raise FinplanError.precondition('intraday source availability contradicts archive evidence',reason='gdelt_availability_invalid')
    expected_times={start+timedelta(minutes=15*k) for k in range(4)}
    if {dt(a['batch_at']) for a in lineage['attempts']}!=expected_times or {dt(s['batch_at']) for s in lineage['sources']} | {dt(t) for t in d['missing_batches']}!=expected_times:
        raise FinplanError.precondition('intraday archive attempts differ from frozen quarter-hour window',reason='gdelt_window_invalid')
    attempts={a['batch_at']:a for a in lineage['attempts']}
    successful={s['batch_at']:s for s in lineage['sources']}
    if set(successful)!={a['batch_at'] for a in lineage['attempts'] if a['http_status']==200} or set(d['missing_batches'])!=set(attempts)-set(successful):
        raise FinplanError.precondition('intraday successful sources contradict actual HTTP attempts',reason='gdelt_coverage_invalid')
    for batch,source in successful.items():
        attempt=attempts[batch]
        if attempt['archive_checksum']!=source['archive_checksum'] or attempt['url']!=source['url'] or dt(source['source_available_at'])>cutoff:
            raise FinplanError.precondition('intraday source identity or availability differs from captured window',reason='gdelt_availability_invalid')
    if n and (lineage['source_available_at'] is None or dt(lineage['source_available_at'])!=max(dt(s['source_available_at']) for s in lineage['sources'])):
        raise FinplanError.precondition('intraday available time must match observed batches',reason='gdelt_availability_invalid')
    if not n and lineage['source_available_at'] is not None:
        raise FinplanError.precondition('missing archive cannot have invented historical availability',reason='gdelt_availability_invalid')
    names=[]
    for f in d['features']:
        names.append(f['instrument_id'])
        for key in ('article_count','cluster_count'):
            if (n==0 and f[key] is not None) or (n>0 and (type(f[key]) is not int or f[key]<0)):
                raise FinplanError.precondition('intraday counts must distinguish observations from unknown values',reason='gdelt_source_invalid')
        if not n and f['tone'] is not None:raise FinplanError.precondition('unknown archive cannot contain observed tone',reason='gdelt_source_invalid')
        if f['missing']!=(n<4):raise FinplanError.precondition('intraday missingness must reflect batch coverage',reason='gdelt_coverage_invalid')
        if f['tone'] is not None and (not math.isfinite(f['tone']) or abs(f['tone'])>100):raise FinplanError.precondition('intraday tone invalid',reason='gdelt_source_invalid')
    if len(names)!=len(set(names)):raise FinplanError.precondition('duplicate intraday feature symbol',reason='gdelt_source_invalid')
    return d

def corpus(documents,symbols,*,availability_mode='historical_source_proxy'):
    if availability_mode not in MODES:raise ValueError('invalid availability mode')
    days={}
    for w in documents:
        d=validate_document(w)
        if not set(symbols)<={f['instrument_id'] for f in d['features']}:raise FinplanError.precondition('intraday corpus lacks instruments',reason='gdelt_universe_missing')
        if d['source_day'] in days and days[d['source_day']]['checksum']!=w['checksum']:raise FinplanError.precondition('conflicting immutable day',reason='gdelt_source_conflict')
        v=copy.deepcopy(w);v.setdefault('canonical_document_json',json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False));days[d['source_day']]=v
    return {'format':'finplan.gdelt-corpus/1','feature_version':FEATURE_VERSION,'symbols':list(symbols),'availability_mode':availability_mode,'documents':[days[k] for k in sorted(days)]}

def extract(dataset,*,symbols,decision_time):
    cutoff=dt(decision_time);local=cutoff.astimezone(ZoneInfo('America/New_York'))
    if dataset.get('feature_version')!=FEATURE_VERSION or not set(symbols)<=set(dataset.get('symbols',[])) or local.hour!=9 or local.minute!=0 or local.second!=0:
        raise FinplanError.precondition('intraday state needs exact frozen source and 09:00 cutoff',reason='gdelt_window_invalid')
    selected=next((w for w in dataset['documents'] if w['document']['source_day']==local.date().isoformat()),None)
    if selected is None:raise FinplanError.precondition('no actual current-day archive audit exists',reason='gdelt_coverage_missing')
    d=selected['document'];lineage=d['lineage'];n=d['captured_batches'];mode=dataset['availability_mode']
    if dt(d['decision_cutoff'])!=cutoff or (n and dt(lineage['source_available_at'])>cutoff):raise FinplanError.precondition('source arrived after frozen formation cutoff',reason='gdelt_availability_invalid')
    if mode=='platform':
        if not n:raise FinplanError.precondition('no current-day window source is available',reason='gdelt_coverage_missing')
        if not lineage.get('central_available_at') or dt(lineage['central_available_at'])>cutoff or dt(lineage['retrieved_at'])>cutoff:raise FinplanError.precondition('central source was unavailable at morning formation',reason='gdelt_availability_invalid')
    age=(cutoff-dt(d['window_end'])).total_seconds()/3600
    if age<0 or age>MAX_WINDOW_AGE_HOURS:raise FinplanError.precondition('intraday window outside registered freshness limit',reason='gdelt_coverage_missing')
    by={r['instrument_id']:r for r in d['features']};features={}
    for s in symbols:
        f=by[s];features[s]={'tone_scaled':0. if f['tone'] is None else f['tone']/100.,'article_count_log_scaled':min(1.,math.log1p(f['article_count'] or 0)/math.log1p(10000)),'cluster_count_log_scaled':min(1.,math.log1p(f['cluster_count'] or 0)/math.log1p(10000)),'window_age_scaled':age/MAX_WINDOW_AGE_HOURS,'source_coverage_fraction':n/4.}
    proof={'document_checksum':selected['checksum'],'source_day':d['source_day'],'window_start':d['window_start'],'window_end':d['window_end'],'lineage':copy.deepcopy(lineage),'availability_mode':mode,'quality_status':d['quality_status'],'captured_batches':n,'expected_batches':4,'missing_batches':d['missing_batches'],'disclosures':d['disclosures'],'window_age_hours':age,'max_window_age_hours':MAX_WINDOW_AGE_HOURS,'freshness_basis':'window_end_at_fixed_09:00_formation','not_calibrated_stock_sentiment':True,'missingness_basis':'retrospective_archive_audit' if mode=='historical_source_proxy' else 'actual_central_capture'}
    return features,proof

class FrozenGdeltProvider:
    def __init__(self,dataset):
        self._dataset=corpus(dataset['documents'],dataset['symbols'],availability_mode=dataset['availability_mode']);self.corpus_checksum=sha256_checksum(canonical_json_bytes(self._dataset));self._by={d['document']['source_day']:d for d in self._dataset['documents']}
    def __call__(self,instruments,decision_time):
        day=dt(decision_time).astimezone(ZoneInfo('America/New_York')).date().isoformat();w=self._by.get(day)
        if w is None:raise FinplanError.precondition('intraday audit missing for required state',reason='gdelt_coverage_missing')
        value={**self._dataset,'documents':[w]};extract(value,symbols=instruments,decision_time=decision_time);return copy.deepcopy(value)

def read_live(client,symbols,decision_time):
    day=dt(decision_time).astimezone(ZoneInfo('America/New_York')).date().isoformat();r=client.get_gdelt_intraday_metadata(symbols,day,day,decision_time,availability_mode='platform');require_valid(r,'api/gdelt-intraday-response')
    dataset=corpus(r['documents'],symbols,availability_mode='platform');f,p=extract(dataset,symbols=symbols,decision_time=decision_time);return dataset,f,p

def freeze_from_api(client,symbols,start_date,end_date,decision_at,*,availability_mode='historical_source_proxy',required_days=None):
    from datetime import date,timedelta
    from concurrent.futures import ThreadPoolExecutor
    start,end=map(date.fromisoformat,(start_date,end_date));chunks=[]
    required=None if required_days is None else {date.fromisoformat(d).isoformat() for d in required_days}
    if required is not None and (not required or any(d<start_date or d>end_date for d in required)):
        raise ValueError('required session days must be nonempty and within the frozen interval')
    if end<start or (end-start).days>1100:raise ValueError('intraday interval must be chronological and bounded')
    while start<=end:
        stop=min(end,start+timedelta(days=30))
        if required is None or any(start.isoformat()<=d<=stop.isoformat() for d in required):chunks.append((start,stop))
        start=stop+timedelta(days=1)
    def read(pair):return client.get_gdelt_intraday_metadata(symbols,pair[0].isoformat(),pair[1].isoformat(),decision_at,availability_mode=availability_mode)
    docs=[];reads=[]
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(read,chunks))
    for (a,b),r in zip(chunks,results):
        require_valid(r,'api/gdelt-intraday-response')
        expected={(a+timedelta(days=k)).isoformat() for k in range((b-a).days+1)}
        if required is not None:expected &= required
        actual=[w['document']['source_day'] for w in r['documents']]
        if r['feature_version']!=FEATURE_VERSION or r['availability_mode']!=availability_mode or set(r['symbols'])!=set(symbols) or dt(r['decision_at'])!=dt(decision_at) or not expected<=set(actual) or len(actual)!=len(set(actual)) or any(d<a.isoformat() or d>b.isoformat() for d in actual):
            raise FinplanError.precondition('intraday response lacks required audited source days',reason='gdelt_coverage_missing')
        docs.extend(w for w in r['documents'] if w['document']['source_day'] in expected);reads.append(r['coverage'])
    dataset=corpus(docs,symbols,availability_mode=availability_mode)
    return dataset,{'checksum':FrozenGdeltProvider(dataset).corpus_checksum,'reads':reads,'source_days':len(docs),'required_session_days':sorted(required) if required is not None else None,'availability_mode':availability_mode}
