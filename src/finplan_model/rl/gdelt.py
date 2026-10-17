"""Versioned GDELT organization metadata, shared by historical and live states.

Numeric source tone and mention counts are not calibrated stock sentiment.
Historical source availability is an explicit archive reconstruction; actual
central first-seen availability remains separately enforced for live decisions.
"""
from __future__ import annotations

import copy
from datetime import UTC, date, datetime, time, timedelta
import math
import json

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.core.outcome import require_valid
from finplan_model.etf_news import dt

FEATURE_VERSION='gdelt-gkg-v1-organization-tone-v1'
STATE_FEATURE='gdelt_organization_tone_v1'
FEATURE_ORDER=('tone_scaled','article_count_log_scaled','cluster_count_log_scaled','availability_age_scaled','missing_tone')
MODES=('platform','historical_source_proxy')


def validate_document(wrapper):
    document=wrapper['document']
    require_valid(document,'gdelt-metadata-document')
    # Producer source identity uses Python's JSON representation. Preserve its
    # literal bytes inside RFC8785-canonical model artifacts (notably 1.0 vs 1).
    encoded=wrapper.get('canonical_document_json')
    if encoded is None:
        encoded=json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False)
    if not isinstance(encoded,str) or json.loads(encoded)!=document or sha256_checksum(encoded.encode())!=wrapper['checksum']:
        raise FinplanError.precondition('GDELT document checksum mismatch',reason='gdelt_checksum_mismatch')
    if document.get('feature_version')!=FEATURE_VERSION or document.get('rejected_rows')!=0 or not document.get('row_count'):
        raise FinplanError.precondition('GDELT numeric source version or quality is invalid',reason='gdelt_source_invalid')
    lineage=document['lineage'];rights=lineage['rights'];day=date.fromisoformat(document['source_day'])
    if rights.get('rights_status')!='research_and_training' or not rights.get('evidence_url') or not str(rights.get('evidence_checksum','')).startswith('sha256:') or not lineage.get('archive_checksum') or not lineage.get('attribution'):
        raise FinplanError.precondition('GDELT machine metadata requires captured rights and source evidence',reason='gdelt_rights_missing')
    source=dt(lineage['source_available_at']);modified=dt(lineage['source_last_modified'])
    earliest=datetime.combine(day+timedelta(days=1),time(12),tzinfo=UTC)
    if lineage.get('source_day')!=document['source_day'] or source<max(modified,earliest) or dt(lineage['retrieved_at'])<modified:
        raise FinplanError.precondition('GDELT source availability contradicts its archive vintage',reason='gdelt_availability_invalid')
    names=[]
    for feature in document['features']:
        names.append(feature['instrument_id'])
        for key in ('article_count','cluster_count'):
            value=feature[key]
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise FinplanError.precondition('GDELT counts must be observed nonnegative integers',reason='gdelt_source_invalid')
        if not isinstance(feature['missing'],bool) or (feature['tone'] is not None and (isinstance(feature['tone'],bool) or not math.isfinite(float(feature['tone'])) or abs(float(feature['tone']))>100)):
            raise FinplanError.precondition('GDELT numeric tone is invalid',reason='gdelt_source_invalid')
        if feature['tone'] is None and not feature['missing']:
            raise FinplanError.precondition('missing GDELT tone must retain missingness',reason='gdelt_source_invalid')
    if len(names)!=len(set(names)):
        raise FinplanError.precondition('GDELT feature symbols must be unique',reason='gdelt_source_invalid')
    return document


def corpus(documents, symbols, *, availability_mode='historical_source_proxy'):
    if availability_mode not in MODES or not symbols or len(symbols)!=len(set(symbols)):
        raise FinplanError.precondition('GDELT corpus mode or universe is invalid',reason='gdelt_corpus_invalid')
    days={}
    for wrapper in documents:
        document=validate_document(wrapper);day=document['source_day']
        if not set(symbols)<={f['instrument_id'] for f in document['features']}:
            raise FinplanError.precondition('GDELT source lacks requested instruments',reason='gdelt_universe_missing')
        if day in days and days[day]['checksum']!=wrapper['checksum']:
            raise FinplanError.precondition('GDELT immutable source day has conflicting versions',reason='gdelt_source_conflict')
        preserved=copy.deepcopy(wrapper)
        preserved.setdefault('canonical_document_json',json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False))
        days[day]=preserved
    return {'format':'finplan.gdelt-corpus/1','feature_version':FEATURE_VERSION,'symbols':list(symbols),
            'availability_mode':availability_mode,'documents':[days[d] for d in sorted(days)]}


def extract(dataset, *, symbols, decision_time):
    if dataset.get('format')!='finplan.gdelt-corpus/1' or dataset.get('feature_version')!=FEATURE_VERSION or dataset.get('availability_mode') not in MODES or not set(symbols)<=set(dataset['symbols']):
        raise FinplanError.precondition('GDELT state requires a compatible frozen corpus',reason='gdelt_corpus_invalid')
    cutoff=dt(decision_time);eligible=[]
    for wrapper in dataset['documents']:
        document=wrapper['document'];lineage=document['lineage'];available=dt(lineage['source_available_at'])
        if available>cutoff:continue
        if dataset['availability_mode']=='platform':
            central=lineage.get('central_available_at')
            if not central or dt(central)>cutoff or dt(lineage['retrieved_at'])>cutoff:continue
        eligible.append((date.fromisoformat(document['source_day']),available,wrapper))
    if not eligible:
        raise FinplanError.precondition('no GDELT numeric archive was available at the decision cutoff',reason='gdelt_coverage_missing')
    _,available,selected=max(eligible,key=lambda row:row[0]);age=(cutoff-available).total_seconds()/3600
    if age>48:
        raise FinplanError.precondition('GDELT state source is older than the frozen 48-hour limit',reason='gdelt_coverage_missing')
    by_id={row['instrument_id']:row for row in selected['document']['features']};features={}
    for symbol in symbols:
        row=by_id[symbol]
        features[symbol]={'tone_scaled':0. if row['tone'] is None else float(row['tone'])/100.,
                          'article_count_log_scaled':min(1.,math.log1p(row['article_count'])/math.log1p(10000)),
                          'cluster_count_log_scaled':min(1.,math.log1p(row['cluster_count'])/math.log1p(10000)),
                          'availability_age_scaled':age/48.,'missing_tone':float(row['missing'])}
    provenance={'document_checksum':selected['checksum'],'source_day':selected['document']['source_day'],
                'lineage':copy.deepcopy(selected['document']['lineage']),'availability_mode':dataset['availability_mode'],
                'disclosures':copy.deepcopy(selected['document'].get('disclosures',[])),
                'source_age_hours':age,'max_source_age_hours':48.,'freshness_basis':'source_available_at',
                'not_calibrated_stock_sentiment':True}
    return features,provenance


class FrozenGdeltProvider:
    def __init__(self,dataset):
        checked=corpus(dataset['documents'],dataset['symbols'],availability_mode=dataset['availability_mode'])
        if canonical_json_bytes(checked)!=canonical_json_bytes(dataset):
            raise FinplanError.precondition('GDELT corpus must be canonical and versioned',reason='gdelt_corpus_invalid')
        self._dataset=checked;self._frozen=canonical_json_bytes(checked);self.corpus_checksum=sha256_checksum(self._frozen)
        self._cache={}

    def __call__(self,instruments,decision_time):
        if not set(instruments)<=set(self._dataset['symbols']):
            raise FinplanError.precondition('GDELT corpus lacks requested instruments',reason='gdelt_universe_missing')
        if decision_time not in self._cache:
            _,proof=extract(self._dataset,symbols=instruments,decision_time=decision_time)
            selected=next(w for w in self._dataset['documents'] if w['checksum']==proof['document_checksum'])
            if len(self._cache)>=2048:self._cache.clear()
            self._cache[decision_time]={**self._dataset,'documents':[selected]}
        # Return independent selected source bytes. A rollout sees only one
        # eligible daily archive, not repeatedly serialized year-long corpora.
        return copy.deepcopy(self._cache[decision_time])


def freeze_from_api(client,symbols,start_date,end_date,decision_at,*,availability_mode='historical_source_proxy'):
    """Bounded <=31-day reads with unchanged source/central availability evidence."""
    from concurrent.futures import ThreadPoolExecutor
    start=date.fromisoformat(start_date);end=date.fromisoformat(end_date);documents=[];reads=[];chunks=[]
    if end<start or (end-start).days>730:raise ValueError('GDELT interval must be chronological and at most two years')
    while start<=end:
        stop=min(end,start+timedelta(days=30))
        chunks.append((start,stop));start=stop+timedelta(days=1)
    def read(pair):
        a,b=pair
        return client.get_gdelt_metadata(symbols,a.isoformat(),b.isoformat(),decision_at,availability_mode=availability_mode)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(read,chunks))
    for (a,b),result in zip(chunks,results):
        require_valid(result,'api/gdelt-metadata-response')
        if result['feature_version']!=FEATURE_VERSION or result['availability_mode']!=availability_mode or set(result['symbols'])!=set(symbols) or dt(result['decision_at'])!=dt(decision_at):
            raise FinplanError.precondition('GDELT response does not match the frozen query',reason='gdelt_query_mismatch')
        if not result['coverage']['complete']:
            raise FinplanError.precondition('GDELT training archive coverage is incomplete',reason='gdelt_coverage_missing',missing_days=result['coverage']['missing_days'][:20])
        actual=[date.fromisoformat(w['document']['source_day']) for w in result['documents']]
        expected=[a+timedelta(days=k) for k in range((b-a).days+1)]
        if sorted(actual)!=expected or result['coverage']['start_date']!=a.isoformat() or result['coverage']['end_date']!=b.isoformat():
            raise FinplanError.precondition('GDELT response does not contain every requested immutable source day',reason='gdelt_coverage_missing')
        if any(dt(w['document']['lineage']['source_available_at'])>dt(decision_at) for w in result['documents']):
            raise FinplanError.precondition('GDELT response contains archives unavailable at the requested cutoff',reason='gdelt_availability_invalid')
        documents.extend(result['documents']);reads.append(result['coverage'])
    frozen=corpus(documents,symbols,availability_mode=availability_mode)
    return frozen,{'checksum':FrozenGdeltProvider(frozen).corpus_checksum,'reads':reads,
                   'source_days':len(frozen['documents']),'availability_mode':availability_mode}


def morning_market(market):
    """Previous completed prices, next actual session 09:00 New York decision.

    Execution remains the following row's open (09:30), after that decision.
    The final row has no subsequent open or action; preserve its completed-close
    timestamp as an unused terminal mark. Calendar gaps are never guessed.
    """
    from zoneinfo import ZoneInfo
    from finplan_model.sim.market import MarketData
    local=ZoneInfo('America/New_York');cutoffs={}
    for previous,current in zip(market.sessions,market.sessions[1:]):
        cutoffs[previous]=datetime.combine(current,time(9),tzinfo=local).astimezone(UTC)
    cutoffs[market.sessions[-1]]=market.decision_time(market.sessions[-1])
    return MarketData(market.sessions,[b for i in market.instruments for b in market.bars_of(i)],
                      decision_times=cutoffs,synthetic=market.synthetic,dataset_id=market.dataset_id,dataset_checksum=market.dataset_checksum)


def read_live(client,symbols,decision_time):
    """Shared current central first-seen source for PPO and classical inference."""
    day=dt(decision_time).date()
    response=client.get_gdelt_metadata(symbols,(day-timedelta(days=2)).isoformat(),(day-timedelta(days=1)).isoformat(),decision_time,availability_mode='platform')
    require_valid(response,'api/gdelt-metadata-response')
    if response['feature_version']!=FEATURE_VERSION or response['availability_mode']!='platform' or set(response['symbols'])!=set(symbols) or dt(response['decision_at'])!=dt(decision_time):
        raise FinplanError.precondition('GDELT serving response does not match the current query',reason='gdelt_query_mismatch')
    dataset=corpus(response['documents'],symbols,availability_mode='platform')
    features,proof=extract(dataset,symbols=symbols,decision_time=decision_time)
    return dataset,features,proof
