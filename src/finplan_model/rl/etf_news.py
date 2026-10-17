"""Explicit immutable corpus provider for opt-in ETF news observation experiments."""
import json
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.core.outcome import require_valid
from finplan_model.core.errors import FinplanError

PARTITIONED_FORMAT = 'finplan.partitioned-etf-news/1'


def validate_corpus(dataset):
    """Validate every bounded API response before combining a wider research universe."""
    if dataset.get('format') != PARTITIONED_FORMAT:
        require_valid(dataset, 'api/research-news-response')
        return dataset
    if set(dataset) != {'format', 'symbols', 'decision_at', 'parts'} or not dataset['parts']:
        raise FinplanError.precondition('partitioned news corpus envelope is invalid', reason='news_corpus_invalid')
    symbols=[]; deliveries={}; first=None
    for part in dataset['parts']:
        require_valid(part, 'api/research-news-response')
        if part['decision_at'] != dataset['decision_at'] or set(symbols) & set(part['symbols']):
            raise FinplanError.precondition('partitioned news cutoffs and symbol coverage must match', reason='news_corpus_partition_mismatch')
        symbols.extend(part['symbols']); first=first or part
        if (part['contract_version'],part['historical_coverage']) != (first['contract_version'],first['historical_coverage']):
            raise FinplanError.precondition('partitioned news provenance differs', reason='news_corpus_partition_mismatch')
        for row in part['deliveries']:
            key=row['record_id']
            if key in deliveries and canonical_json_bytes(deliveries[key]) != canonical_json_bytes(row):
                raise FinplanError.precondition('immutable delivery differs between partitions', reason='news_corpus_conflicting_delivery')
            deliveries[key]=row
    if len(dataset['symbols']) != len(set(dataset['symbols'])) or set(symbols) != set(dataset['symbols']):
        raise FinplanError.precondition('partitioned news universe is incomplete', reason='news_corpus_universe_missing')
    return {**first, 'symbols':list(dataset['symbols']), 'decision_at':dataset['decision_at'],
            'deliveries':sorted(deliveries.values(),key=lambda r:(r['available_at'],r['record_id']))}


def read_partitioned_news(client, instruments, decision_time):
    """Respect the central API's five-symbol bound without changing its contracts."""
    names=list(instruments)
    if len(names)<=5:return client.get_research_news(names,decision_time)
    parts=[client.get_research_news(names[k:k+4],decision_time) for k in range(0,len(names),4)]
    doc={'format':PARTITIONED_FORMAT,'symbols':names,'decision_at':decision_time,'parts':parts}
    validate_corpus(doc)
    return doc

class FrozenEtfNewsProvider:
    """Retains real retrieval/availability dates; requesting old cutoffs cannot backdate data.

    Persist corpus_checksum with experiment provenance. This provider does not grant
    training rights; build_observation enforces them and per-decision coverage.
    """
    def __init__(self,dataset):
        validate_corpus(dataset)
        self._frozen=canonical_json_bytes(dataset)
        self.corpus_checksum=sha256_checksum(self._frozen)

    def __call__(self,instruments,decision_time):
        doc=json.loads(self._frozen)
        if not set(instruments)<=set(doc['symbols']):
            raise FinplanError.precondition('Frozen news corpus does not cover requested instruments',reason='news_corpus_universe_missing')
        doc['symbols']=list(instruments)
        doc['decision_at']=decision_time
        if doc.get('format') == PARTITIONED_FORMAT:
            doc['parts']=[{**p,'symbols':[s for s in p['symbols'] if s in instruments], 'decision_at':decision_time}
                          for p in doc['parts'] if set(p['symbols']) & set(instruments)]
        return doc
