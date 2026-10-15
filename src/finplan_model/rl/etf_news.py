"""Explicit immutable corpus provider for opt-in ETF news observation experiments."""
import json
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.core.outcome import require_valid
from finplan_model.core.errors import FinplanError

class FrozenEtfNewsProvider:
    """Retains real retrieval/availability dates; requesting old cutoffs cannot backdate data.

    Persist corpus_checksum with experiment provenance. This provider does not grant
    training rights; build_observation enforces them and per-decision coverage.
    """
    def __init__(self,dataset):
        require_valid(dataset,'api/research-news-response')
        self._frozen=canonical_json_bytes(dataset)
        self.corpus_checksum=sha256_checksum(self._frozen)

    def __call__(self,instruments,decision_time):
        doc=json.loads(self._frozen)
        if not set(instruments)<=set(doc['symbols']):
            raise FinplanError.precondition('Frozen news corpus does not cover requested instruments',reason='news_corpus_universe_missing')
        doc['symbols']=list(instruments)
        doc['decision_at']=decision_time
        return doc
