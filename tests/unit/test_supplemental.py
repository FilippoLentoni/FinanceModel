from datetime import UTC,datetime
import hashlib
import pytest
from finplan_model.supplemental import eligible_features


def test_supplemental_retrieval_cannot_backfill_a_historical_decision():
    payload=b'news';r={'provider':'fixture','payload_checksum':'sha256:'+hashlib.sha256(payload).hexdigest(),'payment_receipt_checksum':'sha256:'+'a'*64,'delivery_status':'received','quality_status':'validated','published_at':'2026-01-01T00:00:00Z','retrieved_at':'2026-01-02T00:00:00Z','available_at':'2026-01-02T00:01:00Z','research_rights':'research_and_training','extractor_version':'crypto-research-features-v1','features':{'sentiment':.1}}
    with pytest.raises(ValueError,match='unavailable'):eligible_features(r,decision_time=datetime(2026,1,1,12,tzinfo=UTC),payload=payload)
    assert eligible_features(r,decision_time=datetime(2026,1,3,tzinfo=UTC),payload=payload)=={'sentiment':.1}
    with pytest.raises(ValueError,match='integrity'):eligible_features(r,decision_time=datetime(2026,1,3,tzinfo=UTC),payload=b'changed')
