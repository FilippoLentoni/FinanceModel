"""DS-01 (deterministic, content-addressed datasets) and DS-08 (synthetic label propagates); task 3.1."""

from __future__ import annotations

from finplan_model.core.artifacts import InMemoryArtifactStore
from finplan_model.datasets import Dataset, InMemoryDatasetCatalog, LocalDatasetCatalog, PreparationConfig, prepare_dataset
from finplan_model.evaluate import evaluate
from finplan_model.sim.config import SimulationConfig
from finplan_model.strategies import EqualWeight

from .support import Env, config


def test_repeat_preparation_reuses_the_dataset_and_writes_nothing():
    env = Env()
    sid = env.backfill()
    first = env.prepare([sid])
    objects_after_first = dict(env.store.objects)
    second = env.prepare([sid])
    assert first.reused is False and second.reused is True
    assert second.manifest_checksum == first.manifest_checksum
    assert second.record == first.record  # the first record (created_at included) is reused
    assert env.store.objects.keys() == objects_after_first.keys()


def test_same_inputs_config_and_digest_give_byte_identical_files_and_manifest():
    a, b = Env(seed=1), Env(seed=1)
    sid_a, sid_b = a.backfill(), b.backfill()
    assert sid_a == sid_b
    pa = a.prepare([sid_a])
    b.ctx.clock.advance(hours=5)  # a later run: only the record's created_at differs
    pb = b.prepare([sid_b])
    assert pa.manifest_checksum == pb.manifest_checksum and pa.dataset_id == pb.dataset_id
    assert {k: v[1] for k, v in a.store.objects.items()} == {k: v[1] for k, v in b.store.objects.items()}
    assert pa.record["created_at"] != pb.record["created_at"]


def test_fresh_catalog_recomputes_the_same_manifest_without_new_objects():
    env = Env()
    sid = env.backfill()
    first = env.prepare([sid])
    n = len(env.store.objects)
    again = prepare_dataset([sid], config(), reader=env.provider.reader, store=env.store, catalog=InMemoryDatasetCatalog(), ctx=env.ctx)
    assert again.manifest_checksum == first.manifest_checksum and len(env.store.objects) == n


def test_feature_lookback_change_gives_new_configuration_and_manifest():
    env = Env()
    sid = env.backfill()
    base = env.prepare([sid])
    changed = env.prepare([sid], config(features={"lookback_sessions": 30}))
    assert changed.record["configuration_id"] != base.record["configuration_id"]
    assert changed.manifest_checksum != base.manifest_checksum and changed.dataset_id != base.dataset_id
    assert PreparationConfig.from_dict(config()).configuration_id == base.record["configuration_id"]


def test_image_digest_is_part_of_the_dataset_identity():
    env = Env()
    sid = env.backfill()
    a = env.prepare([sid])
    b = env.prepare([sid], ctx=env.ctx.with_run(env.ctx.ids.run_id(), image_digest="sha256:" + "a" * 64))
    assert a.dataset_id != b.dataset_id and b.record["image_digest"] == "sha256:" + "a" * 64


def test_ticker_is_part_of_the_configuration_id():
    assert PreparationConfig.from_dict(config(instrument={"tickers": ["SPY"]})).configuration_id != PreparationConfig.from_dict(config(instrument={"tickers": ["IVV"]})).configuration_id


def test_local_catalog_persists_and_verifies(tmp_path):
    env = Env()
    sid = env.backfill()
    cat = LocalDatasetCatalog(tmp_path)
    p1 = prepare_dataset([sid], config(), reader=env.provider.reader, store=env.store, catalog=cat, ctx=env.ctx)
    p2 = prepare_dataset([sid], config(), reader=env.provider.reader, store=env.store, catalog=LocalDatasetCatalog(tmp_path), ctx=env.ctx)
    assert p2.reused and p2.record == p1.record


def test_synthetic_label_propagates_to_dataset_market_and_results():
    env = Env()
    sid = env.backfill()
    prep = env.prepare([sid])
    assert prep.record["synthetic"] is True and prep.ref["synthetic"] is True
    ds = Dataset.load(prep.record, env.store)
    assert ds.manifest["synthetic"] is True
    market = ds.market_data()
    assert market.synthetic is True and market.dataset_checksum == prep.manifest_checksum
    fold = ds.folds[0]
    res = evaluate(EqualWeight(), market, SimulationConfig(), start=fold.test.start, end=fold.test.end)
    assert res.synthetic is True and res.to_dict()["synthetic"] is True


def test_dataset_load_verifies_file_checksums():
    import pytest

    from finplan_model.core.errors import FinplanError

    env = Env()
    prep = env.prepare([env.backfill()])
    fid = next(f["artifact_id"] for f in Dataset.load(prep.record, env.store).manifest["files"] if f["name"] == "observations")
    ref, _ = env.store.objects[fid]
    env.store.objects[fid] = (ref, b'{"bars": []}')
    with pytest.raises(FinplanError) as ei:
        Dataset.load(prep.record, env.store)
    assert ei.value.code == "PRECONDITION_FAILED"


def test_holdout_is_not_in_the_general_market_data():
    env = Env()
    ds = env.prepare([env.backfill()]).load(env.store)
    market = ds.market_data()
    assert market.sessions[-1] < ds.holdout_bounds.start
    assert all(b.session_date < ds.holdout_bounds.start for b in ds.bars())


def test_store_is_shared_between_independent_stores_only_by_content():
    a, b = InMemoryArtifactStore(), InMemoryArtifactStore()
    assert a.put_json({"x": 1}, kind="research_dataset_file") == b.put_json({"x": 1}, kind="research_dataset_file")
