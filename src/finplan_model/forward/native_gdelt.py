"""Hourly reuse of an existing eligible native09:00 GDELT formation.

This adapter reads immutable platform captures. It never retrieves/backdates web
news or changes the frozen native policy observation, cutoff or freshness rules.
"""
from datetime import datetime,time,timedelta
from zoneinfo import ZoneInfo
from finplan_model.etf_news import dt
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum


def latest_completed_formation(now):
    from finplan_model.rl.gdelt_intraday import formation_cutoff
    local=dt(now).astimezone(ZoneInfo('America/New_York'))
    day=local.date() if local.time()>=time(9) else local.date()-timedelta(days=1)
    # Validate through the unchanged native contract, using a completed formation.
    native=datetime.combine(day,time(9),tzinfo=local.tzinfo)
    return formation_cutoff(native.astimezone(dt(now).tzinfo).isoformat())


def load_native_inputs(client,symbols,now):
    from finplan_model.rl.gdelt_intraday import read_live
    native=latest_completed_formation(now)
    # Existing canonical backend read verifies source-day/window/checksum/rights,
    # original central availability <=formation and native1.5h window freshness.
    dataset,features,provenance=read_live(client,symbols,native)
    source_end=dt(provenance['window_end']);at=dt(now)
    lineage=provenance['lineage']
    return {'status':'available','decision_time':native,'dataset':dataset,'features':features,'provenance':provenance,
            'cadence_status':'reused_native_source_features',
            'native_source_formation_at':native,'reuse_as_of':now,
            'source_day':provenance['source_day'],'source_available_at':lineage['source_available_at'],
            'central_available_at':lineage['central_available_at'],
            'source_age_hours_at_reuse':(at-source_end).total_seconds()/3600,
            'native_window_age_hours':provenance['window_age_hours'],
            'native_window_max_age_hours':provenance['max_window_age_hours'],
            'adapter_version':'hourly_native_feature_reuse/2',
            'selection_metadata_role':'Adapter provenance; not an added policy feature',
            'source_checksum':sha256_checksum(canonical_json_bytes(dataset)),
            'disclosure':'Existing native09:00 source features reused with current synthetic holdings; not fresh intraday news. Actual source age is separate from frozen formation-window freshness.'}


def annotate_outcome(outcome,native_input):
    """Source reuse must not falsely claim that a failed/noninvoked actor inferred."""
    outcome['native_input_status']=native_input['status']
    outcome['native_input_detail']={k:v for k,v in native_input.items() if k!='dataset'}
    if native_input['status']=='available':
        outcome['native_input_status']='available_reused_native_inputs'
        if outcome['status'] in ('succeeded','recommended','hold','reused'):
            outcome['native_input_status']='inferred_from_reused_native_inputs'
            outcome['signal_status']='inferred_from_reused_native_inputs'
    return outcome
