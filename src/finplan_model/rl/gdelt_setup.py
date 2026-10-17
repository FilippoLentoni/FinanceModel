"""Bind an accepted source query to immutable, verified CPU-setup evidence."""
from __future__ import annotations
import copy,time
from datetime import date
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.etf_news import dt
from .gdelt_intraday import FEATURE_VERSION,STATE_FEATURE,freeze_from_api

def source_request(snapshot,protocol,snapshot_id,symbols,cutoff):
    if not 1<=len(symbols)<=9 or len(set(symbols))!=len(symbols):
        raise FinplanError.validation('registered GDELT training needs one to nine distinct risky instruments',pointer='/configuration/payload/universe')
    if (date.fromisoformat(snapshot.record['coverage']['end'])-date.fromisoformat(protocol['splits']['train']['start'])).days not in range(1101):
        raise FinplanError.precondition('registered source interval is outside bounded history',reason='gdelt_source_interval_invalid')
    query={'format':'finplan.gdelt-source-query/1','input_snapshot_id':snapshot_id,
           'snapshot_manifest_checksum':snapshot.manifest_checksum,'symbols':sorted(symbols),
           'feature_version':FEATURE_VERSION,'start_date':protocol['splits']['train']['start'],
           'end_date':snapshot.record['coverage']['end'],'decision_at':cutoff,
           'availability_mode':'historical_source_proxy','splits':copy.deepcopy(protocol['splits'])}
    return {'query':query,'query_checksum':sha256_checksum(canonical_json_bytes(query)),'status':'source_freeze_pending_cpu_setup'}

def prepare(inp,protocol,*,deadline,clock=time.monotonic):
    request=protocol['rl'].get('gdelt_corpus_request')
    if not request:return None
    query=request['query'];snapshot=inp.snapshot.snapshot
    expected=source_request(snapshot,protocol,inp.spec['input_snapshot_id'],inp.universe,query['decision_at'])
    if request!=expected or STATE_FEATURE not in protocol['rl']['env']['features'] or dt(query['decision_at'])>dt(inp.ctx.now_ts()):
        raise FinplanError.precondition('accepted source query differs from verified job inputs',reason='gdelt_source_query_mismatch')
    def remaining():
        if deadline is not None and clock()>=deadline:
            raise FinplanError.precondition('source setup exhausted the bounded job runtime before learner start',reason='gdelt_setup_deadline')
    remaining();start=clock();end=date.fromisoformat(query['end_date'])
    required=[day.isoformat() for day in inp.market.sessions if any(
        date.fromisoformat(w['start'])<=day<=min(end,date.fromisoformat(w['end']) if w.get('end') else end)
        for w in protocol['splits'].values())]
    if not required or inp.platform is None:
        raise FinplanError.precondition('source setup lacks verified sessions or authenticated reader',reason='gdelt_training_source_missing')
    inp.ctx.log('gdelt_setup_started',source_query_checksum=request['query_checksum'],required_sessions=len(required))
    dataset,evidence=freeze_from_api(inp.platform,query['symbols'],query['start_date'],query['end_date'],query['decision_at'],
                                    availability_mode=query['availability_mode'],required_days=required,max_workers=2)
    for wrapper in dataset['documents']:
        central=wrapper['document']['lineage'].get('central_available_at')
        lineage=wrapper['document']['lineage']
        observed=[central,lineage.get('retrieved_at'),lineage.get('platform_available_at'),*(source.get('retrieved_at') for source in lineage['sources'])]
        if any(not value or dt(value)>dt(query['decision_at']) for value in observed):
            raise FinplanError.precondition('source was not centrally frozen at accepted query cutoff',reason='gdelt_source_vintage_mismatch')
    remaining();ref=inp.artifacts.put_json(dataset,kind='run_artifact',domain='finance')
    if ref.checksum!=evidence['checksum']:
        raise FinplanError.precondition('frozen source artifact checksum differs from validated corpus',reason='gdelt_training_corpus_integrity')
    metadata={'artifact':ref.to_dict(),'checksum':ref.checksum,'feature_version':FEATURE_VERSION,
              'availability_mode':query['availability_mode'],'source_days':len(required),'required_session_days':required,
              'source_query_checksum':request['query_checksum']}
    protocol['rl']['gdelt_corpus']=metadata
    manifest={'format':'finplan.gdelt-source-realization/1','source_query':query,'source_query_checksum':request['query_checksum'],
              'required_session_days':required,'corpus':metadata,'protocol':copy.deepcopy(protocol)}
    from finplan_model.selection.protocol import protocol_id
    manifest['realized_protocol_checksum']=sha256_checksum(canonical_json_bytes(protocol))
    manifest['realized_protocol_id']=protocol_id(protocol)
    manifest_ref=inp.artifacts.put_json(manifest,kind='run_artifact',domain='finance');remaining()
    inp.ctx.log('gdelt_setup_finished',corpus_checksum=ref.checksum,source_manifest_checksum=manifest_ref.checksum,required_sessions=len(required))
    protocol['rl']['gdelt_corpus']={**metadata,'dataset':dataset}
    return {'status':'complete','source_query_checksum':request['query_checksum'],'corpus':metadata,
            'source_manifest':manifest_ref.to_dict(),'realized_protocol_checksum':manifest['realized_protocol_checksum'],'realized_protocol_id':manifest['realized_protocol_id'],
            'setup_wall_seconds':clock()-start}

def verified_realization(result,spec,artifacts):
    try:return _verified_realization(result,spec,artifacts)
    except (KeyError,TypeError,AttributeError,ValueError) as exc:
        raise FinplanError.precondition('malformed immutable source realization',reason='gdelt_realization_mismatch') from exc

def _verified_realization(result,spec,artifacts):
    request=spec['selection_protocol']['rl'].get('gdelt_corpus_request')
    setup=result.get('payload',{}).get('model_selection',{}).get('source_setup') or {}
    refs={r['artifact_id']:r for r in result.get('artifacts',[])}
    manifest_ref=setup.get('source_manifest') or {};corpus=setup.get('corpus') or {}
    if not artifacts or not result.get('artifacts_complete') or setup.get('status')!='complete' or refs.get(manifest_ref.get('artifact_id'))!=manifest_ref or refs.get(corpus.get('artifact',{}).get('artifact_id'))!=corpus.get('artifact'):
        raise FinplanError.precondition('successful source result lacks immutable realization references',reason='gdelt_realization_missing')
    manifest=artifacts.get_json(manifest_ref);protocol=copy.deepcopy(manifest['protocol'])
    accepted=copy.deepcopy(protocol);accepted['rl'].pop('gdelt_corpus',None)
    from finplan_model.selection.protocol import protocol_id
    checksum=sha256_checksum(canonical_json_bytes(protocol))
    from finplan_model.rl.spec import EnvSpec
    model=result['payload']['model_selection']['rl']['ppo']
    chosen=result['payload']['model_selection']['selection']['chosen_hyperparameters']['ppo']
    expected_env=EnvSpec.from_dict(protocol['rl']['env']).with_reward(**chosen['reward']).document()
    if (manifest.get('format')!='finplan.gdelt-source-realization/1' or
        manifest_ref['kind']!='run_artifact' or corpus['artifact']['kind']!='run_artifact' or
        model['environment']!=expected_env or model['hyperparameters']!=protocol['rl']['ppo'] or sorted(row['seed'] for row in model['seeds'])!=sorted(protocol['rl']['seeds']) or
        accepted!=spec['selection_protocol'] or manifest['source_query']!=request['query'] or
        manifest['source_query_checksum']!=request['query_checksum'] or setup['source_query_checksum']!=request['query_checksum'] or
        sha256_checksum(canonical_json_bytes(request['query']))!=request['query_checksum'] or
        manifest['corpus']!=corpus or protocol['rl']['gdelt_corpus']!=corpus or
        corpus['checksum']!=corpus['artifact']['checksum'] or corpus['source_query_checksum']!=request['query_checksum'] or
        checksum!=manifest['realized_protocol_checksum'] or checksum!=setup['realized_protocol_checksum'] or
        manifest['required_session_days']!=corpus['required_session_days'] or
        request['query']['input_snapshot_id']!=spec['input_snapshot_id'] or
        request['query']['snapshot_manifest_checksum']!=result['dataset_checksum'] or
        request['query']['symbols']!=sorted(spec['universe']) or request['query']['feature_version']!=FEATURE_VERSION or
        manifest['realized_protocol_id']!=protocol_id(protocol) or setup['realized_protocol_id']!=protocol_id(protocol) or
        result['payload']['model_selection']['protocol_id']!=protocol_id(protocol)):
        raise FinplanError.precondition('realized source differs from accepted immutable protocol',reason='gdelt_realization_mismatch')
    return corpus,{'training_source_query_checksum':request['query_checksum'],
                   'training_source_manifest_checksum':manifest_ref['checksum'],'training_realized_protocol_checksum':checksum,'training_realized_protocol_id':protocol_id(protocol)}
