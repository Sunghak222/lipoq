"""Package verified saved experiments for the submission notebook; no model calls."""
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

import numpy as np
from common.comet import sha256
from common.results import artifact_path, method_name, read_jsonl, write_json
from random_search.reference import load_reference
from vqa.audit import verify
from vqa.config import load_config


def portable(value):
    if isinstance(value, str):
        return value.replace(str(ROOT)+'/', '')
    if isinstance(value, list):
        return [portable(v) for v in value]
    if isinstance(value, dict):
        return {k: portable(v) for k, v in value.items()}
    return value


def collect(source, seeds):
    current = load_config(ROOT/'configs/vqa.yaml')
    payload = {'schema_version': 1, 'prepared_at': datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(),
               'requested_seeds': seeds, 'config': portable(current), 'runs': [], 'availability': []}
    summaries = [(p, json.loads(p.read_text())) for p in source.rglob('summary.json')]
    for seed in seeds:
        completed = [p.parent for p, data in summaries
                     if data.get('status') == 'completed' and method_name(data) == 'vqa'
                     and json.loads(artifact_path(p.parent, 'config.json').read_text())['seed'] == seed]
        if not completed:
            payload['availability'].append({'seed': seed, 'vqa': 'not available', 'random': 'not available'})
            continue
        ref = load_reference(max(completed, key=lambda p: p.name))
        path, cfg, summary = ref['path'], ref['config'], ref['summary']
        if cfg['seed'] != seed:
            raise ValueError(f'Seed folder/config mismatch: {path}')
        for field in ('search', 'condition', 'circuit', 'optimizer'):
            if cfg[field] != current[field]:
                raise ValueError(f'Archive has different {field}; select compatible experiments: {path}')
        audit = verify(path)
        evaluations = read_jsonl(artifact_path(path, 'evaluations.jsonl'))
        names = ('config.json', 'summary.json', 'history.csv', 'evaluations.jsonl', 'candidates.jsonl',
                 'initial_parameters.npy', 'final_parameters.npy', 'model.json', 'catalog.json', 'domain.json')
        run = {'seed': seed, 'source': str(path.relative_to(ROOT)), 'config': portable(cfg),
               'summary': portable(summary), 'audit': portable(audit),
               'model': json.loads((artifact_path(path, 'model.json')).read_text()),
               'catalog': json.loads((artifact_path(path, 'catalog.json')).read_text()),
               'domain': json.loads((artifact_path(path, 'domain.json')).read_text()),
               'history': list(csv.DictReader((artifact_path(path, 'history.csv')).open())),
               'candidates': [{k:r[k] for k in ('composition_key','comet_score','from_persistent_cache')} for r in ref['records']],
               'evaluations': [{k:r[k] for k in ('phase','iteration','counts','mean_comet_score','shot_standard_error')} for r in evaluations],
               'initial_parameters': np.load(artifact_path(path, 'initial_parameters.npy'), allow_pickle=False).tolist(),
               'final_parameters': np.load(artifact_path(path, 'final_parameters.npy'), allow_pickle=False).tolist(),
               'source_sha256': {name:sha256(artifact_path(path, name)) for name in names}, 'random': None}
        matched = []
        for candidate, result in summaries:
            if method_name(result) != 'random':
                continue
            if result.get('status') == 'completed' and Path(result.get('reference_run','')).resolve() == path:
                matched.append(candidate.parent)
        if matched:
            random_path = max(matched, key=lambda p:p.name)
            random_summary = json.loads((random_path/'summary.json').read_text())
            records = read_jsonl(random_path/'candidates.jsonl')
            budget = ref['budget']
            if random_summary['model_fingerprint'] != summary['model_fingerprint']:
                raise ValueError('VQA/random model fingerprint mismatch')
            indices = np.random.default_rng(random_summary['seed']).choice(ref['encoder'].cardinality, budget, replace=False)
            if [r['composition_key'] for r in records] != [ref['encoder'].composition_at(i).key for i in indices]:
                raise ValueError('Random log does not match uniform without-replacement sampling')
            if len(records) != random_summary['counts']['unique_sampled_compositions'] or len(records) != budget:
                raise ValueError('Mismatched unique candidate budgets')
            if max(r['comet_score'] for r in records) != random_summary['best']['comet_score']:
                raise ValueError('Random best does not match its candidate log')
            run['random'] = {'source': str(random_path.relative_to(ROOT)), 'summary': portable(random_summary),
                'candidates': [{k:r[k] for k in ('composition_key','comet_score')} for r in records],
                'comparison': portable(json.loads((artifact_path(random_path, 'comparison.json')).read_text())),
                'source_sha256': {name:sha256(artifact_path(random_path, name)) for name in ('summary.json','config.json','candidates.jsonl','comparison.json')}}
        payload['runs'].append(run)
        payload['availability'].append({'seed':seed,'vqa':'completed','random':'completed' if matched else 'not available'})
    if not payload['runs']:
        raise ValueError('No completed VQA evidence found; never create synthetic results.')
    if len({run['summary']['model_fingerprint'] for run in payload['runs']}) != 1:
        raise ValueError('Selected runs use different COMET model fingerprints')
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/lnp_search')
    parser.add_argument('--seeds',type=int,nargs='+',default=[21,22,23])
    parser.add_argument('--output',type=Path,default=ROOT/'notebooks/data/experiment_snapshot.json.gz')
    args=parser.parse_args()
    if len(args.seeds)!=len(set(args.seeds)):
        parser.error('Seeds must be distinct')
    args.output = args.output.resolve()
    data=collect(args.source.resolve(),args.seeds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(gzip.compress(json.dumps(data,allow_nan=False,separators=(',',':')).encode(),mtime=0))
    manifest={'schema_version':1,'prepared_at':data['prepared_at'],'file':args.output.name,
              'sha256':sha256(args.output),'availability':data['availability'],
              'description':'Verified real-COMET archives; compact score/count records, not synthetic data. Original file hashes are inside the snapshot.'}
    write_json(args.output.with_name('manifest.json'),manifest)
    snapshot_label = args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output
    print(json.dumps({'snapshot':str(snapshot_label),'bytes':args.output.stat().st_size,
                      'availability':data['availability']},indent=2))


if __name__=='__main__':
    main()
