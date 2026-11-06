"""Real COMET benchmark; reference scores never enter the search algorithms."""
from copy import deepcopy
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from .benchmark_config import load_benchmark_config
from .benchmark_metrics import TrialEvaluator, exact_learning, random_search
from .comet import CometOracle, SCORE_DESCRIPTION, sha256
from .continuous import refine
from .encoding import Composition, CompositionEncoder
from .results import append_csv, create_run_dir, write_json
from .vqa import optimize


def audit_archive(path, cfg, encoder, reference, tolerance, fingerprint):
    path = Path(path).resolve()
    archived_cfg = json.loads((path/'config.json').read_text())
    for section in (archived_cfg['search']['candidates'], archived_cfg['search']['ratios']['bounds']):
        if 'helper' in section and 'phospholipid' not in section:
            section['phospholipid'] = section.pop('helper')
    summary = json.loads((path/'summary.json').read_text())
    for field in ('search', 'condition'):
        if archived_cfg[field] != cfg[field]:
            raise ValueError(f'Historical run has incompatible {field}')
    if summary['model_fingerprint'] != fingerprint:
        raise ValueError('Historical model fingerprint differs from current COMET')
    initial = np.load(path/'initial_parameters.npy', allow_pickle=False)
    final = np.load(path/'final_parameters.npy', allow_pickle=False)
    result = exact_learning(encoder, archived_cfg, initial, final, reference,
                            summary['best']['composition_key'], tolerance)
    return {**result, 'source_run': str(path), 'seed': archived_cfg['seed'],
            'archived_best': summary['best'], 'initial_parameters': initial.tolist(),
            'final_parameters': final.tolist(),
            'archived_best_minus_reference_score': summary['best']['comet_score']-result['found_best_reference_score'],
            'source_hashes': {name: sha256(path/name) for name in
                              ('config.json', 'summary.json', 'initial_parameters.npy', 'final_parameters.npy')}}


def metric_row(space, seed, method, stage, best, unique, requests, seconds, comet_seconds, budget):
    return {'space': space, 'seed': seed, 'method': method, 'stage': stage,
            'best_score': best['comet_score'], 'unique_comet_evaluations': unique,
            'composition_requests': requests, 'seconds': seconds, 'comet_seconds': comet_seconds,
            'matched_discrete_budget': budget}


def aggregate(rows, seeds, tolerance):
    groups = []
    for space in ('base', 'expanded'):
        for stage in sorted({r['stage'] for r in rows}):
            for method in ('vqa', 'random'):
                selected = [r for r in rows if (r['space'], r['stage'], r['method']) == (space, stage, method)]
                if not selected:
                    continue
                group = {'space': space, 'stage': stage, 'method': method, 'seeds': len(selected)}
                for metric in ('best_score', 'unique_comet_evaluations', 'seconds', 'comet_seconds'):
                    values = [r[metric] for r in selected]
                    group[metric] = {'mean': float(np.mean(values)),
                                     'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0,
                                     'min': min(values), 'max': max(values)}
                groups.append(group)
    pairs = []
    for space in ('base', 'expanded'):
        for stage in sorted({r['stage'] for r in rows}):
            deltas = []
            for seed in seeds:
                selected = {r['method']: r for r in rows if (r['space'], r['stage'], r['seed']) == (space, stage, seed)}
                if len(selected) == 2:
                    deltas.append(selected['vqa']['best_score']-selected['random']['best_score'])
            if deltas:
                pairs.append({'space': space, 'stage': stage, 'vqa_minus_random': deltas,
                              'vqa_wins': sum(d > tolerance for d in deltas),
                              'ties': sum(abs(d) <= tolerance for d in deltas),
                              'random_wins': sum(d < -tolerance for d in deltas)})
    return {'groups': groups, 'paired_comparisons': pairs,
            'caution': 'These seed comparisons are descriptive, not evidence of quantum advantage or biological efficacy.'}


def run_benchmark(config_path, reference_run=None):
    started = time.perf_counter()
    directory = None
    oracle = None
    try:
        settings, experiments = load_benchmark_config(config_path)
        directory = create_run_dir(settings['output_directory'], 'benchmark')
        print(f'Benchmark results: {directory}', flush=True)
        write_json(directory/'config.json', settings)
        write_json(directory/'experiments.json', experiments)
        write_json(directory/'environment.json', {
            'started_at': datetime.now().astimezone().isoformat(), 'python': sys.version,
            'platform': platform.platform(), 'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}})
        encoders = {name: CompositionEncoder(cfg['search']) for name, cfg in experiments.items()}
        base = experiments['base']
        oracle = CometOracle(base['comet'], base['condition'], directory/'comet_load.log')
        write_json(directory/'model.json', oracle.provenance)
        write_json(directory/'catalog.json', oracle.catalog)
        preflight = oracle.preflight()
        write_json(directory/'preflight.json', preflight)
        for name, encoder in encoders.items():
            write_json(directory/f'{name}_domain.json', oracle.validate_search(encoder))
            write_json(directory/f'{name}_encoding.json', encoder.metadata())
        # Warm all selected molecule features before either method is timed.
        warmup_start = time.perf_counter()
        encoder = encoders['expanded']
        warmup = [Composition(tuple(ids[i % len(ids)] for ids in encoder.candidates),
                              encoder.ratios[0], encoder.denominator)
                  for i in range(max(map(len, encoder.candidates)))]
        oracle.predict(warmup)
        warmup_seconds = time.perf_counter()-warmup_start
        print('Real COMET preflight passed; evaluating the entire base grid.', flush=True)
        exhaustive = TrialEvaluator(oracle, directory/'exhaustive')
        exhaustive_start = time.perf_counter()
        encoder = encoders['base']
        batch_size = base['comet']['batch_size']
        for offset in range(0, encoder.cardinality, batch_size):
            exhaustive.score([encoder.composition_at(i) for i in range(offset, min(offset+batch_size, encoder.cardinality))], 'exhaustive')
        exhaustive_summary = {'best': exhaustive.best, 'evaluations': len(exhaustive.seen),
                              'seconds': time.perf_counter()-exhaustive_start,
                              'comet_seconds': exhaustive.comet_seconds,
                              'scope': 'Global maximum only within the finite base grid, fixed model and condition',
                              'score_tolerance': settings['score_tolerance']}
        write_json(directory/'exhaustive'/'summary.json', exhaustive_summary)
        reference = dict(exhaustive.scores)
        archived = None
        if reference_run:
            archived = audit_archive(reference_run, base, encoder, reference, settings['score_tolerance'], oracle.fingerprint)
            write_json(directory/'archived_run_analysis.json', archived)
        print(f'Base grid optimum: {exhaustive.best["comet_score"]:.9f}', flush=True)
        rows, trials, equal_total = [], [], []
        for space, experiment in experiments.items():
            encoder = encoders[space]
            for seed in settings['seeds']:
                budget = None
                pair_directories = {}
                for method in ('vqa', 'random'):
                    cfg = deepcopy(experiment)
                    cfg['seed'] = seed
                    trial_dir = directory/space/f'seed-{seed}'/method
                    evaluator = TrialEvaluator(oracle, trial_dir)
                    pair_directories[method] = trial_dir
                    write_json(trial_dir/'config.json', cfg)
                    print(f'{space} seed={seed} method={method} discrete search', flush=True)
                    search_start = time.perf_counter()
                    if method == 'vqa':
                        search = optimize(cfg, encoder, evaluator, trial_dir)
                        budget = len(evaluator.seen)
                    else:
                        search = random_search(encoder, evaluator, budget, seed, batch_size)
                    discrete_seconds = time.perf_counter()-search_start
                    if len(evaluator.seen) != budget:
                        raise RuntimeError('Paired distinct-composition budgets differ')
                    discrete = {'best': evaluator.best, 'counts': evaluator.stats(),
                                'seconds': discrete_seconds, 'comet_seconds': evaluator.comet_seconds,
                                'search': search}
                    write_json(trial_dir/'discrete.json', discrete)
                    row = metric_row(space, seed, method, 'discrete', evaluator.best, budget,
                                     evaluator.query_requests, discrete_seconds, evaluator.comet_seconds, budget)
                    rows.append(row)
                    append_csv(directory/'comparison.csv', row)
                    refinement = None
                    if settings['continuous']['enabled']:
                        print(f'{space} seed={seed} method={method} continuous refinement', flush=True)
                        refinement = refine(evaluator, cfg['search']['ratios']['bounds'], settings['continuous'], trial_dir)
                        row = metric_row(space, seed, method, 'with_continuous', evaluator.best, len(evaluator.seen),
                                         evaluator.query_requests, discrete_seconds+refinement['seconds'],
                                         evaluator.comet_seconds, budget)
                        rows.append(row)
                        append_csv(directory/'comparison.csv', row)
                    learning = None
                    if space == 'base' and method == 'vqa':
                        learning = exact_learning(encoder, cfg,
                            np.load(trial_dir/'initial_parameters.npy', allow_pickle=False),
                            np.load(trial_dir/'final_parameters.npy', allow_pickle=False), reference,
                            discrete['best']['composition_key'], settings['score_tolerance'])
                        write_json(trial_dir/'exact_learning.json', learning)
                    trial = {'space': space, 'seed': seed, 'method': method, 'discrete': discrete,
                             'continuous': refinement, 'final_best': evaluator.best,
                             'counts': evaluator.stats(),
                             'seconds': discrete_seconds+(refinement['seconds'] if refinement else 0),
                             'comet_seconds': evaluator.comet_seconds,
                             'exact_learning': {**learning,
                                 'initial': {k: v for k, v in learning['initial'].items() if k != 'probabilities'},
                                 'final': {k: v for k, v in learning['final'].items() if k != 'probabilities'}} if learning else None}
                    trials.append(trial)
                    write_json(trial_dir/'summary.json', trial)
                    print(f'{space} seed={seed} {method}: grid={discrete["best"]["comet_score"]:.6f}, '
                          f'final={evaluator.best["comet_score"]:.6f}, unique={len(evaluator.seen)}, '
                          f'seconds={trial["seconds"]:.2f}', flush=True)
                if settings['continuous']['enabled']:
                    records = {m: [json.loads(line) for line in (p/'candidates.jsonl').read_text().splitlines()]
                               for m, p in pair_directories.items()}
                    common = min(map(len, records.values()))
                    comparison = {'space': space, 'seed': seed, 'matched_total_unique': common,
                                  'definition': 'Post-hoc prefix at equal total unique evaluations; full batch time charged',
                                  **{m: {'best_score': records[m][common-1]['best_so_far_score'],
                                         'seconds_to_prefix': records[m][common-1]['elapsed_seconds']}
                                     for m in records}}
                    equal_total.append(comparison)
        summary = {'status': 'completed', 'score_description': SCORE_DESCRIPTION,
                   'model_fingerprint': oracle.fingerprint, 'seeds': settings['seeds'],
                   'protocol': {'discrete_budget': 'Per seed, random uses the VQA unique count, without replacement',
                                'cache': 'Fresh independent in-memory score cache per method/seed, reused only within its refinement',
                                'timing': 'Method wall times include search and recording; exclude shared load, preflight, warmup, exhaustive and exact analyses',
                                'order': 'VQA first to determine paired budget; shared model and all molecule features prewarmed',
                                'continuous_budget': 'Same top_k and maxfev per start; report actual unique counts and equal-total prefixes',
                                'numerics': 'Float32 COMET may differ slightly by batch layout; grid gaps use one reference table'},
                   'spaces': {n: {'qubits': e.num_qubits, 'compositions': e.cardinality,
                                  'ratio_tuples': len(e.ratios), 'candidate_counts': list(map(len, e.candidates))}
                              for n, e in encoders.items()},
                   'exhaustive': exhaustive_summary, 'archived_run_analysis': archived,
                   'trials': trials, 'aggregate': aggregate(rows, settings['seeds'], settings['score_tolerance']),
                   'equal_total_prefixes': equal_total,
                   'timing': {'total_seconds': time.perf_counter()-started, 'model_load_seconds': oracle.load_seconds,
                              'preflight_seconds': preflight['seconds'], 'warmup_seconds': warmup_seconds,
                              'exhaustive_seconds': exhaustive_summary['seconds'],
                              'comet_evaluation_seconds': oracle.evaluation_seconds},
                   'actual_comet_composition_evaluations_including_setup': oracle.compositions_evaluated,
                   'setup_comet_evaluations': {'preflight': 3, 'warmup': len(warmup), 'exhaustive': len(reference)}}
        write_json(directory/'summary.json', summary)
        from .benchmark_report import write_report
        write_report(summary, directory/'report.md', base['condition'])
        print(json.dumps({'status': 'completed', 'result': str(directory/'summary.json'),
                          'seconds': summary['timing']['total_seconds']}, indent=2), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        failure = {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                   'error_type': type(error).__name__, 'error': str(error), 'synthetic_fallback': False,
                   'seconds': time.perf_counter()-started}
        if directory:
            write_json(directory/'failure.json', failure)
            (directory/'traceback.txt').write_text(traceback.format_exc())
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 1
