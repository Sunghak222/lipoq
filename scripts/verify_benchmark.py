"""Audit a completed benchmark from its saved measurements, without COMET calls."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lnp_vqa.benchmark_metrics import exact_learning
from lnp_vqa.continuous import BoundedSimplex
from lnp_vqa.encoding import CompositionEncoder


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verify(path):
    summary = json.loads((path/'summary.json').read_text())
    settings = json.loads((path/'config.json').read_text())
    experiments = json.loads((path/'experiments.json').read_text())
    assert summary['status'] == 'completed'
    exhaustive = read_jsonl(path/'exhaustive'/'candidates.jsonl')
    reference = {c['composition_key']: c['comet_score'] for c in exhaustive}
    base_encoder = CompositionEncoder(experiments['base']['search'])
    assert set(reference) == {c.key for c in base_encoder.compositions()}
    assert len(reference) == len(exhaustive) == summary['exhaustive']['evaluations']
    assert max(reference.values()) == summary['exhaustive']['best']['comet_score']
    actual_calls = sum(summary['setup_comet_evaluations'].values())
    paired_budgets = {}
    for trial in summary['trials']:
        cfg = experiments[trial['space']]
        encoder = CompositionEncoder(cfg['search'])
        feasible = BoundedSimplex(cfg['search']['ratios']['bounds'])
        directory = path/trial['space']/f"seed-{trial['seed']}"/trial['method']
        records = read_jsonl(directory/'candidates.jsonl')
        scores = {r['composition_key']: r['comet_score'] for r in records}
        assert len(scores) == len(records) == trial['counts']['unique_comet_evaluations']
        actual_calls += len(scores)
        assert max(scores.values()) == trial['final_best']['comet_score']
        grid = [r for r in records if r['stage'] == 'discrete']
        assert len(grid) == trial['discrete']['counts']['unique_comet_evaluations']
        assert max(r['comet_score'] for r in grid) == trial['discrete']['best']['comet_score']
        assert trial['seconds'] >= trial['comet_seconds'] >= 0
        pair = (trial['space'], trial['seed'])
        paired_budgets.setdefault(pair, []).append(len(grid))
        best = -math.inf
        for index, row in enumerate(records, 1):
            assert row['unique_evaluations'] == index
            best = max(best, row['comet_score'])
            assert row['best_so_far_score'] == best
            feasible.validate([c['molar_fraction'] for c in row['components']])
        if trial['method'] == 'random':
            indices = np.random.default_rng(trial['seed']).choice(encoder.cardinality, len(grid), replace=False)
            assert [encoder.composition_at(i).key for i in indices] == [r['composition_key'] for r in grid]
        else:
            evaluations = read_jsonl(directory/'evaluations.jsonl')
            assert len(evaluations) == 3*cfg['optimizer']['iterations']+2
            for row in evaluations:
                assert sum(row['counts'].values()) == cfg['circuit']['shots']
                mean = sum(n*scores[encoder.decode(bits).key] for bits, n in row['counts'].items())/cfg['circuit']['shots']
                assert math.isclose(mean, row['mean_comet_score'], abs_tol=1e-12, rel_tol=0)
            if trial['space'] == 'base':
                exact = exact_learning(encoder, cfg,
                    np.load(directory/'initial_parameters.npy', allow_pickle=False),
                    np.load(directory/'final_parameters.npy', allow_pickle=False),
                    reference, trial['discrete']['best']['composition_key'], settings['score_tolerance'])
                saved = json.loads((directory/'exact_learning.json').read_text())
                assert exact == saved
        if trial['continuous']:
            continuous = trial['continuous']
            assert len(records)-len(grid) == continuous['additional_unique_comet_evaluations']
            assert sum(r['nfev'] for r in continuous['solvers']) == continuous['objective_requests']
            assert all(r['nfev'] <= settings['continuous']['maxfev_per_start'] for r in continuous['solvers'])
            assert trial['final_best']['comet_score'] >= trial['discrete']['best']['comet_score']
            for solver in continuous['solvers']:
                feasible.validate(solver['returned_fractions'])
    assert len(summary['trials']) == 4*len(settings['seeds'])
    assert all(len(counts) == 2 and counts[0] == counts[1] for counts in paired_budgets.values())
    assert actual_calls == summary['actual_comet_composition_evaluations_including_setup']
    for prefix in summary['equal_total_prefixes']:
        pairs = {m: read_jsonl(path/prefix['space']/f"seed-{prefix['seed']}"/m/'candidates.jsonl') for m in ('vqa', 'random')}
        budget = min(map(len, pairs.values()))
        assert prefix['matched_total_unique'] == budget
        for method, records in pairs.items():
            assert prefix[method]['best_score'] == max(r['comet_score'] for r in records[:budget])
    return {'status': 'verified', 'trials': len(summary['trials']), 'matched_pairs': len(paired_budgets),
            'base_reference_compositions': len(reference), 'actual_comet_evaluations': actual_calls}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark_directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.benchmark_directory), indent=2))
