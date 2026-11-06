"""Portable English report generated exclusively from measured benchmark results."""
from pathlib import Path


def write_report(summary, destination, condition):
    seed_count = len(summary['seeds'])
    refinements = [t['continuous'] for t in summary['trials'] if t['continuous']]
    refinement_note = (f"Both methods use identical Powell settings for refinement of their top {refinements[0]['settings']['top_k']} distinct lipid identities. Actual unique calls can differ due to caching and termination."
                       if refinements else 'Continuous refinement was disabled for this suite.')
    lines = ['# LipoQ benchmark results', '',
             f"All scores below are frozen pretrained COMET predictions for {condition['cell']}, not measured efficacy.",
             f'Results describe this checkpoint, preprocessing seed, search bounds, and {seed_count} configured search seeds.', '',
             '## Protocol', '',
             '- Independent empty score caches per method and seed; refinement reuses only its own search cache.',
             '- Random search samples unique compositions uniformly without replacement and matches the VQA unique count for each seed.',
             '- Method times exclude shared model loading, preflight, warmup, exhaustive enumeration, and exact statevector analysis.',
             '- '+refinement_note,
             f"- Exact statistics use the full base score table only after search. Float32 batch-layout differences are handled with a {summary['exhaustive']['score_tolerance']:g} score tolerance.", '',
             '## Exhaustive base grid', '']
    exhaustive = summary['exhaustive']
    lines += [f"Evaluated **{exhaustive['evaluations']}** unique formulations. Grid maximum: **{exhaustive['best']['comet_score']:.9f}**.", '',
              '| Component | Molecule | Molar fraction |', '|---|---|---:|']
    for c in exhaustive['best']['components']:
        lines.append(f"| {c['group']} | {c['name']} | {c['molar_fraction']:.6f} |")
    archived = summary.get('archived_run_analysis')
    if archived:
        lines += ['', '## Previously saved VQA run', '',
                  f"Seed {archived['seed']}; gap between its best sampled composition and the grid optimum: **{archived['gap_to_grid_optimum']:.9f}**.", '',
                  '| Exact statistic | Before training | After training |', '|---|---:|---:|',
                  f"| Mean COMET score | {archived['initial']['mean_comet_score']:.9f} | {archived['final']['mean_comet_score']:.9f} |",
                  f"| Probability of the grid maximum | {100*archived['initial']['probability_of_grid_maximum']:.6f}% | {100*archived['final']['probability_of_grid_maximum']:.6f}% |"]
    lines += ['', f'## Exact learning diagnostics for {seed_count} seeds (base grid)', '',
              '| Seed | Exact initial mean | Exact final mean | Initial optimum probability | Final optimum probability | Grid regret |',
              '|---|---:|---:|---:|---:|---:|']
    for trial in summary['trials']:
        analysis = trial['exact_learning']
        if analysis:
            lines.append(f"| {trial['seed']} | {analysis['initial']['mean_comet_score']:.6f} | {analysis['final']['mean_comet_score']:.6f} | "
                         f"{100*analysis['initial']['probability_of_grid_maximum']:.6g}% | {100*analysis['final']['probability_of_grid_maximum']:.6g}% | {analysis['gap_to_grid_optimum']:.6f} |")
    lines += ['', '## Matched discrete search and continuous refinement', '',
              'Values are mean ± sample standard deviation over the configured seeds. Continuous rows include discrete search cost.', '',
              '| Space | Stage | Method | Best score | Unique COMET evaluations | Wall time (s) | COMET time (s) |',
              '|---|---|---|---:|---:|---:|---:|']
    for group in summary['aggregate']['groups']:
        def value(key, places):
            v = group[key]
            return f"{v['mean']:.{places}f} ± {v['std']:.{places}f}"
        lines.append(f"| {group['space']} | {group['stage']} | {group['method']} | {value('best_score', 6)} | "
                     f"{value('unique_comet_evaluations', 1)} | {value('seconds', 2)} | {value('comet_seconds', 2)} |")
    lines += ['', '| Space | Stage | VQA wins | Ties | Random wins |', '|---|---|---:|---:|---:|']
    for pair in summary['aggregate']['paired_comparisons']:
        lines.append(f"| {pair['space']} | {pair['stage']} | {pair['vqa_wins']} | {pair['ties']} | {pair['random_wins']} |")
    lines += ['', '## Per-seed measurements', '',
              '| Space | Seed | Method | Grid evaluations | Grid best | Grid time (s) | Total evaluations | Final best | Total time (s) |',
              '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for t in summary['trials']:
        d = t['discrete']
        lines.append(f"| {t['space']} | {t['seed']} | {t['method']} | {d['counts']['unique_comet_evaluations']} | {d['best']['comet_score']:.6f} | "
                     f"{d['seconds']:.2f} | {t['counts']['unique_comet_evaluations']} | {t['final_best']['comet_score']:.6f} | {t['seconds']:.2f} |")
    lines += ['', '## Equal-total-evaluation prefixes after refinement', '',
              'Post-hoc prefixes use the smaller total unique count in each pair; a batched evaluation is charged its full elapsed time. These are diagnostics, not separately rerun stopping policies.', '',
              '| Space | Seed | Equal total evaluations | VQA best | Random best | VQA time to prefix (s) | Random time to prefix (s) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for p in summary['equal_total_prefixes']:
        lines.append(f"| {p['space']} | {p['seed']} | {p['matched_total_unique']} | {p['vqa']['best_score']:.6f} | {p['random']['best_score']:.6f} | "
                     f"{p['vqa']['seconds_to_prefix']:.2f} | {p['random']['seconds_to_prefix']:.2f} |")
    if refinements:
        lines += ['', '## Best formulations found after continuous refinement', '',
                  'These are the best found across the recorded trials; they are not certified continuous optima.', '',
                  '| Space | Best score | Ionizable | Phospholipid | Sterol | PEG | Molar fractions (same order) |',
                  '|---|---:|---|---|---|---|---|']
        for space in ('base', 'expanded'):
            best = max((t['final_best'] for t in summary['trials'] if t['space'] == space), key=lambda b: b['comet_score'])
            names = ' | '.join(c['name'] for c in best['components'])
            fractions = ' / '.join(f"{c['molar_fraction']:.10f}" for c in best['components'])
            lines.append(f"| {space} | {best['comet_score']:.9f} | {names} | {fractions} |")
        solvers = [s for r in refinements for s in r['solvers']]
        lines += ['', f"{sum(s['success'] for s in solvers)} of {len(solvers)} Powell starts reported convergence; "
                  f"{sum(not s['success'] for s in solvers)} ended without convergence. See each saved termination message before interpreting a solution."]
    lines += ['', '## Scope and accounting', '',
              f"- Base space: {summary['spaces']['base']['compositions']} formulations; expanded space: {summary['spaces']['expanded']['compositions']} formulations.",
              '- Only the base discrete grid was exhaustively evaluated. Expanded and continuous maxima are best found, not certified global optima.',
              '- Powell is capped by its configured evaluation budget; individual success/termination messages are retained in continuous.json.',
              '- More optimizer calls, different budgets, folds, conformers, or seeds can change the outcome.',
              '- Expanded VQA encoding uses modulo aliases for non-power-of-two fields; random composition sampling is uniform.',
              '- Methods run sequentially on a shared, prewarmed CPU model. VQA runs first to establish the paired budget; times are not hardware-independent estimates.',
              f"- Entire suite: {summary['timing']['total_seconds']:.2f} s; actual COMET input evaluations including setup: {summary['actual_comet_composition_evaluations_including_setup']}.",
              f"- Setup evaluations: {summary['setup_comet_evaluations']}; total COMET time: {summary['timing']['comet_evaluation_seconds']:.2f} s.",
              f"- Checkpoint/model fingerprint: `{summary['model_fingerprint']}`.",
              '- These comparisons do not establish quantum advantage, statistical superiority, or biological efficacy.', '']
    Path(destination).write_text('\n'.join(lines))
