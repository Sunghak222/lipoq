"""Method-independent matched-budget reports; never execute a model or optimizer."""
import json
from pathlib import Path

from .results import artifact_path, method_name, output_path, read_jsonl, write_csv, write_json
from .plotting import best_score_curve, prepare_matplotlib, save_figure


def write_comparison(reference, summary, directory):
    baseline = reference['summary']
    if baseline['model_fingerprint'] != summary['model_fingerprint']:
        raise ValueError('Cannot compare different COMET fingerprints')
    config = json.loads(artifact_path(directory, 'config.json').read_text())
    for field in ('condition', 'search'):
        if config[field] != reference['config'][field]:
            raise ValueError(f'Cannot compare different {field} settings')
    rows = []
    for method, seed, result in [(method_name(baseline), reference['config']['seed'], baseline),
                                 (method_name(summary), summary['seed'], summary)]:
        counts = result['counts']
        rows.append({'method': method, 'seed': seed, 'best_score': result['best']['comet_score'],
                     'unique_compositions': counts['unique_sampled_compositions'],
                     'new_comet_evaluations': counts['unique_comet_evaluations'],
                     'persistent_cache_hits': counts.get('unique_loaded_from_persistent_cache', 0),
                     'search_seconds': result['timing'].get('search_seconds', result.get('optimization_seconds', result.get('search_seconds'))),
                     'search_comet_seconds': result['timing']['search_comet_seconds'],
                     'total_seconds_before_plotting': result['timing']['total_seconds'],
                     'timing_directly_comparable': False})
    if rows[0]['unique_compositions'] != rows[1]['unique_compositions']:
        raise ValueError('Cannot compare unequal completed composition budgets.')
    comparison = {'status': 'completed', 'matched_unique_budget': reference['budget'],
                  'reference_run': str(reference['path']), 'rows': rows,
                  'reference_minus_candidate_best': rows[0]['best_score']-rows[1]['best_score'],
                  'score_tolerance': 1e-5,
                  'protocol': {'scores': 'Same frozen COMET fingerprint and cell/process conditions.',
                               'budget': 'All distinct encountered compositions, including cached scores.',
                               'timing': 'Descriptive only; cache and execution conditions are not controlled.',
                               'numerics': 'Float32 batch layout can cause small score differences.',
                               'scope': 'One matched pair; no statistical or quantum-advantage claim.'}}
    write_json(output_path(directory, 'comparison.json'), comparison)
    write_csv(output_path(directory, 'comparison.csv'), rows)
    return comparison


def plot_comparison(reference, directory):
    directory = Path(directory)
    comparison = json.loads(artifact_path(directory, 'comparison.json').read_text())
    candidate_records = read_jsonl(directory/'candidates.jsonl')
    output = directory/'plots'
    output.mkdir(parents=True, exist_ok=True)
    _, plt = prepare_matplotlib()
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.subplots_adjust(left=.09, right=.96, top=.77, bottom=.22)
    labels = [row['method'].replace('_', ' ').upper() for row in comparison['rows']]
    fig.text(.09, .94, f'LipoQ | {labels[0]} vs {labels[1]}', fontsize=22, weight='bold', color='#172B40')
    fig.text(.09, .88, f"{reference['config']['condition']['cell']}  /  {reference['encoder'].cardinality:,} compositions"
             f"  /  matched budget: {reference['budget']:,} distinct compositions", fontsize=11, color='#536679')
    for row, records, color in zip(comparison['rows'], [reference['records'], candidate_records], ['#2367BD', '#DE7B24']):
        values = best_score_curve(records)
        ax.step(range(1, len(values)+1), values, where='post', color=color, lw=2.4,
                label=f"{row['method'].upper()} | seed {row['seed']} | best {values[-1]:.4f}")
        ax.scatter([len(values)], [values[-1]], color=color, s=40, zorder=4)
    ax.legend(loc='lower right', frameon=False, fontsize=11)
    ax.set_xlabel('Distinct compositions encountered (cached scores included)', fontsize=11, labelpad=12)
    ax.set_ylabel('Best predicted COMET score found', fontsize=11)
    ax.grid(axis='y', color='#E5EAF0')
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    ax.tick_params(labelsize=10)
    fig.text(.09, .105, 'COMET predictions, not measured efficacy. Within-batch order follows each saved candidate log.', fontsize=10, color='#536679')
    fig.text(.09, .06, 'Matched candidate budget; runtime is not directly comparable without controlled execution conditions.', fontsize=10, color='#536679')
    save_figure(fig, output, 'comparison')
    plt.close(fig)
    return {'output': str(output), 'figures': ['comparison.png', 'comparison.svg']}
