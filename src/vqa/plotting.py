"""Export scientific figures from saved runs, without model or circuit execution."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import sys

from common.plotting import COLORS, best_score_curve, prepare_matplotlib, save_figure
from common.results import artifact_path, method_name, snapshot_lines, write_csv, write_json


def read_run(path):
    config = json.loads((artifact_path(path, 'config.json')).read_text())
    text, history_hash = snapshot_lines(artifact_path(path, 'history.csv'))
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f'No complete history rows yet: {path}')
    iterations = [int(r['iteration']) for r in rows]
    if any(a >= b for a, b in zip(iterations, iterations[1:])):
        raise ValueError(f'Non-increasing iteration numbers: {path}')
    for row in rows:
        for key in ('mean_comet_score', 'best_so_far_score'):
            if not math.isfinite(float(row[key])):
                raise ValueError(f'Nonfinite {key}: {path}')
    summary_path, failure_path = path/'summary.json', path/'failure.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    completed = summary.get('status') == 'completed' and summary.get('iterations') == iterations[-1]
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else {}
    status = 'completed' if completed else failure.get('status', summary.get('status', 'incomplete snapshot'))
    candidate_text, candidate_hash = snapshot_lines(artifact_path(path, 'candidates.jsonl'))
    candidates = [json.loads(line) for line in candidate_text.splitlines() if line.strip()]
    keys = [r['composition_key'] for r in candidates]
    if not candidates or len(keys) != len(set(keys)):
        raise ValueError(f'Missing or duplicate unique candidate records: {path}')
    running_best = best_score_curve(candidates)
    encoding_path = artifact_path(path, 'encoding.json')
    encoding = json.loads(encoding_path.read_text()) if encoding_path.exists() else {}
    model_path = artifact_path(path, 'model.json')
    model = json.loads(model_path.read_text()) if model_path.exists() else {}
    return {'path': path, 'config': config, 'rows': rows, 'iterations': iterations,
            'summary': summary, 'status': status, 'completed': completed,
            'candidates': candidates, 'best_curve': running_best, 'encoding': encoding,
            'model': model, 'history_sha256': history_hash, 'candidates_sha256': candidate_hash}


def discover(source, complete_only=False):
    source = Path(source)
    if artifact_path(source, "history.csv").is_file():
        paths = [source]
    else:
        paths = [p.parent.parent if p.parent.name == "metrics" else p.parent
                 for p in source.rglob("history.csv")]
    if not paths:
        raise ValueError(f"No history.csv found under {source}; run an experiment first.")
    selected = {}
    for path in sorted(set(paths), key=lambda p: (p.name, str(p))):
        config_path = artifact_path(path, "config.json")
        if not config_path.is_file() or not (path / "candidates.jsonl").is_file():
            continue
        config = json.loads(config_path.read_text())
        summary_path = path / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if method_name(summary) not in (None, "vqa") or "circuit" not in config:
            continue
        if complete_only and summary.get("status") != "completed":
            continue
        selected[config["seed"]] = path
    runs = [read_run(path) for seed, path in sorted(selected.items())]
    if not runs:
        raise ValueError("No matching VQA runs; omit --complete-only to plot incomplete snapshots.")
    reference = runs[0]["config"]
    reference_model = runs[0]["model"]
    for run in runs[1:]:
        for field in ("condition", "search", "circuit", "optimizer"):
            if run["config"][field] != reference[field]:
                raise ValueError(f"Cannot overlay different {field} settings; choose a narrower folder.")
        if reference_model and run["model"] and run["model"] != reference_model:
            raise ValueError("Cannot overlay different COMET model/preprocessing provenance.")
    return runs


def trailing_mean(values, window):
    # Exclude the initial (iteration-zero) sample: every smooth value uses only
    # the previous `window` monitored updates, and never uses future samples.
    return [math.nan if i < window else math.fsum(values[i-window+1:i+1])/window
            for i in range(len(values))]


def render(runs, output, window=5):
    if window < 1:
        raise ValueError('--window must be positive')
    output.mkdir(parents=True, exist_ok=True)
    matplotlib, plt = prepare_matplotlib()
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    cfg = runs[0]['config']
    space_size = runs[0]['encoding'].get('unique_search_space_size', '?')
    cell = cfg['condition']['cell']
    shots = cfg['circuit']['shots']
    partial = any(not r['completed'] for r in runs)
    status_text = 'INCOMPLETE SNAPSHOT' if partial else 'COMPLETED RUNS'
    seed_handles = [Line2D([0], [0], color=COLORS[i % len(COLORS)], lw=2.5,
                          label=f"Seed {r['config']['seed']} ({r['status']})")
                    for i, r in enumerate(runs)]

    def style_axis(ax):
        ax.grid(axis='y', color='#E5EAF0', lw=.8)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax.margins(x=.025, y=.12)

    def save(fig, name):
        save_figure(fig, output, name)
        plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.4), sharex=True,
                             gridspec_kw={'height_ratios': [1.65, 1]})
    fig.subplots_adjust(left=.085, right=.97, top=.80, bottom=.20, hspace=.28)
    fig.text(.085, .956, 'LipoQ | VQA learning curves', fontsize=22, weight='bold')
    fig.text(.085, .915, f'{cell}  /  {space_size} compositions  /  {shots} shots per evaluation', color='#536679')
    fig.text(.97, .956, status_text, fontsize=10, ha='right', color='#A55C00' if partial else '#177B61')
    fig.legend(handles=seed_handles, loc='upper left', bbox_to_anchor=(.078, .895),
               frameon=False, ncol=min(3, len(runs)), fontsize=9)
    axes[0].set_title('A   Mean score of generated candidates', loc='left', fontsize=13)
    axes[1].set_title('B   Best candidate discovered so far', loc='left', fontsize=13)
    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        x = run['iterations']
        means = [float(r['mean_comet_score']) for r in run['rows']]
        errors = [float(r['shot_standard_error']) if r.get('shot_standard_error') else math.nan for r in run['rows']]
        bests = [float(r['best_so_far_score']) for r in run['rows']]
        axes[0].plot(x, means, color=color, lw=1.1, alpha=.55, marker='o', markersize=2)
        axes[0].fill_between(x, np.array(means)-errors, np.array(means)+errors, color=color, alpha=.10, linewidth=0)
        axes[0].plot(x, trailing_mean(means, window), color=color, lw=2.7)
        axes[1].step(x, bests, color=color, lw=2.3, where='post')
        axes[1].plot(x[-1], bests[-1], 'o', color=color, markersize=4)
        if run['completed']:
            final = run['summary'].get('final_mean_comet_score')
            if final is not None:
                axes[0].scatter([x[-1]], [final], marker='D', s=50, color=color, edgecolor='white', zorder=5)
            final_best = run['summary'].get('best', {}).get('comet_score')
            if final_best is not None:
                axes[1].scatter([x[-1]], [final_best], marker='D', s=42, color=color, edgecolor='white', zorder=5)
            if run['summary'].get('termination_reason') == 'plateau':
                for ax in axes:
                    ax.axvline(x[-1], color=color, ls=':', lw=1, alpha=.7)
    for ax in axes:
        ax.set_ylabel('Predicted COMET score')
        style_axis(ax)
    axes[1].set_xlabel('SPSA update (0 = initialization)')
    fig.text(.085, .102, f'Thin lines: sampled means   |   Thick lines: {window}-update trailing means   |   Shading: ±1 shot standard error', fontsize=9, color='#536679')
    fig.text(.085, .071, 'Diamonds: final returned circuit / final best, when complete   |   Dotted vertical line: plateau stop', fontsize=9, color='#536679')
    fig.text(.085, .040, 'COMET predictions, not measured efficacy. Error bands describe sampling noise, not model uncertainty.', fontsize=9, color='#536679')
    save(fig, 'learning_curves')

    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    fig.subplots_adjust(left=.085, right=.97, top=.75, bottom=.18)
    fig.text(.085, .94, 'LipoQ | Candidate discovery', fontsize=22, weight='bold')
    fig.text(.085, .885, f'{cell}  /  {space_size} possible compositions  /  higher scores are better', color='#536679')
    fig.text(.97, .94, status_text, fontsize=10, ha='right', color='#A55C00' if partial else '#177B61')
    fig.legend(handles=seed_handles, loc='upper left', bbox_to_anchor=(.078, .86), frameon=False, ncol=min(3, len(runs)), fontsize=9)
    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        values = run['best_curve']
        ax.step(range(1, len(values)+1), values, where='post', color=color, lw=2.5)
        ax.scatter([len(values)], [values[-1]], color=color, s=35, zorder=4)
    ax.set_xlabel('Unique compositions encountered in this run (including cached scores)')
    ax.set_ylabel('Best predicted COMET score found')
    style_axis(ax)
    fig.text(.085, .095, 'Includes perturbation, monitoring and final samples. Cached candidates count once per run.', fontsize=9, color='#536679')
    fig.text(.085, .052, 'Within-batch candidate order follows the saved log. This is not a curve of fresh COMET calls or elapsed time.', fontsize=9, color='#536679')
    save(fig, 'discovery_curve')

    metrics = []
    for run in runs:
        rows, candidates = run['rows'], run['candidates']
        summary = run['summary'] if run['completed'] else {}
        counts = summary.get('counts', {})
        metrics.append({'seed': run['config']['seed'], 'status': run['status'], 'last_update': run['iterations'][-1],
                        'initial_measured_mean': float(rows[0]['mean_comet_score']),
                        'last_monitored_mean': float(rows[-1]['mean_comet_score']),
                        'final_returned_circuit_mean': summary.get('final_mean_comet_score'),
                        'best_found_score': run['best_curve'][-1],
                        'unique_compositions': len(candidates),
                        'new_comet_evaluations': counts.get('unique_comet_evaluations', sum(not c.get('from_persistent_cache', False) for c in candidates)),
                        'termination_reason': summary.get('termination_reason', ''),
                        'total_seconds': summary.get('timing', {}).get('total_seconds'),
                        'source_run': str(run['path'])})
    write_csv(output/'run_metrics.csv', metrics)
    manifest = {'selection': 'Latest timestamped run directory per seed; no selection by score',
                'smoothing': f'Trailing {window} monitored updates; initial point excluded',
                'uncertainty': '±1 per-evaluation shot standard error; not model or between-seed uncertainty',
                'incomplete_snapshot': partial, 'matplotlib_version': matplotlib.__version__,
                'inputs': [{'run': str(r['path']), 'history_sha256': r['history_sha256'],
                            'candidates_sha256': r['candidates_sha256'], 'status': r['status'],
                            'last_update': r['iterations'][-1]} for r in runs]}
    write_json(output/'plot_manifest.json', manifest)
    return {'output': str(output), 'runs': len(runs), 'incomplete_snapshot': partial,
            'figures': ['learning_curves.png', 'learning_curves.svg', 'discovery_curve.png', 'discovery_curve.svg']}


def plot_completed_run(directory):
    directory = Path(directory)
    return render([read_run(directory)], directory/'plots')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, nargs='?', default=Path('results/lnp_search'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--complete-only', action='store_true')
    args = parser.parse_args()
    try:
        source = args.source.resolve()
        runs = discover(source, args.complete_only)
        result = render(runs, (args.output or source/'plots').resolve(), args.window)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Plot error: {error}', file=sys.stderr)
        return 1
