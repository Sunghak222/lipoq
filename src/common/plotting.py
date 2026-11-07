"""Shared noninteractive figure export and isolated plotting failure handling."""
import os
import math
from pathlib import Path
import sys
import tempfile
import time

from .results import output_path, write_json

COLORS = ('#2367BD', '#DE7B24', '#159782', '#9467BD', '#C44768', '#677C8A')


def best_score_curve(records):
    values, best = [], -math.inf
    for record in records:
        score = float(record['comet_score'])
        if not math.isfinite(score):
            raise ValueError('Nonfinite candidate score')
        best = max(best, score)
        values.append(best)
    return values


def prepare_matplotlib():
    # Reuse the per-user temporary font cache instead of scanning fonts per run.
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'lipoq-matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.labelcolor': '#273B50', 'text.color': '#172B40',
                         'xtick.color': '#536679', 'ytick.color': '#536679',
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#C7D1DC', 'axes.titleweight': 'bold',
                         'svg.fonttype': 'path', 'savefig.facecolor': 'white'})
    return matplotlib, plt


def save_figure(fig, output, name):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        target = output/f'{name}.{extension}'
        temp = output/f'.{name}.tmp.{extension}'
        fig.savefig(temp, dpi=200, format=extension)
        temp.replace(target)


def export_automatically(directory, generate):
    """A plot failure must not discard a successfully completed experiment."""
    started = time.perf_counter()
    try:
        status = {'status': 'completed', **generate()}
    except Exception as error:
        status = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
        print(f'Experiment saved; automatic plot failed: {error}', file=sys.stderr, flush=True)
    status['seconds'] = time.perf_counter()-started
    write_json(output_path(directory, 'plot_status.json'), status)
    return status
