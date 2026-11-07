"""Small control-flow tests. Test oracle values never enter real experiment outputs."""
import json
from pathlib import Path

import pytest

from common.encoding import CompositionEncoder
from common.plotting import export_automatically
from common.results import append_jsonl, artifact_path, write_json
from common import experiment
from random_search import cli
from random_search.reference import load_reference


@pytest.fixture
def archive(cfg, tmp_path):
    directory = tmp_path/'seed-7'/'run-20261106-000000-test'
    directory.mkdir(parents=True)
    cfg['output']['directory'] = str(directory.parent)  # Legacy per-seed output root.
    encoder = CompositionEncoder(cfg['search'])
    catalog = {i: {'name': str(i), 'smiles': 'UNIT-TEST-ONLY'} for ids in encoder.candidates for i in ids}
    for i in range(7):
        c = encoder.composition_at(i)
        append_jsonl(directory/'candidates.jsonl', {'composition_key': c.key, 'comet_score': float(i),
                                                  **c.as_dict(catalog)})
    write_json(directory/'config.json', cfg)
    write_json(directory/'summary.json', {
        'status': 'completed', 'command': 'run', 'model_fingerprint': 'UNIT-TEST-ONLY',
        'best': {'comet_score': 6.}, 'counts': {'unique_sampled_compositions': 7,
        'unique_comet_evaluations': 5, 'unique_loaded_from_persistent_cache': 2},
        'optimization_seconds': 1., 'timing': {'search_comet_seconds': .5, 'total_seconds': 2.}})
    return directory, catalog


def test_reference_budget_validation_and_dry_run_without_model(archive, tmp_path, monkeypatch, capsys):
    directory, _ = archive
    assert load_reference(directory.parent)['path'] == directory
    monkeypatch.setattr(experiment, 'CometOracle', lambda *a: pytest.fail('Dry run must not load COMET'))
    output = tmp_path/'should-not-exist'
    assert cli.run_random(directory.parent, output, dry_run=True) == 0
    assert not output.exists()
    capsys.readouterr()
    assert cli.run_random(directory.parent, dry_run=True) == 0
    plan = json.loads(capsys.readouterr().out)
    assert Path(plan['output_root']) == tmp_path/'random'/'seed-7'
    summary = json.loads((directory/'summary.json').read_text())
    summary['counts']['unique_sampled_compositions'] += 1
    write_json(directory/'summary.json', summary)
    with pytest.raises(ValueError, match='budget'):
        load_reference(directory)


def test_random_only_completes_matched_budget_and_exports_real_files(archive, tmp_path, monkeypatch):
    directory, catalog = archive
    calls = []

    class TestOracle:
        fingerprint = 'UNIT-TEST-ONLY'
        provenance = {'source': 'UNIT-TEST-ONLY'}
        evaluation_seconds = 0.
        compositions_evaluated = 0
        load_seconds = 0.

        def __init__(self, *args):
            self.catalog = catalog

        def validate_search(self, encoder):
            return {'status': 'test-only'}

        def preflight(self):
            self.compositions_evaluated += 3
            return {'status': 'test-only'}

        def predict(self, compositions):
            calls.extend(c.key for c in compositions)
            self.compositions_evaluated += len(compositions)
            return [float(c.lipid_ids[0]) for c in compositions]

    monkeypatch.setattr(experiment, 'CometOracle', TestOracle)
    output = tmp_path/'random-unit-test'
    assert cli.run_random(directory, output) == 0
    run = next(output.rglob('run-*'))
    summary = json.loads((run/'summary.json').read_text())
    assert len(calls) == len(set(calls)) == summary['counts']['unique_comet_evaluations'] == 7
    assert summary['preflight_composition_evaluations'] == 3
    comparison = json.loads(artifact_path(run, 'comparison.json').read_text())
    assert {r['unique_compositions'] for r in comparison['rows']} == {7}
    assert all(not r['timing_directly_comparable'] for r in comparison['rows'])
    assert json.loads(artifact_path(run, 'plot_status.json').read_text())['status'] == 'completed'
    assert (run/'plots/comparison.png').stat().st_size > 1000
    assert (run/'plots/comparison.svg').stat().st_size > 1000
    assert not (run/'failure.json').exists()


def test_mismatched_model_rejected_before_any_prediction(archive, tmp_path, monkeypatch):
    directory, _ = archive

    class WrongOracle:
        fingerprint = 'OTHER-MODEL'
        evaluation_seconds = 0.

        def __init__(self, *args):
            pass

        def preflight(self):
            pytest.fail('Must reject mismatch before preflight or search')

    monkeypatch.setattr(experiment, 'CometOracle', WrongOracle)
    output = tmp_path/'mismatch'
    assert cli.run_random(directory, output) == 1
    run = next(output.rglob('run-*'))
    assert not (run/'candidates.jsonl').exists()
    assert 'fingerprint' in json.loads((run/'summary.json').read_text())['error']


def test_plot_failure_preserves_completed_summary(tmp_path):
    write_json(tmp_path/'summary.json', {'status': 'completed'})

    def broken_plot():
        raise ImportError('test-only missing plotting dependency')

    status = export_automatically(tmp_path, broken_plot)
    assert status['status'] == 'failed'
    assert json.loads((tmp_path/'summary.json').read_text())['status'] == 'completed'
