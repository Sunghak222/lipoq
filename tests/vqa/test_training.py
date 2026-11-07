"""Validate optimizer accounting and simulator sampling, using test-only objectives."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from vqa.config import load_config
from common.encoding import CompositionEncoder
from vqa.circuit import QuantumSampler
from vqa.optimizer import optimize
from common.results import artifact_path


class TestObjective:
    __test__ = False

    def __init__(self):
        self.best = {'comet_score': -1.0}
        self.seen = set()
        self.unique_comet_evaluations = 0

    def evaluate_counts(self, counts, encoder):
        self.seen.update(counts)
        score = sum(int(k, 2)*n for k,n in counts.items())/sum(counts.values())
        self.best = {'comet_score': max(self.best['comet_score'], max(map(lambda k: int(k,2), counts)))}
        return {'mean_comet_score': score, 'shot_standard_error': 1.0}


def test_paired_sampling_reuses_randomness_without_advancing_rng():
    a,b=QuantumSampler(3,1,17),QuantumSampler(3,1,17)
    theta=np.linspace(.1,2,6)
    counts,seed=a.sample(theta,64)
    assert a.sample(theta,64,seed)==(counts,seed)
    assert b.sample(theta,64)==(counts,seed)
    assert a.sample(theta,64)==b.sample(theta,64)


def test_resamplings_phases_averaging_and_stop_reason(cfg,tmp_path):
    cfg['optimizer'].update(iterations=3,resamplings=2,paired_sampling=True,
                            trust_region=True,last_avg=2,initialization='uniform')
    cfg['circuit']['shots']=32
    encoder=CompositionEncoder(cfg['search'])
    result=optimize(cfg,encoder,TestObjective(),tmp_path)
    records=[json.loads(s) for s in (artifact_path(tmp_path, 'evaluations.jsonl')).read_text().splitlines()]
    assert result['objective_calls']==result['expected_objective_calls']==17
    assert result['termination_reason']=='iteration_limit'
    assert result['convergence_certified'] is False
    assert records[0]['phase']=='initial' and records[-1]['phase']=='final'
    for iteration in range(1,4):
        block=[r for r in records if r['iteration']==iteration and r['phase']!='final']
        assert [r['phase'] for r in block]==['perturb_plus','perturb_minus']*2+['monitor']
        assert block[0]['measurement_seed']==block[1]['measurement_seed']
        assert block[2]['measurement_seed']==block[3]['measurement_seed']
    monitors=[r['theta'] for r in records if r['phase']=='monitor']
    assert np.allclose(np.load(artifact_path(tmp_path, 'final_parameters.npy')),np.mean(monitors[-2:],axis=0))
    theta0=np.load(artifact_path(tmp_path, 'initial_parameters.npy'))
    assert np.allclose(QuantumSampler(encoder.num_qubits,1,0).probabilities(theta0),1/2**encoder.num_qubits)


@pytest.mark.parametrize('field,value',[('resamplings',0),('last_avg',1000),('paired_sampling',1),('initialization','invalid')])
def test_invalid_training_options(cfg,tmp_path,field,value):
    cfg['optimizer'][field]=value
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):load_config(path)
