from vqa.early_stopping import PlateauStopper
from common.encoding import CompositionEncoder
from vqa.optimizer import optimize


def test_plateau_waits_for_burn_in_and_patience():
    stopper=PlateauStopper({'enabled':True,'min_iterations':4,'window':2,'patience':2,'min_delta':.02,'confidence_z':1})
    flags=[stopper.observe(i,1.0,.01) for i in range(1,7)]
    assert flags==[False]*5+[True]
    assert stopper.last_check['stale_windows']==2


def test_improving_scores_do_not_stop_and_noise_is_not_improvement():
    a=PlateauStopper({'enabled':True,'min_iterations':2,'window':1,'patience':2,'min_delta':.01,'confidence_z':1})
    assert not any(a.observe(i,float(i),.01) for i in range(1,20))
    b=PlateauStopper({'enabled':True,'min_iterations':2,'window':1,'patience':2,'min_delta':.01,'confidence_z':1})
    assert b.observe(1,1.,.2) is False
    assert b.observe(2,1.01,.2) is False
    assert b.observe(3,.99,.2) is True


def test_disabled_and_unknown_uncertainty_do_not_stop():
    a=PlateauStopper({'enabled':False})
    b=PlateauStopper({'enabled':True,'min_iterations':2,'window':1,'patience':1})
    for i in range(1,20):
        assert not a.observe(i,1.,.1)
        assert not b.observe(i,1.,None)


def test_qiskit_early_stop_keeps_final_measurement_and_actual_budget(cfg,tmp_path):
    class FlatObjective:
        best={'comet_score':1.0};seen=set();unique_comet_evaluations=0
        def evaluate_counts(self,counts,encoder):
            return {'mean_comet_score':1.,'shot_standard_error':.01}
    cfg['optimizer'].update(iterations=50,last_avg=1)
    cfg['optimizer']['early_stopping'].update(enabled=True,min_iterations=2,window=1,patience=2)
    result=optimize(cfg,CompositionEncoder(cfg['search']),FlatObjective(),tmp_path)
    assert result['iterations']==3
    assert result['maximum_iterations']==50
    assert result['termination_reason']=='plateau'
    assert result['objective_calls']==result['expected_objective_calls']==11
    assert result['final_mean_comet_score']==1.
