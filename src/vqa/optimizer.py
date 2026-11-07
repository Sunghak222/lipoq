"""First-order SPSA on direct COMET sample means."""
import time
import numpy as np
from qiskit import qasm3
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals

from common.results import append_csv, append_jsonl, output_path, write_json
from .circuit import QuantumSampler
from .early_stopping import PlateauStopper


def optimize(cfg, encoder, evaluator, run_dir):
    start = time.perf_counter()
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(3)
    initial_rng = np.random.default_rng(seeds[0])
    sampler = QuantumSampler(encoder.num_qubits, cfg["circuit"]["depth"], seeds[1])
    algorithm_globals.random_seed = int(seeds[2].generate_state(1)[0])
    settings = cfg["optimizer"]
    resamplings = settings["resamplings"]
    paired = settings["paired_sampling"]
    stopper = PlateauStopper(settings["early_stopping"])
    if settings["initialization"] == "uniform":
        theta0 = np.zeros(len(sampler.parameters))
        theta0[-encoder.num_qubits:] = np.pi/2
    else:
        theta0 = initial_rng.uniform(0, 2*np.pi, len(sampler.parameters))
    # Export a measured circuit for inspection, although the simulator samples the state directly.
    measured = sampler.circuit.copy()
    measured.measure_all()
    (output_path(run_dir, "circuit.qasm")).write_text(qasm3.dumps(measured))
    np.save(output_path(run_dir, "initial_parameters.npy"), theta0)
    objective_calls = 0
    last_metrics = None
    phase = "initial"
    iteration = 0
    slot = 0
    pair_seed = None
    plus = None
    gradients = []

    def objective(theta):
        nonlocal objective_calls, last_metrics, slot, pair_seed, plus
        current_phase = phase
        reuse_seed = None
        if phase == "spsa":
            if iteration == settings["iterations"] or stopper.should_stop:
                current_phase = "final"
            elif slot < 2*resamplings:
                current_phase = "perturb_plus" if slot % 2 == 0 else "perturb_minus"
                if paired and slot % 2:
                    reuse_seed = pair_seed
            else:
                current_phase = "monitor"
        counts, measurement_seed = sampler.sample(theta, cfg["circuit"]["shots"], reuse_seed)
        metrics = evaluator.evaluate_counts(counts, encoder)
        if current_phase == "perturb_plus":
            pair_seed = measurement_seed
            plus = (np.asarray(theta).copy(), metrics["mean_comet_score"])
        elif current_phase == "perturb_minus":
            displacement = plus[0]-np.asarray(theta)
            gradients.append(-(plus[1]-metrics["mean_comet_score"])/displacement)
        if phase == "spsa":
            slot += 1
        objective_calls += 1
        last_metrics = metrics
        append_jsonl(output_path(run_dir, "evaluations.jsonl"), {
            "objective_call": objective_calls, "phase": current_phase,
            "iteration": iteration+1 if current_phase in ("perturb_plus", "perturb_minus", "monitor") else iteration,
            "paired_sampling": paired and current_phase in ("perturb_plus", "perturb_minus"),
            "measurement_seed": measurement_seed, "counts": counts,
            "theta": np.asarray(theta).tolist(), **metrics,
            "elapsed_optimization_seconds": time.perf_counter()-start,
        })
        return -metrics["mean_comet_score"]

    initial_mean = -objective(theta0)

    def progress(theta, loss, step, accepted):
        row = {"iteration": iteration, "objective_calls": objective_calls,
               "mean_comet_score": -float(loss), "best_so_far_score": evaluator.best["comet_score"],
               "shot_standard_error": last_metrics["shot_standard_error"],
               "step_norm": float(step), "accepted": bool(accepted),
               "unique_sampled_compositions": len(evaluator.seen),
               "comet_evaluations": evaluator.unique_comet_evaluations,
               "gradient_estimate_norm": float(np.linalg.norm(np.mean(gradients, axis=0))) if gradients else None,
               "early_stop_stale_windows": stopper.stale_windows,
               "early_stop_requested": stopper.should_stop,
               "elapsed_optimization_seconds": time.perf_counter()-start}
        append_csv(output_path(run_dir, "history.csv"), row)
        write_json(output_path(run_dir, "optimizer_state.json"), {**row, "theta": np.asarray(theta).tolist()})
        print(f"iteration={iteration:3d} mean={-loss:.6f} best={evaluator.best['comet_score']:.6f} "
              f"unique={len(evaluator.seen)}", flush=True)

    progress(theta0, -initial_mean, 0, True)

    def callback(nfev, theta, loss, step, accepted):
        nonlocal iteration, slot
        if slot != 2*resamplings+1:
            raise RuntimeError("Unexpected SPSA objective-call sequence")
        iteration += 1
        stopper.observe(iteration, -float(loss), last_metrics["shot_standard_error"])
        progress(theta, loss, step, accepted)
        if stopper.last_check is not None and stopper.last_check["iteration"] == iteration:
            append_jsonl(output_path(run_dir, "early_stopping.jsonl"), stopper.last_check)
        gradients.clear()
        slot = 0

    ks = np.arange(1, settings["iterations"]+1, dtype=float)
    learning = settings["learning_rate"] / (settings["stability_constant"]+ks)**settings["alpha"]
    perturbation = settings["perturbation"] / ks**settings["gamma"]
    optimizer = SPSA(maxiter=settings["iterations"], learning_rate=learning, perturbation=perturbation,
                     second_order=False, blocking=False, resamplings=resamplings, callback=callback,
                     trust_region=settings["trust_region"], last_avg=settings["last_avg"],
                     termination_checker=lambda *args: stopper.should_stop)
    phase = "spsa"
    result = optimizer.minimize(objective, theta0)
    expected_calls = (2*resamplings+1)*int(result.nit)+2
    if objective_calls != expected_calls:
        raise RuntimeError("Unexpected SPSA objective-call count")
    np.save(output_path(run_dir, "final_parameters.npy"), result.x)
    # Qiskit evaluates the final parameters once more after its last callback.
    # That fresh shot mean is retained separately from the final history row.
    return {"optimizer": "qiskit_algorithms.optimizers.SPSA", "second_order": False,
            "iterations": int(result.nit), "objective_calls": objective_calls,
            "termination_reason": "plateau" if stopper.should_stop else "iteration_limit",
            "convergence_certified": False, "early_stopping_enabled": stopper.settings["enabled"],
            "maximum_iterations": settings["iterations"], "early_stopping": stopper.settings,
            "last_early_stopping_check": stopper.last_check,
            "resamplings": resamplings, "paired_sampling": paired,
            "initialization": settings["initialization"],
            "last_avg": settings["last_avg"], "trust_region": settings["trust_region"],
            "expected_objective_calls": expected_calls,
            "initial_mean_comet_score": initial_mean, "final_mean_comet_score": -float(result.fun),
            "final_mean_is_fresh_shot_estimate": True,
            "num_parameters": len(theta0), "num_qubits": encoder.num_qubits,
            "circuit_depth": sampler.circuit.depth(), "entangling_layers": cfg["circuit"]["depth"],
            "optimization_seconds": time.perf_counter()-start}
