"""Measured Qiskit circuit and first-order SPSA on direct COMET sample means."""
import time
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import SPSA
from qiskit_algorithms.utils import algorithm_globals

from .results import append_csv, append_jsonl, write_json


class QuantumSampler:
    def __init__(self, num_qubits, depth, seed):
        self.circuit = QuantumCircuit(num_qubits)
        self.parameters = ParameterVector("theta", num_qubits * (depth + 1))
        index = 0
        for layer in range(depth + 1):
            for qubit in range(num_qubits):
                self.circuit.ry(self.parameters[index], qubit)
                index += 1
            if layer < depth:
                for qubit in range(num_qubits - 1):
                    self.circuit.cx(qubit, qubit + 1)
        self.rng = np.random.default_rng(seed)

    def probabilities(self, theta):
        """Exact simulator probabilities, for post-training analysis only."""
        bound = self.circuit.assign_parameters(dict(zip(self.parameters, theta)))
        return Statevector.from_instruction(bound).probabilities()

    def sample(self, theta, shots):
        bound = self.circuit.assign_parameters(dict(zip(self.parameters, theta)))
        state = Statevector.from_instruction(bound)
        measurement_seed = int(self.rng.integers(0, 2**32))
        state.seed(measurement_seed)
        counts = {str(k): int(v) for k, v in state.sample_counts(shots).items()}
        return counts, measurement_seed


def optimize(cfg, encoder, evaluator, run_dir):
    start = time.perf_counter()
    seeds = np.random.SeedSequence(cfg["seed"]).spawn(3)
    initial_rng = np.random.default_rng(seeds[0])
    sampler = QuantumSampler(encoder.num_qubits, cfg["circuit"]["depth"], seeds[1])
    algorithm_globals.random_seed = int(seeds[2].generate_state(1)[0])
    theta0 = initial_rng.uniform(0, 2*np.pi, len(sampler.parameters))
    # Export a measured circuit for inspection, although the simulator samples the state directly.
    measured = sampler.circuit.copy()
    measured.measure_all()
    (run_dir / "circuit.qasm").write_text(qasm3.dumps(measured))
    np.save(run_dir / "initial_parameters.npy", theta0)
    objective_calls = 0
    last_metrics = None
    phase = "initial"

    def objective(theta):
        nonlocal objective_calls, last_metrics
        counts, measurement_seed = sampler.sample(theta, cfg["circuit"]["shots"])
        metrics = evaluator.evaluate_counts(counts, encoder)
        objective_calls += 1
        last_metrics = metrics
        append_jsonl(run_dir / "evaluations.jsonl", {
            "objective_call": objective_calls, "phase": phase,
            "measurement_seed": measurement_seed, "counts": counts,
            "theta": np.asarray(theta).tolist(), **metrics,
            "elapsed_optimization_seconds": time.perf_counter()-start,
        })
        write_json(run_dir / "best.json", evaluator.best)
        return -metrics["mean_comet_score"]

    initial_mean = -objective(theta0)
    iteration = 0

    def progress(theta, loss, step, accepted):
        row = {"iteration": iteration, "objective_calls": objective_calls,
               "mean_comet_score": -float(loss), "best_so_far_score": evaluator.best["comet_score"],
               "shot_standard_error": last_metrics["shot_standard_error"],
               "step_norm": float(step), "accepted": bool(accepted),
               "unique_sampled_compositions": len(evaluator.seen),
               "comet_evaluations": evaluator.unique_comet_evaluations,
               "elapsed_optimization_seconds": time.perf_counter()-start}
        append_csv(run_dir / "history.csv", row)
        write_json(run_dir / "optimizer_state.json", {**row, "theta": np.asarray(theta).tolist()})
        print(f"iteration={iteration:3d} mean={-loss:.6f} best={evaluator.best['comet_score']:.6f} "
              f"unique={len(evaluator.seen)}", flush=True)

    progress(theta0, -initial_mean, 0, True)

    def callback(nfev, theta, loss, step, accepted):
        nonlocal iteration
        iteration += 1
        progress(theta, loss, step, accepted)

    settings = cfg["optimizer"]
    ks = np.arange(1, settings["iterations"]+1, dtype=float)
    learning = settings["learning_rate"] / (settings["stability_constant"]+ks)**settings["alpha"]
    perturbation = settings["perturbation"] / ks**settings["gamma"]
    optimizer = SPSA(maxiter=settings["iterations"], learning_rate=learning, perturbation=perturbation,
                     second_order=False, blocking=False, resamplings=1, callback=callback)
    phase = "spsa"
    result = optimizer.minimize(objective, theta0)
    np.save(run_dir / "final_parameters.npy", result.x)
    # Qiskit evaluates the final parameters once more after its last callback.
    # That fresh shot mean is retained separately from the final history row.
    return {"optimizer": "qiskit_algorithms.optimizers.SPSA", "second_order": False,
            "iterations": int(result.nit), "objective_calls": objective_calls,
            "initial_mean_comet_score": initial_mean, "final_mean_comet_score": -float(result.fun),
            "final_mean_is_fresh_shot_estimate": True,
            "num_parameters": len(theta0), "num_qubits": encoder.num_qubits,
            "circuit_depth": sampler.circuit.depth(), "entangling_layers": cfg["circuit"]["depth"],
            "optimization_seconds": time.perf_counter()-start}

