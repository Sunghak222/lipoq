"""Shallow RY/CX circuit and reproducible finite-shot statevector sampling."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


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

    def sample(self, theta, shots, measurement_seed=None):
        bound = self.circuit.assign_parameters(dict(zip(self.parameters, theta)))
        state = Statevector.from_instruction(bound)
        if measurement_seed is None:
            measurement_seed = int(self.rng.integers(0, 2**32))
        state.seed(measurement_seed)
        counts = {str(k): int(v) for k, v in state.sample_counts(shots).items()}
        return counts, measurement_seed
