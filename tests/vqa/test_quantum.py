import numpy as np
from vqa.circuit import QuantumSampler


def test_actual_qiskit_endianness_and_sampling_seed():
    sampler = QuantumSampler(2, 1, 42)
    # Only final RY on q0 rotates by pi. Expected measurement is q1q0 = 01.
    counts, _ = sampler.sample([0, 0, np.pi, 0], 32)
    assert counts == {"01": 32}
    a, b = QuantumSampler(3, 1, 10), QuantumSampler(3, 1, 10)
    theta = np.linspace(0.1, 2.7, 6)
    assert a.sample(theta, 63) == b.sample(theta, 63)
    assert a.sample(theta, 63) == b.sample(theta, 63)

