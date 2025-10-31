"""
Quantum Suffix Sketch (Q-SubSketch)

Approximate substring membership using amplitude-encoded rolling hashes.
"""
import numpy as np
from qiskit import QuantumCircuit
from .utils import make_hash_functions, bitstring_to_int

class QSubSketch:
    """Quantum Suffix Sketch for substring search."""
    def __init__(self, m, k, L, stride=1, theta=np.pi/4):
        """
        Args:
            m: Number of qubits (buckets)
            k: Number of hash functions
            L: Substring length
            stride: Step size for rolling window
            theta: Phase rotation angle
        """
        self.m = m
        self.k = k
        self.L = L
        self.stride = stride
        self.theta = theta
        self.hash_functions = make_hash_functions(k)

    def rolling_hashes(self, text):
        """Compute k rolling hashes for each substring window."""
        n = len(text)
        windows = []
        for i in range(0, n - self.L + 1, self.stride):
            window = text[i:i+self.L]
            hvec = [h(bitstring_to_int(window)) % self.m for h in self.hash_functions]
            windows.append(hvec)
        return windows

    def build_sketch_circuit(self, text):
        """Build quantum circuit encoding all substring hashes."""
        qc = QuantumCircuit(self.m)
        qc.h(range(self.m))
        for hvec in self.rolling_hashes(text):
            for idx in hvec:
                qc.rz(self.theta, idx)
        return qc

    def build_query_circuit(self, text, pattern):
        """Build circuit to query for pattern membership."""
        qc = self.build_sketch_circuit(text)
        hvec = [h(bitstring_to_int(pattern)) % self.m for h in self.hash_functions]
        for idx in hvec:
            qc.rz(-self.theta, idx)
        qc.h(range(self.m))
        qc.measure_all()
        return qc

    def query(self, text, pattern, shots=512, noise_model=None):
        """Query if pattern is a substring of text."""
        from qiskit_aer import AerSimulator
        simulator = AerSimulator(method='automatic', noise_model=noise_model) if noise_model else AerSimulator(method='automatic')
        qc = self.build_query_circuit(text, pattern)
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        zero_bitstring = '0' * self.m
        all_zero_count = counts.get(zero_bitstring, 0)
        expectation = all_zero_count / shots
        return expectation
