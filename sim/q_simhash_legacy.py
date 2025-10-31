"""
Quantum Similarity Hash (Q-SimHash)

Approximate nearest-neighbor search using amplitude-phase encoding.
"""
import numpy as np
from qiskit import QuantumCircuit
from .utils import make_hash_functions, bitstring_to_int

class QSimHash:
    """Quantum Similarity Hash for approximate nearest neighbor."""
    def __init__(self, m, k, theta=np.pi/4):
        """
        Args:
            m: Number of qubits (buckets)
            k: Number of hash functions (hyperplanes)
            theta: Phase rotation angle
        """
        self.m = m
        self.k = k
        self.theta = theta
        self.hash_functions = make_hash_functions(k)

    def encode_vector(self, vec):
        """
        Encode a binary vector as phase rotations.
        vec: bytes or list/array of 0/1
        """
        if isinstance(vec, bytes):
            bits = np.unpackbits(np.frombuffer(vec, dtype=np.uint8))
        else:
            bits = np.array(vec, dtype=np.uint8)
        return bits

    def build_encoding_circuit(self, vec):
        """
        Build circuit encoding vector signs into phases.
        """
        qc = QuantumCircuit(self.m)
        qc.h(range(self.m))
        bits = self.encode_vector(vec)
        for i, b in enumerate(bits[:self.k]):
            idx = self.hash_functions[i](bitstring_to_int(vec)) % self.m
            angle = self.theta if b else -self.theta
            qc.rz(angle, idx)
        return qc

    def build_similarity_circuit(self, vec1, vec2):
        """
        Build circuit to estimate similarity between two vectors.
        """
        qc = self.build_encoding_circuit(vec1)
        bits2 = self.encode_vector(vec2)
        for i, b in enumerate(bits2[:self.k]):
            idx = self.hash_functions[i](bitstring_to_int(vec2)) % self.m
            angle = -self.theta if b else self.theta
            qc.rz(angle, idx)
        qc.h(range(self.m))
        qc.measure_all()
        return qc

    def similarity(self, vec1, vec2, shots=512, noise_model=None):
        """
        Estimate similarity between two vectors.
        """
        from qiskit_aer import AerSimulator
        simulator = AerSimulator(method='automatic', noise_model=noise_model) if noise_model else AerSimulator(method='automatic')
        qc = self.build_similarity_circuit(vec1, vec2)
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        zero_bitstring = '0' * self.m
        all_zero_count = counts.get(zero_bitstring, 0)
        expectation = all_zero_count / shots
        return expectation
