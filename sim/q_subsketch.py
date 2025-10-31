"""
Quantum Suffix Sketch (Q-SubSketch)

Approximate substring membership using amplitude-encoded rolling hashes.

REFACTORED: Now inherits from AmplitudeSketch base class.
"""
import numpy as np
from qiskit import QuantumCircuit
from .amplitude_sketch import AmplitudeSketch
from .utils import bitstring_to_int

class QSubSketch(AmplitudeSketch):
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
        # Initialize base class
        super().__init__(m, k, theta)
        
        # Q-SubSketch specific parameters
        self.L = L
        self.stride = stride

    def rolling_hashes(self, text):
        """
        Compute k rolling hashes for each substring window.
        
        Args:
            text: Text to extract substrings from (bytes or string)
            
        Returns:
            List of hash vectors, each with k indices
        """
        text_bytes = text if isinstance(text, bytes) else text.encode()
        n = len(text_bytes)
        windows = []
        
        for i in range(0, n - self.L + 1, self.stride):
            window = text_bytes[i:i+self.L]
            hvec = [h(bitstring_to_int(window)) % self.m for h in self.hash_functions]
            windows.append(hvec)
        
        return windows
    
    def _build_insert_circuit(self, text):
        """
        Build circuit for inserting text (all substrings).
        
        Args:
            text: Text to insert (bytes or string)
            
        Returns:
            QuantumCircuit with phase encodings for all substrings
        """
        qc = QuantumCircuit(self.m)
        
        # Initialize to |+⟩^⊗m
        qc.h(range(self.m))
        
        # Encode all substring hashes
        for hvec in self.rolling_hashes(text):
            for idx in hvec:
                qc.rz(self.theta, idx)
        
        return qc

    def build_sketch_circuit(self, text):
        """Build quantum circuit encoding all substring hashes (legacy API)."""
        return self._build_insert_circuit(text)

    def build_query_circuit(self, text, pattern):
        """Build circuit to query for pattern membership (legacy API)."""
        qc = self.build_sketch_circuit(text)
        
        # Apply inverse rotations for query pattern
        pattern_bytes = pattern if isinstance(pattern, bytes) else pattern.encode()
        hvec = [h(bitstring_to_int(pattern_bytes)) % self.m for h in self.hash_functions]
        for idx in hvec:
            qc.rz(-self.theta, idx)
        
        qc.h(range(self.m))
        qc.measure_all()
        return qc
    
    def insert(self, text):
        """Insert text into sketch (stateful API compatibility)."""
        self.n_inserts += 1

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
    
    def get_circuit_depth(self, text):
        """Estimate circuit depth for text insertion."""
        text_bytes = text if isinstance(text, bytes) else text.encode()
        n_windows = max(1, (len(text_bytes) - self.L + 1) // self.stride)
        return self.k * n_windows
