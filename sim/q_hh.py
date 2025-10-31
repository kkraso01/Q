"""
Quantum Heavy Hitters (Q-HH) - Top-K Frequency Estimation

Identifies the most frequent items in a stream using quantum phase-weighted encoding.
Similar to Count-Min Sketch but uses quantum amplitude for frequency estimation.
"""
import numpy as np
from collections import Counter
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from .utils import make_hash_functions, bitstring_to_int


class QHH:
    """Quantum Heavy Hitters for top-k frequency estimation."""
    
    def __init__(self, m, k=3, theta=np.pi/4):
        """
        Initialize Q-HH.
        
        Args:
            m: Number of qubits (bucket count)
            k: Number of hash functions
            theta: Phase rotation angle (default π/4)
        """
        self.m = m
        self.k = k
        self.theta = theta
        self.hash_functions = make_hash_functions(k)
        self.stream = []
        self._circuit_cache = {}
        self._statevector_cache = {}
    
    def _get_indices(self, x):
        """Get k qubit indices (buckets) for item x."""
        x_int = bitstring_to_int(x)
        return [h(x_int) % self.m for h in self.hash_functions]
    
    def build_circuit(self, items):
        """
        Build circuit encoding items with frequency-weighted phases.
        
        Args:
            items: List of items in stream (with repetitions)
            
        Returns:
            QuantumCircuit with frequency-weighted phase encodings
        """
        # Cache based on item frequencies
        freq_counter = Counter(items)
        cache_key = tuple(sorted((hash(x), count) for x, count in freq_counter.items()))
        
        if cache_key in self._circuit_cache:
            return self._circuit_cache[cache_key].copy()
        
        qc = QuantumCircuit(self.m)
        # Initialize to superposition
        qc.h(range(self.m))
        
        # Apply phase rotations weighted by frequency
        for item, frequency in freq_counter.items():
            indices = self._get_indices(item)
            # Weight rotation by log(frequency) to avoid overflow
            weighted_theta = self.theta * np.log1p(frequency)
            for idx in indices:
                qc.rz(weighted_theta, idx)
        
        self._circuit_cache[cache_key] = qc.copy()
        return qc
    
    def estimate_frequency(self, x, items=None, shots=512, noise_level=0.0):
        """
        Estimate frequency of item x in stream.
        
        Args:
            x: Item to estimate frequency for
            items: Stream to query (default: self.stream)
            shots: Number of measurement shots
            noise_level: Depolarizing noise probability per gate
            
        Returns:
            float: Estimated frequency
        """
        if items is None:
            items = self.stream
        
        if len(items) == 0:
            return 0.0
        
        # Build base circuit
        qc_base = self.build_circuit(items)
        
        # Build query circuit: add query item's pattern
        qc_query = qc_base.copy()
        indices = self._get_indices(x)
        for idx in indices:
            qc_query.rz(self.theta, idx)
        
        qc_query.measure_all()
        
        # Simulate
        if noise_level > 0:
            noise_model = NoiseModel()
            error = depolarizing_error(noise_level, 2)
            noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])
            simulator = AerSimulator(noise_model=noise_model)
        else:
            simulator = AerSimulator()
        
        result = simulator.run(qc_query, shots=shots).result()
        counts = result.get_counts()
        
        # Estimate frequency from measurement overlap
        # Higher overlap with all-zero state → higher frequency
        zero_state = '0' * self.m
        overlap = counts.get(zero_state, 0) / shots
        
        # Convert overlap to frequency estimate
        # Higher overlap means item appears frequently
        # Scale by total stream length
        frequency_estimate = overlap * len(items)
        
        return frequency_estimate
    
    def top_k(self, k_top, items=None, shots=512, noise_level=0.0):
        """
        Find top-k most frequent items.
        
        Args:
            k_top: Number of top items to return
            items: Stream to query (default: self.stream)
            shots: Number of measurement shots
            noise_level: Noise level
            
        Returns:
            list: Top-k items with estimated frequencies [(item, freq), ...]
        """
        if items is None:
            items = self.stream
        
        if len(items) == 0:
            return []
        
        # Get unique items
        unique_items = set(items)
        
        # Estimate frequency for each unique item
        estimates = []
        for item in unique_items:
            freq = self.estimate_frequency(item, items, shots, noise_level)
            estimates.append((item, freq))
        
        # Sort by frequency (descending) and return top-k
        estimates.sort(key=lambda x: x[1], reverse=True)
        return estimates[:k_top]
    
    def insert(self, x):
        """Insert item into stream."""
        self.stream.append(x)
        # Clear caches when stream changes
        self._circuit_cache.clear()
        self._statevector_cache.clear()
    
    def get_true_frequencies(self, items=None):
        """Get true frequency counts (for evaluation)."""
        if items is None:
            items = self.stream
        return Counter(items)
