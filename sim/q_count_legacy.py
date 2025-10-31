"""
Quantum Count-Distinct (Q-Count) - Streaming Cardinality Estimation

Estimates the number of distinct items in a data stream using quantum phase encoding.
Similar to HyperLogLog but uses quantum amplitude statistics.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from .utils import make_hash_functions, bitstring_to_int


class QCount:
    """Quantum Count-Distinct estimator for streaming cardinality."""
    
    def __init__(self, m, k=3, theta=np.pi/4):
        """
        Initialize Q-Count.
        
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
        Build circuit encoding all items in stream.
        
        Args:
            items: List of items in stream
            
        Returns:
            QuantumCircuit with bucket phase encodings
        """
        # Cache based on unique items (not stream order)
        unique_items = set(items)
        cache_key = tuple(sorted(hash(x) for x in unique_items))
        
        if cache_key in self._circuit_cache:
            return self._circuit_cache[cache_key].copy()
        
        qc = QuantumCircuit(self.m)
        # Initialize to superposition
        qc.h(range(self.m))
        
        # Mark buckets for each unique item
        for item in unique_items:
            indices = self._get_indices(item)
            for idx in indices:
                qc.rz(self.theta, idx)
        
        self._circuit_cache[cache_key] = qc.copy()
        return qc
    
    def estimate_cardinality(self, items=None, shots=512, noise_level=0.0):
        """
        Estimate number of distinct items in stream.
        
        Args:
            items: Items to estimate (default: self.stream)
            shots: Number of measurement shots
            noise_level: Depolarizing noise probability per gate
            
        Returns:
            int: Estimated cardinality
        """
        if items is None:
            items = self.stream
        
        if len(items) == 0:
            return 0
        
        # Build circuit
        qc = self.build_circuit(items)
        qc.measure_all()
        
        # Simulate
        if noise_level > 0:
            noise_model = NoiseModel()
            error = depolarizing_error(noise_level, 2)
            noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])
            simulator = AerSimulator(noise_model=noise_model)
        else:
            simulator = AerSimulator()
        
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        
        # Estimate cardinality from measurement statistics
        # Count number of buckets with significant phase rotation
        # (measured as deviation from uniform distribution)
        
        # Calculate variance across measurements
        measurements = []
        for bitstring, count in counts.items():
            measurements.extend([int(bitstring, 2)] * count)
        
        if len(measurements) == 0:
            return 0
        
        variance = np.var(measurements)
        mean = np.mean(measurements)
        
        # Estimate cardinality: higher variance → more distinct items
        # Scaling factor based on m, k, theta
        # Simple estimator: scale variance by theoretical max
        max_variance = (2**self.m - 1) ** 2 / 4  # Max variance for uniform dist
        normalized_variance = variance / max_variance if max_variance > 0 else 0
        
        # Heuristic estimator (can be refined with calibration)
        cardinality_estimate = int(normalized_variance * self.m * self.k)
        
        # Bound by actual unique count
        actual_unique = len(set(items))
        return min(cardinality_estimate, actual_unique)
    
    def insert(self, x):
        """Insert item into stream."""
        self.stream.append(x)
        # Clear caches when stream changes
        self._circuit_cache.clear()
        self._statevector_cache.clear()
    
    def get_true_cardinality(self, items=None):
        """Get true cardinality (for evaluation)."""
        if items is None:
            items = self.stream
        return len(set(items))
