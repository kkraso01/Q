"""
Quantum Hashed Trie (QHT) - Prefix Membership Structure

Supports prefix-based membership queries using phase-encoded trie structure.
Each character in a prefix is mapped to qubit buckets via hash functions.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from .utils import make_hash_functions, bitstring_to_int


class QHT:
    """Quantum Hashed Trie for prefix membership queries."""
    
    def __init__(self, m, b, L, theta=np.pi/4, k=3):
        """
        Initialize QHT.
        
        Args:
            m: Number of qubits (memory size)
            b: Branching factor (alphabet cardinality)
            L: Maximum depth (prefix length)
            theta: Phase rotation angle (default π/4)
            k: Number of hash functions per character
        """
        self.m = m
        self.b = b
        self.L = L
        self.theta = theta
        self.k = k
        self.hash_functions = make_hash_functions(k)
        self.inserted_items = []
        self._circuit_cache = {}
        self._statevector_cache = {}
    
    def _get_indices(self, char, depth):
        """Get k qubit indices for character at given depth."""
        # Combine character and depth for unique hashing
        combined = hash((char, depth))
        return [h(combined) % self.m for h in self.hash_functions]
    
    def _extract_prefix(self, x, length):
        """Extract prefix of given length from item x."""
        if isinstance(x, bytes):
            x = x.decode('utf-8', errors='ignore')
        return str(x)[:min(length, len(str(x)))]
    
    def build_insert_circuit(self, items):
        """
        Build circuit encoding all inserted items.
        
        Args:
            items: List of items (strings or bytes) to insert
            
        Returns:
            QuantumCircuit with prefix encodings
        """
        # Cache based on item set
        cache_key = tuple(sorted(hash(x) for x in items))
        if cache_key in self._circuit_cache:
            return self._circuit_cache[cache_key].copy()
        
        qc = QuantumCircuit(self.m)
        # Initialize to superposition
        qc.h(range(self.m))
        
        # Insert each item by encoding all its prefixes
        for item in items:
            item_str = item.decode('utf-8') if isinstance(item, bytes) else str(item)
            # Encode prefixes from depth 1 to min(L, len(item))
            for depth in range(1, min(self.L + 1, len(item_str) + 1)):
                char = item_str[depth - 1]
                indices = self._get_indices(char, depth)
                for idx in indices:
                    qc.rz(self.theta, idx)
        
        self._circuit_cache[cache_key] = qc.copy()
        return qc
    
    def query(self, prefix, items=None, shots=512, noise_level=0.0):
        """
        Query for prefix membership.
        
        Args:
            prefix: Prefix string to query
            items: Items to query against (default: self.inserted_items)
            shots: Number of measurement shots
            noise_level: Depolarizing noise probability per gate
            
        Returns:
            float: Acceptance probability (0 to 1)
        """
        if items is None:
            items = self.inserted_items
        
        # Build base circuit with insertions
        qc_base = self.build_insert_circuit(items)
        
        # Build query circuit: apply same rotations for prefix
        qc_query = qc_base.copy()
        prefix_str = prefix.decode('utf-8') if isinstance(prefix, bytes) else str(prefix)
        
        for depth in range(1, min(self.L + 1, len(prefix_str) + 1)):
            char = prefix_str[depth - 1]
            indices = self._get_indices(char, depth)
            for idx in indices:
                qc_query.rz(self.theta, idx)
        
        # Measure all qubits
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
        
        # Acceptance: measure overlap with |0...0⟩ state
        zero_state = '0' * self.m
        acceptance = counts.get(zero_state, 0) / shots
        
        return acceptance
    
    def insert(self, x):
        """Insert item into the trie."""
        self.inserted_items.append(x)
        # Clear caches
        self._circuit_cache.clear()
        self._statevector_cache.clear()
    
    def get_circuit_depth(self, items=None):
        """Get circuit depth for given items."""
        if items is None:
            items = self.inserted_items
        qc = self.build_insert_circuit(items)
        return qc.depth()
