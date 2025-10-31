"""
Quantum Approximate Membership (QAM) - "Quantum Bloom Filter"

Supports insert(x) and query(x ∈ S?) with tunable false-positive rate.
Uses phase-based encoding where items are mapped to qubit indices via k hash functions.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from .utils import make_hash_functions, bitstring_to_int


class QAM:
    """Quantum Approximate Membership data structure."""
    
    def __init__(self, m, k, theta=np.pi/4):
        """
        Initialize QAM.
        
        Args:
            m: Number of qubits (memory size)
            k: Number of hash functions
            theta: Phase rotation angle (default π/4)
        """
        self.m = m
        self.k = k
        self.theta = theta
        self.hash_functions = make_hash_functions(k)
        self.inserted_items = []
        
    def _get_indices(self, x):
        """Get k qubit indices for item x."""
        x_int = bitstring_to_int(x)
        return [h(x_int) % self.m for h in self.hash_functions]
    
    def build_insert_circuit(self, items, deleted_items=None):
        """
        Build circuit with all insertions.
        
        Args:
            items: List of items to insert
            
        Returns:
            QuantumCircuit with phase encodings
        """
        qc = QuantumCircuit(self.m)
        
        # Initialize to |+⟩^⊗m for superposition
        qc.h(range(self.m))
        
        # Insert each item by applying phase rotations
        for item in items:
            indices = self._get_indices(item)
            for idx in indices:
                qc.rz(self.theta, idx)

        # Apply inverse phase for deleted items
        if deleted_items:
            for item in deleted_items:
                indices = self._get_indices(item)
                for idx in indices:
                    qc.rz(-self.theta, idx)

        return qc
    
    def build_query_circuit(self, items, query_item, deleted_items=None):
        """
        Build circuit to query membership.
        
        Args:
            items: List of inserted items
            query_item: Item to query
            deleted_items: List of deleted items (optional)
        Returns:
            QuantumCircuit for query
        """
        qc = self.build_insert_circuit(items, deleted_items=deleted_items)
        # Apply inverse rotations for query item
        query_indices = self._get_indices(query_item)
        for idx in query_indices:
            qc.rz(-self.theta, idx)
        # Hadamard before measurement
        qc.h(range(self.m))
        # Measure all qubits
        qc.measure_all()
        return qc
    
    def query(self, items, query_item, shots=512, noise_model=None, deleted_items=None):
        """
        Query if item is in set.
        
        Args:
            items: List of inserted items
            query_item: Item to query
            shots: Number of measurement shots
            noise_model: Optional Qiskit noise model
            deleted_items: List of deleted items (optional)
        Returns:
            Expectation value (higher = more likely member)
        """
        qc = self.build_query_circuit(items, query_item, deleted_items=deleted_items)
        # Run simulation with automatic method selection
        simulator = AerSimulator(method='automatic', noise_model=noise_model) if noise_model else AerSimulator(method='automatic')
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        # Calculate expectation: count |0...0⟩ occurrences
        zero_bitstring = '0' * self.m
        all_zero_count = counts.get(zero_bitstring, 0)
        expectation = all_zero_count / shots
        return expectation
    
    def query_statevector(self, items, query_item, deleted_items=None):
        """
        Query using statevector (no noise, exact).
        
        Args:
            items: List of inserted items
            query_item: Item to query
            deleted_items: List of deleted items (optional)
        Returns:
            Probability of measuring |0...0⟩
        """
        qc = self.build_query_circuit(items, query_item, deleted_items=deleted_items)
        qc.remove_final_measurements()
        state = Statevector.from_instruction(qc)
        prob_zero = np.abs(state.data[0])**2
        return prob_zero

    def delete(self, item):
        """Track deleted items (for simulation)."""
        if not hasattr(self, 'deleted_items'):
            self.deleted_items = []
        self.deleted_items.append(item)
    
    def insert(self, item):
        """Track inserted items."""
        self.inserted_items.append(item)
    
    def batch_query(self, items, query_items, shots=512, threshold=0.5, noise_model=None):
        """
        Query multiple items efficiently.
        
        Args:
            items: List of inserted items
            query_items: List of items to query
            shots: Number of shots per query
            threshold: Decision threshold
            noise_model: Optional noise model
            
        Returns:
            List of (item, expectation, is_member) tuples
        """
        results = []
        for q_item in query_items:
            exp = self.query(items, q_item, shots=shots, noise_model=noise_model)
            is_member = exp >= threshold
            results.append((q_item, exp, is_member))
        
        return results


def create_noise_model(error_rate=0.001):
    """
    Create depolarizing noise model.
    
    Args:
        error_rate: Per-gate error rate (ε)
        
    Returns:
        Qiskit NoiseModel
    """
    noise_model = NoiseModel()
    
    # Single-qubit gate error
    error_1q = depolarizing_error(error_rate, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['rz', 'h'])
    
    # Two-qubit gate error
    error_2q = depolarizing_error(error_rate, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz'])
    
    return noise_model
