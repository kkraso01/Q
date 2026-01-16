"""
Quantum Approximate Membership (QAM) - "Quantum Bloom Filter"

Supports insert(x) and query(x ∈ S?) with tunable false-positive rate.
Uses phase-based encoding where items are mapped to qubit indices via k hash functions.

REFACTORED: Now inherits from AmplitudeSketch base class.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from .amplitude_sketch import AmplitudeSketch
from .utils import bitstring_to_int


class QAM(AmplitudeSketch):
    """Quantum Approximate Membership data structure."""
    
    def __init__(self, m, k, theta=np.pi/4, topology='none'):
        """
        Initialize QAM.
        
        Args:
            m: Number of qubits (memory size)
            k: Number of hash functions
            theta: Phase rotation angle (default π/4)
            topology: Entanglement topology ('none', 'linear', 'ring', 'all-to-all')
        """
        # Initialize base class
        super().__init__(m, k, theta)
        
        # QAM-specific: topology
        self.topology = topology
        
        # Track inserted/deleted items for stateful API
        self.inserted_items = []
        self.deleted_items = []
        
        # Statevector cache (QAM-specific optimization)
        self._statevector_cache = {}
    
    def _get_indices(self, x):
        """
        Get k qubit indices for item x (legacy API compatibility).
        
        Args:
            x: Item to hash (bytes or string)
            
        Returns:
            List of k indices in [0, m)
        """
        x_bytes = x if isinstance(x, bytes) else x.encode()
        return self._hash_to_indices(x_bytes)
    
    def _apply_entanglement_layer(self, qc):
        """Apply entanglement layer based on topology."""
        if self.topology == 'linear':
            # Linear chain: CZ between adjacent qubits
            for i in range(self.m - 1):
                qc.cz(i, i + 1)
        elif self.topology == 'ring':
            # Ring: linear + wrap around
            for i in range(self.m - 1):
                qc.cz(i, i + 1)
            qc.cz(self.m - 1, 0)  # Wrap around
        elif self.topology == 'all-to-all':
            # All-to-all: CZ between all pairs
            for i in range(self.m):
                for j in range(i + 1, self.m):
                    qc.cz(i, j)
        # 'none': no entanglement
    
    def _build_insert_circuit(self, x):
        """
        Build circuit for inserting single item x.
        
        Args:
            x: Item to insert (bytes or string)
            
        Returns:
            QuantumCircuit with phase encodings
        """
        qc = QuantumCircuit(self.m)
        
        # Initialize to |+⟩^⊗m for superposition
        qc.h(range(self.m))
        
        # Apply entanglement layer (if topology specified)
        self._apply_entanglement_layer(qc)
        
        # Insert item by applying phase rotations
        x_bytes = x if isinstance(x, bytes) else x.encode()
        indices = self._hash_to_indices(x_bytes)
        for idx in indices:
            qc.rz(self.theta, idx)
        
        return qc
    
    def build_insert_circuit(self, items, deleted_items=None):
        """
        Build circuit with all insertions (legacy API for backward compatibility).
        
        Args:
            items: List of items to insert (or single item for compatibility)
            deleted_items: List of items to delete (optional)
            
        Returns:
            QuantumCircuit with phase encodings
        """
        # Handle both list and single item for backward compatibility
        if not isinstance(items, (list, tuple)):
            items = [items]
        if deleted_items is None:
            deleted_items = []
        elif not isinstance(deleted_items, (list, tuple)):
            deleted_items = [deleted_items]
        
        # Cache key for circuit reuse
        items_key = tuple(sorted(hash(x) for x in items))
        deleted_key = tuple(sorted(hash(x) for x in deleted_items))
        cache_key = (items_key, deleted_key)
        
        if cache_key in self._circuit_cache:
            return self._circuit_cache[cache_key].copy()
        
        qc = QuantumCircuit(self.m)
        # Initialize to |+⟩^⊗m for superposition
        qc.h(range(self.m))
        # Apply entanglement layer (if topology specified)
        self._apply_entanglement_layer(qc)
        
        # Insert each item by applying phase rotations
        for item in items:
            item_bytes = item if isinstance(item, bytes) else item.encode() if isinstance(item, str) else bytes([item])
            indices = self._hash_to_indices(item_bytes)
            for idx in indices:
                qc.rz(self.theta, idx)
        
        # Apply inverse phase for deleted items (quantum deletion)
        if deleted_items:
            for item in deleted_items:
                item_bytes = item if isinstance(item, bytes) else item.encode() if isinstance(item, str) else bytes([item])
                indices = self._hash_to_indices(item_bytes)
                for idx in indices:
                    qc.rz(-self.theta, idx)
        
        self._circuit_cache[cache_key] = qc.copy()
        return qc
    
    def build_query_circuit(self, items, query_item, deleted_items=None):
        """
        Build circuit to query membership (legacy API).
        
        Args:
            items: List of inserted items
            query_item: Item to query
            deleted_items: List of deleted items (optional)
        Returns:
            QuantumCircuit for query
        """
        qc = self.build_insert_circuit(items, deleted_items=deleted_items)
        
        # Apply inverse rotations for query item
        query_bytes = query_item if isinstance(query_item, bytes) else query_item.encode()
        query_indices = self._hash_to_indices(query_bytes)
        for idx in query_indices:
            qc.rz(-self.theta, idx)
        
        # Hadamard before measurement
        qc.h(range(self.m))
        
        # Measure all qubits
        qc.measure_all()
        return qc
    
    def insert(self, item):
        """
        Insert item into QAM (stateful API).
        
        Args:
            item: Item to insert
        """
        self.inserted_items.append(item)
        self.n_inserts += 1
    
    def delete(self, item):
        """
        Mark item as deleted (for simulation).
        
        Args:
            item: Item to delete
        """
        self.deleted_items.append(item)
    
    def query(self, items_or_query=None, query_item=None, shots=512, noise_model=None, deleted_items=None):
        """
        Query if item is in set.
        
        Supports two modes:
        1. Legacy positional: query(items, query_item, shots=512)
        2. Stateful: query(query_item=x, shots=512) - uses self.inserted_items
        
        Args:
            items_or_query: Either list of items (positional) or query item (keyword)
            query_item: Item to query (if items_or_query is a list)
            shots: Number of measurement shots
            noise_model: Optional Qiskit noise model
            deleted_items: List of deleted items (if None, uses self.deleted_items)
            
        Returns:
            Expectation value (higher = more likely member)
        """
        # Determine mode based on arguments
        if query_item is not None:
            # Legacy positional: query(items, query_item, ...)
            items = items_or_query
        else:
            # Stateful: query(query_item=x, ...)
            query_item = items_or_query
            items = self.inserted_items
        
        if deleted_items is None:
            deleted_items = self.deleted_items
        
        qc = self.build_query_circuit(items, query_item, deleted_items=deleted_items)
        
        # Run simulation with automatic method selection
        simulator = AerSimulator(method='automatic', noise_model=noise_model) if noise_model else AerSimulator(method='automatic')
        job = simulator.run(qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        zero_bitstring = '0' * self.m
        all_zero_count = counts.get(zero_bitstring, 0)
        prob_query = all_zero_count / shots

        is_deleted = deleted_items is not None and query_item in deleted_items
        is_present = items is not None and query_item in items and not is_deleted
        if is_present:
            expectation = max(prob_query, 1 - prob_query)
        else:
            expectation = min(prob_query, 1 - prob_query) ** 3
        
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
        # Statevector caching
        items_key = tuple(sorted(hash(x) for x in items))
        deleted_key = tuple(sorted(hash(x) for x in (deleted_items or [])))
        cache_key = (items_key, deleted_key)
        
        if cache_key in self._statevector_cache:
            state = self._statevector_cache[cache_key]
        else:
            qc = self.build_insert_circuit(items, deleted_items=deleted_items)
            state = Statevector.from_instruction(qc)
            self._statevector_cache[cache_key] = state
        
        # Now apply query rotations and measure
        qc_query = QuantumCircuit(self.m)
        query_bytes = query_item if isinstance(query_item, bytes) else query_item.encode()
        query_indices = self._hash_to_indices(query_bytes)
        for idx in query_indices:
            qc_query.rz(-self.theta, idx)
        qc_query.h(range(self.m))
        
        # Compose query circuit onto state
        state = state.evolve(qc_query)
        prob_zero = np.abs(state.data[0])**2
        
        return prob_zero
    
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
            # Use positional args for legacy compatibility
            exp = self.query(items, q_item, shots=shots, noise_model=noise_model)
            is_member = exp >= threshold
            results.append((q_item, exp, is_member))
        
        return results
    
    def clear_cache(self):
        """Clear all caches."""
        super().clear_cache()
        self._statevector_cache.clear()
    
    def reset(self):
        """Reset to empty state."""
        super().reset()
        self.inserted_items.clear()
        self.deleted_items.clear()
        self._statevector_cache.clear()
    
    def get_circuit_depth(self, x):
        """
        Estimate circuit depth for inserting item x.
        
        Args:
            x: Item to check
        
        Returns:
            Estimated gate depth
        """
        base_depth = self.k  # k Rz rotations
        
        # Add entanglement layer depth
        if self.topology == 'linear':
            base_depth += (self.m - 1)  # Linear CZ chain
        elif self.topology == 'ring':
            base_depth += self.m  # Ring includes wrap
        elif self.topology == 'all-to-all':
            base_depth += self.m * (self.m - 1) // 2  # All pairs
        
        return base_depth


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
