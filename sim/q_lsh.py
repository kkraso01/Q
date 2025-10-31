"""
Quantum Locality-Sensitive Hashing (Q-LSH) - Similarity Search

Supports approximate nearest neighbor (ANN) queries using quantum phase-encoded similarity.
Uses random hyperplanes for cosine similarity estimation via amplitude interference.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from .utils import make_hash_functions, bitstring_to_int


class QLSH:
    """Quantum Locality-Sensitive Hashing for similarity search."""
    
    def __init__(self, m, k=3, theta=np.pi/4, d=128):
        """
        Initialize Q-LSH.
        
        Args:
            m: Number of qubits (bucket count)
            k: Number of hash functions (hyperplanes)
            theta: Phase rotation angle (default π/4)
            d: Vector dimensionality
        """
        self.m = m
        self.k = k
        self.theta = theta
        self.d = d
        self.hash_functions = make_hash_functions(k)
        self.hyperplanes = self._generate_hyperplanes()
        self.inserted_vectors = []
        self._circuit_cache = {}
        self._statevector_cache = {}
    
    def _generate_hyperplanes(self):
        """Generate k random hyperplanes for LSH."""
        np.random.seed(42)  # Deterministic for reproducibility
        return [np.random.randn(self.d) for _ in range(self.k)]
    
    def _get_hash_signature(self, vector):
        """Compute LSH signature (sign pattern) for vector."""
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        # Compute sign for each hyperplane
        signature = []
        for hyperplane in self.hyperplanes:
            dot_product = np.dot(vector, hyperplane)
            signature.append(1 if dot_product >= 0 else 0)
        return tuple(signature)
    
    def _signature_to_indices(self, signature):
        """Convert LSH signature to qubit indices."""
        # Hash the signature to get bucket indices
        sig_hash = hash(signature)
        return [h(sig_hash) % self.m for h in self.hash_functions]
    
    def build_insert_circuit(self, vectors):
        """
        Build circuit encoding all inserted vectors.
        
        Args:
            vectors: List of numpy arrays (vectors to insert)
            
        Returns:
            QuantumCircuit with LSH encodings
        """
        # Cache based on vector signatures
        signatures = [self._get_hash_signature(v) for v in vectors]
        cache_key = tuple(sorted(signatures))
        
        if cache_key in self._circuit_cache:
            return self._circuit_cache[cache_key].copy()
        
        qc = QuantumCircuit(self.m)
        # Initialize to superposition
        qc.h(range(self.m))
        
        # Insert each vector by encoding its LSH signature
        for signature in signatures:
            indices = self._signature_to_indices(signature)
            # Apply phase rotation based on signature
            for idx in indices:
                qc.rz(self.theta, idx)
        
        self._circuit_cache[cache_key] = qc.copy()
        return qc
    
    def cosine_similarity_estimate(self, v1, v2, shots=512, noise_level=0.0):
        """
        Estimate cosine similarity between two vectors using quantum overlap.
        
        Args:
            v1, v2: Numpy arrays (vectors to compare)
            shots: Number of measurement shots
            noise_level: Depolarizing noise probability
            
        Returns:
            float: Estimated cosine similarity (-1 to 1)
        """
        # Build circuits for both vectors
        qc1 = self.build_insert_circuit([v1])
        qc2 = self.build_insert_circuit([v2])
        
        # Create overlap test circuit
        qc_overlap = qc1.copy()
        
        # Apply inverse of qc2 (for overlap measurement)
        sig2 = self._get_hash_signature(v2)
        indices2 = self._signature_to_indices(sig2)
        for idx in indices2:
            qc_overlap.rz(-self.theta, idx)  # Inverse rotation
        
        qc_overlap.measure_all()
        
        # Simulate
        if noise_level > 0:
            noise_model = NoiseModel()
            error = depolarizing_error(noise_level, 2)
            noise_model.add_all_qubit_quantum_error(error, ['cz', 'cx'])
            simulator = AerSimulator(noise_model=noise_model)
        else:
            simulator = AerSimulator()
        
        result = simulator.run(qc_overlap, shots=shots).result()
        counts = result.get_counts()
        
        # Overlap with |0...0⟩ indicates similarity
        zero_state = '0' * self.m
        overlap = counts.get(zero_state, 0) / shots
        
        # Convert overlap to cosine similarity estimate
        # overlap ≈ cos²(angle/2) where angle is between vectors
        # For small angles: overlap ≈ 1 - angle²/4
        # Approximate: cos(angle) ≈ 2*overlap - 1
        similarity = 2 * overlap - 1
        return np.clip(similarity, -1, 1)
    
    def query_knn(self, query_vector, k_neighbors=10, shots=512, noise_level=0.0):
        """
        Find k approximate nearest neighbors.
        
        Args:
            query_vector: Query vector (numpy array)
            k_neighbors: Number of neighbors to return
            shots: Number of measurement shots
            noise_level: Noise level
            
        Returns:
            list: Top-k similar vectors with similarity scores [(vector, similarity), ...]
        """
        if len(self.inserted_vectors) == 0:
            return []
        
        # Estimate similarity with each inserted vector
        similarities = []
        for vec in self.inserted_vectors:
            sim = self.cosine_similarity_estimate(
                query_vector, vec, shots=shots, noise_level=noise_level
            )
            similarities.append((vec, sim))
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k_neighbors]
    
    def insert(self, vector):
        """Insert vector into LSH structure."""
        self.inserted_vectors.append(vector.copy())
        # Clear caches
        self._circuit_cache.clear()
        self._statevector_cache.clear()
    
    def get_circuit_depth(self, vectors=None):
        """Get circuit depth for given vectors."""
        if vectors is None:
            vectors = self.inserted_vectors
        if len(vectors) == 0:
            return 0
        qc = self.build_insert_circuit(vectors)
        return qc.depth()
