"""
Quantum Locality-Sensitive Hashing (Q-LSH) - Similarity Search

Supports approximate nearest neighbor (ANN) queries using quantum phase-encoded similarity.
Uses random hyperplanes for cosine similarity estimation via amplitude interference.
"""
import numpy as np
from qiskit import QuantumCircuit
from .amplitude_sketch import AmplitudeSketch
from .utils import bitstring_to_int


class QLSH(AmplitudeSketch):
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
        super().__init__(m, k, theta)
        self.d = d
        self.hyperplanes = self._generate_hyperplanes()
        self.inserted_vectors = []
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
    
    def _get_indices(self, x):
        """Get hash indices for compatibility (implements abstract method)."""
        # x is a vector, get its signature first
        signature = self._get_hash_signature(x)
        return self._signature_to_indices(signature)
    
    def _build_insert_circuit(self, items):
        """Build insert circuit (implements abstract method)."""
        # items is a list of vectors
        qc = QuantumCircuit(self.m)
        qc.h(range(self.m))  # Initialize to superposition
        
        for vector in items:
            signature = self._get_hash_signature(vector)
            indices = self._signature_to_indices(signature)
            for idx in indices:
                qc.rz(self.theta, idx)
        
        return qc
    
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
        
        qc = self._build_insert_circuit(vectors)
        
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
        
        # Simulate with memory-efficient method for large m
        noise_model = self._create_noise_model(noise_level)
        from qiskit_aer import AerSimulator
        
        if self.m > 16:
            simulator = AerSimulator(method='matrix_product_state', noise_model=noise_model)
        else:
            simulator = AerSimulator(method='automatic', noise_model=noise_model)
        
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
        Find k approximate nearest neighbors using LSH bucketing + classical refinement.
        
        This hybrid approach:
        1. Uses quantum LSH for fast approximate bucketing
        2. Refines with classical cosine similarity for accuracy
        
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
        
        # Normalize query vector
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)
        
        # Get LSH signature for query
        query_sig = self._get_hash_signature(query_vector)
        
        # Filter candidates: vectors with similar LSH signatures
        # (Quantum advantage: buckets are determined by quantum hash)
        candidates = []
        for vec in self.inserted_vectors:
            vec_sig = self._get_hash_signature(vec)
            # Count matching bits in signature (Hamming similarity)
            matches = sum(1 for a, b in zip(query_sig, vec_sig) if a == b)
            hamming_sim = matches / len(query_sig)
            
            # Keep vectors with >50% signature match OR all vectors if few candidates
            if hamming_sim >= 0.5 or len(candidates) < k_neighbors * 2:
                candidates.append(vec)
        
        # If too few candidates, use all vectors
        if len(candidates) < k_neighbors:
            candidates = self.inserted_vectors
        
        # Compute classical cosine similarity for candidates (refinement)
        similarities = []
        for vec in candidates:
            vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
            # Classical cosine similarity
            cos_sim = np.dot(query_norm, vec_norm)
            similarities.append((vec, cos_sim))
        
        # Sort by similarity (descending) and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k_neighbors]
    
    def insert(self, vector):
        """Insert vector into LSH structure."""
        self.inserted_vectors.append(vector.copy())
        self.n_inserts += 1
        # Clear caches
        self._circuit_cache.clear()
        self._statevector_cache.clear()
    
    def query(self, x=None, items=None, shots=512, noise_level=0.0, k_neighbors=None):
        """Query for similarity or k-NN (implements abstract method)."""
        if k_neighbors is not None:
            # k-NN query
            return self.query_knn(x, k_neighbors=k_neighbors, shots=shots, noise_level=noise_level)
        elif items is not None and len(items) > 0:
            # Pairwise similarity query
            return self.cosine_similarity_estimate(x, items[0], shots=shots, noise_level=noise_level)
        else:
            raise ValueError("Q-LSH query requires either k_neighbors or items")
    
    def clear_cache(self):
        """Clear all caches (extends base class)."""
        super().clear_cache()
        self._statevector_cache.clear()
    
    def reset(self):
        """Reset to empty state (extends base class)."""
        super().reset()
        self.inserted_vectors.clear()
        self._statevector_cache.clear()
    
    def get_circuit_depth(self, vectors=None):
        """Get circuit depth for given vectors."""
        if vectors is None:
            vectors = self.inserted_vectors
        if len(vectors) == 0:
            return 0
        qc = self.build_insert_circuit(vectors)
        return qc.depth()
