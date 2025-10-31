"""
Quantum KV-Cache Eviction Policy (Q-KV)

Quantum-aware cache eviction for transformer KV-cache management.
Uses quantum sketches to estimate key importance and make eviction decisions.
"""
import numpy as np
from collections import OrderedDict
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sim.utils import make_hash_functions, bitstring_to_int


class QKVPolicy:
    """Quantum KV-Cache eviction policy."""
    
    def __init__(self, m, k=3, theta=np.pi/4, cache_size=100):
        """
        Initialize Q-KV policy.
        
        Args:
            m: Number of qubits for quantum sketch
            k: Number of hash functions
            theta: Phase rotation angle
            cache_size: Maximum cache size
        """
        self.m = m
        self.k = k
        self.theta = theta
        self.cache_size = cache_size
        self.hash_functions = make_hash_functions(k)
        
        # Cache: key -> (value, access_count)
        self.cache = OrderedDict()
        
        # Quantum sketch state
        self._circuit_cache = {}
        self._key_history = []
    
    def _get_indices(self, key):
        """Get k qubit indices for key."""
        key_int = bitstring_to_int(key)
        return [h(key_int) % self.m for h in self.hash_functions]
    
    def _build_sketch_circuit(self, keys_with_weights):
        """
        Build quantum sketch circuit with weighted keys.
        
        Args:
            keys_with_weights: List of (key, weight) tuples
            
        Returns:
            QuantumCircuit
        """
        qc = QuantumCircuit(self.m)
        qc.h(range(self.m))
        
        # Apply phase rotations weighted by access frequency
        for key, weight in keys_with_weights:
            indices = self._get_indices(key)
            weighted_theta = self.theta * np.log1p(weight)
            for idx in indices:
                qc.rz(weighted_theta, idx)
        
        return qc
    
    def estimate_importance(self, key, shots=256, noise_level=0.0):
        """
        Estimate importance score for a key using quantum overlap.
        
        Args:
            key: Cache key to estimate
            shots: Number of measurement shots
            noise_level: Noise level
            
        Returns:
            float: Importance score (higher = more important)
        """
        if key not in self.cache:
            return 0.0
        
        # Build sketch with all cached keys and their access counts
        keys_with_weights = [
            (k, self.cache[k][1]) for k in self.cache.keys()
        ]
        
        qc_base = self._build_sketch_circuit(keys_with_weights)
        
        # Build query circuit for specific key
        qc_query = qc_base.copy()
        indices = self._get_indices(key)
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
        
        # Importance = overlap with aggregate sketch
        zero_state = '0' * self.m
        importance = counts.get(zero_state, 0) / shots
        
        return importance
    
    def get(self, key):
        """
        Get value from cache and update access pattern.
        
        Args:
            key: Cache key
            
        Returns:
            value or None if not in cache
        """
        if key in self.cache:
            value, access_count = self.cache[key]
            # Update access count
            self.cache[key] = (value, access_count + 1)
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self._key_history.append(key)
            return value
        return None
    
    def put(self, key, value):
        """
        Put key-value pair into cache with quantum-aware eviction.
        
        Args:
            key: Cache key
            value: Cache value
        """
        # If key exists, update it
        if key in self.cache:
            _, access_count = self.cache[key]
            self.cache[key] = (value, access_count + 1)
            self.cache.move_to_end(key)
            return
        
        # If cache is full, evict using quantum importance
        if len(self.cache) >= self.cache_size:
            self._evict_quantum()
        
        # Insert new key
        self.cache[key] = (value, 1)
        self._key_history.append(key)
    
    def _evict_quantum(self):
        """Evict least important key using quantum importance estimation."""
        if len(self.cache) == 0:
            return
        
        # Sample candidates for eviction (bottom 20% by access count)
        sorted_keys = sorted(
            self.cache.keys(),
            key=lambda k: self.cache[k][1]
        )
        num_candidates = max(1, len(sorted_keys) // 5)
        candidates = sorted_keys[:num_candidates]
        
        # Estimate quantum importance for each candidate
        importances = {}
        for key in candidates:
            importance = self.estimate_importance(key, shots=128)
            importances[key] = importance
        
        # Evict key with lowest importance
        evict_key = min(importances, key=importances.get)
        del self.cache[evict_key]
    
    def _evict_lru(self):
        """Evict least recently used key (baseline)."""
        if len(self.cache) > 0:
            self.cache.popitem(last=False)
    
    def _evict_lfu(self):
        """Evict least frequently used key (baseline)."""
        if len(self.cache) > 0:
            lfu_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[lfu_key]
    
    def get_hit_rate(self):
        """Calculate cache hit rate from history."""
        if len(self._key_history) == 0:
            return 0.0
        
        hits = sum(1 for key in self._key_history if key in self.cache)
        return hits / len(self._key_history)
    
    def clear(self):
        """Clear cache and history."""
        self.cache.clear()
        self._key_history.clear()
        self._circuit_cache.clear()


class LRUPolicy:
    """Baseline LRU cache policy for comparison."""
    
    def __init__(self, cache_size=100):
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self._key_history = []
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            self._key_history.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.cache_size:
                self.cache.popitem(last=False)
        self.cache[key] = value
        self._key_history.append(key)
    
    def get_hit_rate(self):
        if len(self._key_history) == 0:
            return 0.0
        hits = sum(1 for key in self._key_history if key in self.cache)
        return hits / len(self._key_history)
    
    def clear(self):
        self.cache.clear()
        self._key_history.clear()


class LFUPolicy:
    """Baseline LFU cache policy for comparison."""
    
    def __init__(self, cache_size=100):
        self.cache_size = cache_size
        self.cache = {}  # key -> (value, access_count)
        self._key_history = []
    
    def get(self, key):
        if key in self.cache:
            value, count = self.cache[key]
            self.cache[key] = (value, count + 1)
            self._key_history.append(key)
            return value
        return None
    
    def put(self, key, value):
        if key in self.cache:
            _, count = self.cache[key]
            self.cache[key] = (value, count + 1)
        else:
            if len(self.cache) >= self.cache_size:
                # Evict LFU
                lfu_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[lfu_key]
            self.cache[key] = (value, 1)
        self._key_history.append(key)
    
    def get_hit_rate(self):
        if len(self._key_history) == 0:
            return 0.0
        hits = sum(1 for key in self._key_history if key in self.cache)
        return hits / len(self._key_history)
    
    def clear(self):
        self.cache.clear()
        self._key_history.clear()
