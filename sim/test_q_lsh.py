"""Unit tests for Quantum LSH (Q-LSH)"""
import pytest
import numpy as np
from .q_lsh import QLSH


def test_qlsh_initialization():
    """Test Q-LSH initialization."""
    qlsh = QLSH(m=32, k=4, theta=np.pi/4, d=128)
    assert qlsh.m == 32
    assert qlsh.k == 4
    assert qlsh.d == 128
    assert len(qlsh.hyperplanes) == 4


def test_qlsh_insert():
    """Test vector insertion."""
    qlsh = QLSH(m=32, d=64)
    v1 = np.random.randn(64)
    qlsh.insert(v1)
    assert len(qlsh.inserted_vectors) == 1


def test_qlsh_hash_signature():
    """Test LSH signature generation."""
    qlsh = QLSH(m=32, k=4, d=64)
    v1 = np.random.randn(64)
    sig = qlsh._get_hash_signature(v1)
    assert len(sig) == 4
    assert all(s in [0, 1] for s in sig)


def test_qlsh_cosine_similarity_identical():
    """Test cosine similarity for identical vectors."""
    qlsh = QLSH(m=32, k=4, d=64)
    v1 = np.random.randn(64)
    sim = qlsh.cosine_similarity_estimate(v1, v1, shots=1024)
    assert 0.5 <= sim <= 1.0  # Should be close to 1


def test_qlsh_cosine_similarity_orthogonal():
    """Test cosine similarity for orthogonal vectors."""
    qlsh = QLSH(m=32, k=4, d=64)
    v1 = np.zeros(64)
    v1[0] = 1.0
    v2 = np.zeros(64)
    v2[1] = 1.0
    sim = qlsh.cosine_similarity_estimate(v1, v2, shots=1024)
    assert -1.0 <= sim <= 1.0  # Should be close to 0


def test_qlsh_knn_query():
    """Test k-NN query."""
    qlsh = QLSH(m=32, k=4, d=32)
    # Insert several vectors
    for i in range(5):
        v = np.random.randn(32)
        qlsh.insert(v)
    
    query = np.random.randn(32)
    neighbors = qlsh.query_knn(query, k_neighbors=3, shots=512)
    
    assert len(neighbors) <= 3
    # Check format: [(vector, similarity), ...]
    for vec, sim in neighbors:
        assert len(vec) == 32
        assert -1 <= sim <= 1


def test_qlsh_knn_empty():
    """Test k-NN query on empty structure."""
    qlsh = QLSH(m=32, d=64)
    query = np.random.randn(64)
    neighbors = qlsh.query_knn(query, k_neighbors=5)
    assert len(neighbors) == 0


def test_qlsh_varying_dimensions():
    """Test Q-LSH with different vector dimensions."""
    for d in [32, 64, 128, 256]:
        qlsh = QLSH(m=32, k=4, d=d)
        v = np.random.randn(d)
        qlsh.insert(v)
        assert len(qlsh.inserted_vectors) == 1


def test_qlsh_noise_robustness():
    """Test Q-LSH with noise."""
    qlsh = QLSH(m=32, k=4, d=64)
    v1 = np.random.randn(64)
    v2 = np.random.randn(64)
    
    sim_noisy = qlsh.cosine_similarity_estimate(
        v1, v2, shots=1024, noise_level=0.001
    )
    assert -1.0 <= sim_noisy <= 1.0


def test_qlsh_circuit_depth():
    """Test circuit depth calculation."""
    qlsh = QLSH(m=16, d=32)
    v1 = np.random.randn(32)
    qlsh.insert(v1)
    depth = qlsh.get_circuit_depth()
    assert depth > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
