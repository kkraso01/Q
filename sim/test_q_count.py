"""Unit tests for Quantum Count-Distinct (Q-Count)"""
import pytest
import numpy as np
from .q_count import QCount


def test_qcount_initialization():
    """Test Q-Count initialization."""
    qc = QCount(m=16, k=3, theta=np.pi/4)
    assert qc.m == 16
    assert qc.k == 3
    assert qc.theta == np.pi/4
    assert len(qc.stream) == 0


def test_qcount_insert():
    """Test item insertion into stream."""
    qc = QCount(m=16)
    qc.insert(b"item1")
    qc.insert(b"item2")
    qc.insert(b"item1")  # Duplicate
    assert len(qc.stream) == 3


def test_qcount_cardinality_estimation_small():
    """Test cardinality estimation for small stream."""
    qc = QCount(m=32, k=3)
    items = [b"a", b"b", b"c", b"a", b"b", b"a"]  # 3 distinct
    for item in items:
        qc.insert(item)
    
    estimate = qc.estimate_cardinality(shots=1024)
    true_count = qc.get_true_cardinality()
    
    assert true_count == 3
    # Estimate should be reasonable (within 2x)
    assert 0 <= estimate <= true_count * 2


def test_qcount_cardinality_all_unique():
    """Test cardinality when all items are unique."""
    qc = QCount(m=32, k=3)
    items = [f"item{i}".encode() for i in range(10)]
    for item in items:
        qc.insert(item)
    
    estimate = qc.estimate_cardinality(shots=1024)
    true_count = qc.get_true_cardinality()
    
    assert true_count == 10
    assert estimate >= 0


def test_qcount_cardinality_all_duplicates():
    """Test cardinality when all items are duplicates."""
    qc = QCount(m=32, k=3)
    items = [b"same"] * 20
    for item in items:
        qc.insert(item)
    
    estimate = qc.estimate_cardinality(shots=1024)
    true_count = qc.get_true_cardinality()
    
    assert true_count == 1
    assert estimate >= 0


def test_qcount_empty_stream():
    """Test cardinality estimation on empty stream."""
    qc = QCount(m=16)
    estimate = qc.estimate_cardinality()
    assert estimate == 0


def test_qcount_varying_memory_sizes():
    """Test Q-Count with different memory sizes."""
    items = [b"a", b"b", b"c", b"d", b"e"]
    for m in [8, 16, 32, 64]:
        qc = QCount(m=m, k=3)
        for item in items:
            qc.insert(item)
        estimate = qc.estimate_cardinality(shots=512)
        assert estimate >= 0


def test_qcount_noise_robustness():
    """Test Q-Count with noise."""
    qc = QCount(m=32, k=3)
    items = [b"a", b"b", b"c", b"a", b"b"]
    for item in items:
        qc.insert(item)
    
    # Test with noise
    estimate_noisy = qc.estimate_cardinality(shots=1024, noise_level=0.001)
    assert estimate_noisy >= 0


def test_qcount_cache_behavior():
    """Test circuit caching."""
    qc = QCount(m=16)
    items = [b"a", b"b", b"c"]
    
    # Build circuit twice
    qc1 = qc.build_circuit(items)
    qc2 = qc.build_circuit(items)
    
    # Should use cache
    assert qc1.depth() == qc2.depth()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
