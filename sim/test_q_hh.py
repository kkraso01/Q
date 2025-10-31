"""Unit tests for Quantum Heavy Hitters (Q-HH)"""
import pytest
import numpy as np
from collections import Counter
from .q_hh import QHH


def test_qhh_initialization():
    """Test Q-HH initialization."""
    qhh = QHH(m=16, k=3, theta=np.pi/4)
    assert qhh.m == 16
    assert qhh.k == 3
    assert qhh.theta == np.pi/4
    assert len(qhh.stream) == 0


def test_qhh_insert():
    """Test item insertion into stream."""
    qhh = QHH(m=16)
    qhh.insert(b"item1")
    qhh.insert(b"item2")
    qhh.insert(b"item1")
    assert len(qhh.stream) == 3


def test_qhh_frequency_estimation():
    """Test frequency estimation for single item."""
    qhh = QHH(m=32, k=3)
    items = [b"a"] * 10 + [b"b"] * 5 + [b"c"] * 2
    for item in items:
        qhh.insert(item)
    
    # Estimate frequency of most frequent item
    freq_a = qhh.estimate_frequency(b"a", shots=1024)
    assert freq_a >= 0


def test_qhh_top_k_basic():
    """Test top-k retrieval."""
    qhh = QHH(m=32, k=3)
    items = [b"a"] * 10 + [b"b"] * 5 + [b"c"] * 2
    for item in items:
        qhh.insert(item)
    
    top_3 = qhh.top_k(3, shots=512)
    
    assert len(top_3) <= 3
    # Should return items with frequency estimates
    for item, freq in top_3:
        assert freq >= 0


def test_qhh_top_k_ordering():
    """Test that top-k returns items in descending frequency order."""
    qhh = QHH(m=32, k=3)
    items = [b"a"] * 20 + [b"b"] * 10 + [b"c"] * 5
    for item in items:
        qhh.insert(item)
    
    top_3 = qhh.top_k(3, shots=1024)
    
    # Frequencies should be in descending order
    if len(top_3) >= 2:
        for i in range(len(top_3) - 1):
            assert top_3[i][1] >= top_3[i+1][1]


def test_qhh_empty_stream():
    """Test top-k on empty stream."""
    qhh = QHH(m=16)
    top_k = qhh.top_k(5)
    assert len(top_k) == 0


def test_qhh_single_item():
    """Test top-k with single unique item."""
    qhh = QHH(m=16)
    items = [b"only"] * 10
    for item in items:
        qhh.insert(item)
    
    top_1 = qhh.top_k(1, shots=512)
    assert len(top_1) == 1
    assert top_1[0][0] == b"only"


def test_qhh_varying_memory_sizes():
    """Test Q-HH with different memory sizes."""
    items = [b"a"] * 5 + [b"b"] * 3 + [b"c"] * 1
    for m in [8, 16, 32, 64]:
        qhh = QHH(m=m, k=3)
        for item in items:
            qhh.insert(item)
        top_2 = qhh.top_k(2, shots=512)
        assert len(top_2) <= 2


def test_qhh_noise_robustness():
    """Test Q-HH with noise."""
    qhh = QHH(m=32, k=3)
    items = [b"a"] * 10 + [b"b"] * 5
    for item in items:
        qhh.insert(item)
    
    # Test with noise
    top_2_noisy = qhh.top_k(2, shots=1024, noise_level=0.001)
    assert len(top_2_noisy) <= 2


def test_qhh_true_frequencies():
    """Test true frequency counter."""
    qhh = QHH(m=16)
    items = [b"a"] * 3 + [b"b"] * 2 + [b"c"] * 1
    for item in items:
        qhh.insert(item)
    
    true_freqs = qhh.get_true_frequencies()
    assert true_freqs[b"a"] == 3
    assert true_freqs[b"b"] == 2
    assert true_freqs[b"c"] == 1


def test_qhh_cache_behavior():
    """Test circuit caching."""
    qhh = QHH(m=16)
    items = [b"a", b"b", b"c"]
    
    # Build circuit twice
    qc1 = qhh.build_circuit(items)
    qc2 = qhh.build_circuit(items)
    
    # Should use cache
    assert qc1.depth() == qc2.depth()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
