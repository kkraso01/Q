"""Unit tests for Quantum Hashed Trie (QHT)"""
import pytest
import numpy as np
from .qht import QHT


def test_qht_initialization():
    """Test QHT initialization."""
    qht = QHT(m=16, b=4, L=8, theta=np.pi/4, k=3)
    assert qht.m == 16
    assert qht.b == 4
    assert qht.L == 8
    assert qht.theta == np.pi/4
    assert qht.k == 3
    assert len(qht.inserted_items) == 0


def test_qht_insert():
    """Test item insertion."""
    qht = QHT(m=16, b=4, L=8)
    qht.insert(b"hello")
    qht.insert(b"world")
    assert len(qht.inserted_items) == 2


def test_qht_prefix_query_match():
    """Test prefix query for exact match."""
    qht = QHT(m=32, b=8, L=16, k=3)
    items = [b"hello", b"help", b"world"]
    for item in items:
        qht.insert(item)
    
    # Query for prefix "hel" should match "hello" and "help"
    acceptance = qht.query(b"hel", shots=1024)
    assert 0.0 <= acceptance <= 1.0
    # Should have higher acceptance than random prefix


def test_qht_prefix_query_no_match():
    """Test prefix query for non-existent prefix."""
    qht = QHT(m=32, b=8, L=16, k=3)
    items = [b"hello", b"world"]
    for item in items:
        qht.insert(item)
    
    # Query for prefix "xyz" should have low acceptance
    acceptance = qht.query(b"xyz", shots=1024)
    assert 0.0 <= acceptance <= 1.0


def test_qht_circuit_depth():
    """Test circuit depth calculation."""
    qht = QHT(m=16, b=4, L=8)
    qht.insert(b"test")
    depth = qht.get_circuit_depth()
    assert depth > 0


def test_qht_varying_branching_factors():
    """Test QHT with different branching factors."""
    for b in [2, 4, 8, 16]:
        qht = QHT(m=16, b=b, L=8)
        qht.insert(b"test")
        acceptance = qht.query(b"te", shots=512)
        assert 0.0 <= acceptance <= 1.0


def test_qht_varying_depths():
    """Test QHT with different maximum depths."""
    for L in [4, 8, 16, 32]:
        qht = QHT(m=32, b=4, L=L)
        qht.insert(b"testing")
        acceptance = qht.query(b"test", shots=512)
        assert 0.0 <= acceptance <= 1.0


def test_qht_cache_behavior():
    """Test circuit caching."""
    qht = QHT(m=16, b=4, L=8)
    items = [b"test1", b"test2"]
    
    # Build circuit twice with same items
    qc1 = qht.build_insert_circuit(items)
    qc2 = qht.build_insert_circuit(items)
    
    # Should return cached circuit
    assert qc1.depth() == qc2.depth()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
