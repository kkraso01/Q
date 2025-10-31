"""Unit tests for QSimHash."""
import numpy as np
from sim.q_simhash import QSimHash

def test_encode_vector_bytes():
    qsh = QSimHash(m=8, k=4)
    vec = b"A"  # 0b01000001
    bits = qsh.encode_vector(vec)
    assert isinstance(bits, np.ndarray)
    assert bits.shape[0] >= 8

def test_build_encoding_circuit():
    qsh = QSimHash(m=8, k=4)
    vec = b"A"
    qc = qsh.build_encoding_circuit(vec)
    assert qc.num_qubits == 8
    assert qc.depth() > 0

def test_similarity_identical():
    np.random.seed(42)
    qsh = QSimHash(m=8, k=4)
    vec = b"A"
    sim = qsh.similarity(vec, vec, shots=128)
    assert 0 <= sim <= 1
    assert sim > 0.3  # Should be high for identical

def test_similarity_orthogonal():
    np.random.seed(42)
    qsh = QSimHash(m=8, k=4)
    vec1 = b"A"
    vec2 = b"\xFF"  # 0b11111111
    sim = qsh.similarity(vec1, vec2, shots=128)
    assert 0 <= sim <= 1
