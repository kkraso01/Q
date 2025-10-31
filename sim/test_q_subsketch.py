"""Unit tests for QSubSketch."""
import numpy as np
from sim.q_subsketch import QSubSketch

def test_rolling_hashes():
    qss = QSubSketch(m=8, k=2, L=3)
    text = b"abcdefg"
    hashes = qss.rolling_hashes(text)
    assert len(hashes) == len(text) - qss.L + 1
    for hvec in hashes:
        assert len(hvec) == qss.k
        assert all(0 <= idx < qss.m for idx in hvec)

def test_build_sketch_circuit():
    qss = QSubSketch(m=8, k=2, L=3)
    text = b"abcdefg"
    qc = qss.build_sketch_circuit(text)
    assert qc.num_qubits == 8
    assert qc.depth() > 0

def test_query_pattern_in_text():
    np.random.seed(42)
    qss = QSubSketch(m=8, k=2, L=3)
    text = b"abcdefghij"
    pattern = b"def"
    exp = qss.query(text, pattern, shots=128)
    assert 0 <= exp <= 1

def test_query_pattern_not_in_text():
    np.random.seed(42)
    qss = QSubSketch(m=8, k=2, L=3)
    text = b"abcdefghij"
    pattern = b"xyz"
    exp = qss.query(text, pattern, shots=128)
    assert 0 <= exp <= 1
