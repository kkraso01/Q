"""
Unit tests for AmplitudeSketch base class and composition.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sim.amplitude_sketch import AmplitudeSketch, SerialComposition
from qiskit import QuantumCircuit


class ConcreteSketch(AmplitudeSketch):
    """Concrete implementation for testing."""
    
    def __init__(self, m=16, k=3, theta=np.pi/4):
        super().__init__(m, k, theta)
        self.inserted_items = []
    
    def insert(self, x: bytes):
        """Simple insert tracking."""
        self.inserted_items.append(x)
        self.n_inserts += 1
    
    def query(self, y: bytes, shots=512, noise_level=0.0):
        """Simple membership check."""
        if y in self.inserted_items:
            return 0.9  # High overlap for members
        else:
            return 0.1  # Low overlap for non-members
    
    def _build_insert_circuit(self, x: bytes):
        """Build simple Rz circuit."""
        qc = QuantumCircuit(self.m)
        indices = self._hash_to_indices(x)
        for idx in indices:
            qc.rz(self.theta, idx)
        return qc


def test_amplitude_sketch_init():
    """Test AmplitudeSketch initialization."""
    sketch = ConcreteSketch(m=16, k=3, theta=np.pi/4)
    
    assert sketch.m == 16
    assert sketch.k == 3
    assert sketch.theta == np.pi/4
    assert sketch.n_inserts == 0
    assert len(sketch._circuit_cache) == 0


def test_amplitude_sketch_insert():
    """Test insert operation."""
    sketch = ConcreteSketch(m=16, k=3)
    
    sketch.insert(b"test")
    
    assert sketch.n_inserts == 1
    assert b"test" in sketch.inserted_items


def test_amplitude_sketch_query():
    """Test query operation."""
    sketch = ConcreteSketch(m=16, k=3)
    
    sketch.insert(b"member")
    
    # Query for member
    score_member = sketch.query(b"member", shots=512)
    assert score_member > 0.5
    
    # Query for non-member
    score_non = sketch.query(b"nonmember", shots=512)
    assert score_non < 0.5


def test_amplitude_sketch_hash_to_indices():
    """Test hashing to qubit indices."""
    sketch = ConcreteSketch(m=16, k=3)
    
    indices = sketch._hash_to_indices(b"test")
    
    assert len(indices) == 3
    for idx in indices:
        assert 0 <= idx < 16


def test_amplitude_sketch_build_insert_circuit():
    """Test circuit construction."""
    sketch = ConcreteSketch(m=8, k=2, theta=np.pi/4)
    
    circuit = sketch._build_insert_circuit(b"test")
    
    assert circuit.num_qubits == 8
    # Should have k Rz gates
    assert circuit.depth() >= 1


def test_amplitude_sketch_error_bound():
    """Test error bound estimation."""
    sketch = ConcreteSketch(m=32, k=4)
    
    # Insert some items
    for i in range(10):
        sketch.insert(f"item{i}".encode())
    
    alpha, beta = sketch.error_bound()
    
    assert 0 <= alpha <= 1
    assert 0 <= beta <= 1
    assert beta <= alpha  # Typically β ≤ α


def test_amplitude_sketch_get_memory_size():
    """Test memory size getter."""
    sketch = ConcreteSketch(m=32, k=3)
    
    assert sketch.get_memory_size() == 32


def test_amplitude_sketch_get_circuit_depth():
    """Test circuit depth estimation."""
    sketch = ConcreteSketch(m=16, k=3)
    
    depth = sketch.get_circuit_depth(b"test")
    
    assert depth >= 3  # At least k gates


def test_amplitude_sketch_reset():
    """Test sketch reset."""
    sketch = ConcreteSketch(m=16, k=3)
    
    sketch.insert(b"test1")
    sketch.insert(b"test2")
    
    assert sketch.n_inserts == 2
    
    sketch.reset()
    
    assert sketch.n_inserts == 0
    assert len(sketch._circuit_cache) == 0


def test_amplitude_sketch_clear_cache():
    """Test cache clearing."""
    sketch = ConcreteSketch(m=16, k=3)
    
    # Manually add to cache
    sketch._circuit_cache[b"test"] = QuantumCircuit(16)
    
    assert len(sketch._circuit_cache) == 1
    
    sketch.clear_cache()
    
    assert len(sketch._circuit_cache) == 0


def test_amplitude_sketch_repr():
    """Test string representation."""
    sketch = ConcreteSketch(m=16, k=3, theta=np.pi/4)
    
    repr_str = repr(sketch)
    
    assert "ConcreteSketch" in repr_str
    assert "m=16" in repr_str
    assert "k=3" in repr_str


def test_amplitude_sketch_get_stats():
    """Test statistics getter."""
    sketch = ConcreteSketch(m=32, k=4)
    
    for i in range(5):
        sketch.insert(f"item{i}".encode())
    
    stats = sketch.get_stats()
    
    assert stats['class'] == 'ConcreteSketch'
    assert stats['m'] == 32
    assert stats['k'] == 4
    assert stats['n_inserts'] == 5
    assert 'load_factor' in stats
    assert 'estimated_alpha' in stats


def test_amplitude_sketch_compose_not_implemented():
    """Test that default compose raises NotImplementedError."""
    sketch1 = ConcreteSketch(m=16, k=3)
    sketch2 = ConcreteSketch(m=16, k=3)
    
    with pytest.raises(NotImplementedError):
        sketch1.compose(sketch2)


def test_serial_composition_init():
    """Test SerialComposition initialization."""
    sketch1 = ConcreteSketch(m=16, k=3)
    sketch2 = ConcreteSketch(m=16, k=3)
    
    composition = SerialComposition([sketch1, sketch2])
    
    assert composition.n_stages == 2
    assert len(composition.sketches) == 2


def test_serial_composition_query():
    """Test query through serial composition."""
    sketch1 = ConcreteSketch(m=16, k=3)
    sketch2 = ConcreteSketch(m=16, k=3)
    
    # Insert into both stages
    sketch1.insert(b"member")
    sketch2.insert(b"member")
    
    composition = SerialComposition([sketch1, sketch2])
    
    # Query for member (should pass both stages)
    score_member = composition.query(b"member", shots=512)
    assert score_member > 0.5
    
    # Query for non-member (should fail at least one stage)
    score_non = composition.query(b"nonmember", shots=512)
    assert score_non < 0.5


def test_serial_composition_error_bound():
    """Test composed error bounds."""
    sketch1 = ConcreteSketch(m=32, k=4)
    sketch2 = ConcreteSketch(m=32, k=4)
    
    # Insert to establish load factors
    for i in range(10):
        sketch1.insert(f"item{i}".encode())
        sketch2.insert(f"item{i}".encode())
    
    composition = SerialComposition([sketch1, sketch2])
    
    alpha_total, beta_total = composition.error_bound()
    
    assert 0 <= alpha_total <= 1
    assert 0 <= beta_total <= 1
    
    # Composed error should be higher than individual
    alpha1, _ = sketch1.error_bound()
    assert alpha_total >= alpha1


def test_serial_composition_get_total_memory():
    """Test total memory calculation."""
    sketch1 = ConcreteSketch(m=16, k=3)
    sketch2 = ConcreteSketch(m=32, k=3)
    sketch3 = ConcreteSketch(m=24, k=3)
    
    composition = SerialComposition([sketch1, sketch2, sketch3])
    
    total_memory = composition.get_total_memory()
    
    # Should be max across stages
    assert total_memory == 32


def test_serial_composition_early_termination():
    """Test early termination in serial composition."""
    sketch1 = ConcreteSketch(m=16, k=3)
    sketch2 = ConcreteSketch(m=16, k=3)
    
    # Only insert into first stage
    sketch1.insert(b"test")
    # Second stage remains empty
    
    composition = SerialComposition([sketch1, sketch2])
    
    # Query should terminate early at second stage
    score = composition.query(b"test", shots=512)
    
    # Score should be low due to second stage failing
    assert score < 0.5


@pytest.mark.parametrize("m,k", [(16, 3), (32, 4), (64, 5)])
def test_amplitude_sketch_various_sizes(m, k):
    """Test with various memory and hash function counts."""
    sketch = ConcreteSketch(m=m, k=k)
    
    assert sketch.m == m
    assert sketch.k == k
    
    sketch.insert(b"test")
    score = sketch.query(b"test", shots=256)
    
    assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
