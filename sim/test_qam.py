"""Unit tests for QAM implementation."""
import pytest
import numpy as np
from sim.qam import QAM, create_noise_model
from sim.utils import splitmix64, make_hash_functions, bitstring_to_int


class TestUtils:
    """Test utility functions."""
    
    def test_splitmix64_deterministic(self):
        """Verify hash function is deterministic."""
        x = 42
        seed = 7
        h1 = splitmix64(x, seed)
        h2 = splitmix64(x, seed)
        assert h1 == h2, "Hash should be deterministic"
    
    def test_splitmix64_different_seeds(self):
        """Verify different seeds produce different hashes."""
        x = 42
        h1 = splitmix64(x, seed=0)
        h2 = splitmix64(x, seed=1)
        assert h1 != h2, "Different seeds should produce different hashes"
    
    def test_make_hash_functions(self):
        """Verify hash function generation."""
        k = 3
        hash_fns = make_hash_functions(k)
        assert len(hash_fns) == k, f"Should generate {k} hash functions"
        
        # Test independence
        x = 12345
        hashes = [h(x) for h in hash_fns]
        assert len(set(hashes)) == k, "Hash functions should be independent"
    
    def test_bitstring_to_int(self):
        """Test bitstring conversion."""
        s = b"test"
        result = bitstring_to_int(s)
        assert isinstance(result, int), "Should return integer"
        assert result > 0, "Should be positive"
        
        # Test string input
        s2 = "test"
        result2 = bitstring_to_int(s2)
        assert result == result2, "Bytes and string should produce same result"


class TestQAM:
    """Test QAM data structure."""
    
    def test_initialization(self):
        """Test QAM initialization."""
        m, k = 16, 3
        qam = QAM(m=m, k=k)
        assert qam.m == m, "Incorrect m"
        assert qam.k == k, "Incorrect k"
        assert len(qam.hash_functions) == k, "Should have k hash functions"
    
    def test_get_indices(self):
        """Test index generation from items."""
        np.random.seed(42)
        qam = QAM(m=16, k=3)
        
        item = b"test"
        indices = qam._get_indices(item)
        
        assert len(indices) == 3, "Should generate k indices"
        assert all(0 <= idx < 16 for idx in indices), "Indices should be in range [0, m)"
    
    def test_build_insert_circuit(self):
        """Test circuit construction for insertions."""
        qam = QAM(m=8, k=2)
        items = [b"apple", b"banana"]
        
        qc = qam.build_insert_circuit(items)
        
        assert qc.num_qubits == 8, "Circuit should have m qubits"
        assert qc.depth() > 0, "Circuit should have non-zero depth"
    
    def test_query_statevector(self):
        """Test statevector query (exact)."""
        np.random.seed(42)
        qam = QAM(m=16, k=3, theta=np.pi/4)
        
        items = [b"apple", b"banana"]
        
        # Query inserted item
        prob_in = qam.query_statevector(items, b"apple")
        assert 0 <= prob_in <= 1, "Probability should be in [0, 1]"
        
        # Query non-inserted item
        prob_out = qam.query_statevector(items, b"grape")
        assert 0 <= prob_out <= 1, "Probability should be in [0, 1]"
        
        # Inserted item should have higher probability
        assert prob_in > prob_out, "Inserted item should have higher acceptance"
    
    def test_query_with_shots(self):
        """Test query with measurement shots."""
        np.random.seed(42)
        qam = QAM(m=16, k=3, theta=np.pi/4)
        
        items = [b"apple"]
        query_item = b"apple"
        
        exp = qam.query(items, query_item, shots=512)
        
        assert 0 <= exp <= 1, "Expectation should be in [0, 1]"
        assert exp > 0.3, "Should detect inserted item with high probability"
    
    def test_no_cloning_constraint(self):
        """Verify circuit respects no-cloning (uses unitary operations)."""
        qam = QAM(m=8, k=2)
        items = [b"test"]
        
        qc = qam.build_insert_circuit(items)
        
        # Check all operations are unitary (h, rz) or measurements
        allowed_ops = {'h', 'rz', 'measure', 'barrier'}
        for instr in qc.data:
            assert instr.operation.name in allowed_ops, f"Non-unitary operation: {instr.operation.name}"
    
    def test_false_positive_rate(self):
        """Test false positive rate is reasonable."""
        np.random.seed(42)
        qam = QAM(m=16, k=3, theta=np.pi/4)  # Reduced m to fit in memory
        
        # Insert items
        inserted = [bytes(np.random.randint(0, 256, 8)) for _ in range(5)]
        
        # Test non-inserted items
        test_items = [bytes(np.random.randint(0, 256, 8)) for _ in range(10)]
        
        false_positives = 0
        for item in test_items:
            if item not in inserted:
                exp = qam.query(inserted, item, shots=256)
                if exp >= 0.5:
                    false_positives += 1
        
        fp_rate = false_positives / len(test_items)
        assert fp_rate < 0.8, f"False positive rate too high: {fp_rate}"
    
    def test_noise_model(self):
        """Test noise model creation and application."""
        noise_model = create_noise_model(error_rate=0.01)
        assert noise_model is not None, "Should create noise model"
        
        qam = QAM(m=16, k=3)
        items = [b"test"]
        query_item = b"test"
        
        # Should run without error
        exp = qam.query(items, query_item, shots=128, noise_model=noise_model)
        assert 0 <= exp <= 1, "Expectation should be valid with noise"
    
    def test_batch_query(self):
        """Test batch query functionality."""
        np.random.seed(42)
        qam = QAM(m=16, k=3)
        
        items = [b"apple", b"banana"]
        query_items = [b"apple", b"grape"]
        
        results = qam.batch_query(items, query_items, shots=256, threshold=0.5)
        
        assert len(results) == len(query_items), "Should return result for each query"
        
        for item, exp, is_member in results:
            assert 0 <= exp <= 1, "Expectation should be in [0, 1]"
            assert isinstance(is_member, (bool, np.bool_)), "is_member should be boolean"


class TestReproducibility:
    """Test reproducibility of results."""
    
    def test_deterministic_with_seed(self):
        """Verify results are deterministic with fixed seed."""
        np.random.seed(42)
        qam1 = QAM(m=16, k=3)
        items = [b"test"]
        exp1 = qam1.query_statevector(items, b"test")
        
        np.random.seed(42)
        qam2 = QAM(m=16, k=3)
        exp2 = qam2.query_statevector(items, b"test")
        
        assert exp1 == exp2, "Results should be reproducible with same seed"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
