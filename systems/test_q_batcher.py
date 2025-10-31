"""
Unit tests for Q-Batcher batch overlap optimization.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from systems.q_batcher import QBatcher


def test_q_batcher_init():
    """Test Q-Batcher initialization."""
    batcher = QBatcher(m=8, shots_per_pair=256)
    
    assert batcher.m == 8
    assert batcher.shots_per_pair == 256


def test_batch_overlap_test():
    """Test batch overlap computation."""
    batcher = QBatcher(m=4, shots_per_pair=128)
    
    query_phases = np.random.uniform(0, 2*np.pi, 4)
    candidate_phases = [
        np.random.uniform(0, 2*np.pi, 4) for _ in range(5)
    ]
    
    overlaps = batcher.batch_overlap_test(query_phases, candidate_phases)
    
    assert len(overlaps) == 5
    for overlap in overlaps:
        assert 0 <= overlap <= 1


def test_batch_overlap_identical():
    """Test overlap between identical phase patterns."""
    batcher = QBatcher(m=4, shots_per_pair=512)
    
    phases = np.zeros(4)  # All zero phases
    
    # Overlap with itself should be high
    overlaps = batcher.batch_overlap_test(phases, [phases])
    
    assert len(overlaps) == 1
    # Should be close to 1 (within measurement noise)
    assert overlaps[0] > 0.8


def test_batch_overlap_orthogonal():
    """Test overlap between orthogonal phase patterns."""
    batcher = QBatcher(m=4, shots_per_pair=512)
    
    phases1 = np.zeros(4)
    phases2 = np.full(4, np.pi)  # π phase → orthogonal
    
    overlaps = batcher.batch_overlap_test(phases1, [phases2])
    
    assert len(overlaps) == 1
    # Orthogonal states should have low overlap
    assert overlaps[0] < 0.5


def test_batch_with_batching():
    """Test processing with explicit batch size."""
    batcher = QBatcher(m=4, shots_per_pair=128)
    
    query_phases = np.random.uniform(0, 2*np.pi, 4)
    n_candidates = 10
    candidate_phases = [
        np.random.uniform(0, 2*np.pi, 4) for _ in range(n_candidates)
    ]
    
    # Process in batches of 3
    overlaps = batcher.batch_overlap_test(query_phases, candidate_phases, batch_size=3)
    
    assert len(overlaps) == n_candidates


def test_estimate_amortized_cost():
    """Test amortized cost estimation."""
    batcher = QBatcher(m=8, shots_per_pair=256)
    
    batch_sizes = [1, 5, 10, 20]
    prev_speedup = 1.0
    
    for batch_size in batch_sizes:
        cost_metrics = batcher.estimate_amortized_cost(batch_size)
        
        assert 'individual_cost' in cost_metrics
        assert 'batch_cost_per_item' in cost_metrics
        assert 'speedup' in cost_metrics
        assert cost_metrics['batch_size'] == batch_size
        
        # Speedup should increase with batch size
        if batch_size > 1:
            assert cost_metrics['speedup'] >= prev_speedup
        
        prev_speedup = cost_metrics['speedup']


def test_amortized_cost_scaling():
    """Test that amortized cost decreases with batch size."""
    batcher = QBatcher(m=8, shots_per_pair=256)
    
    cost_1 = batcher.estimate_amortized_cost(1)
    cost_10 = batcher.estimate_amortized_cost(10)
    
    # Per-item cost should be lower for larger batch
    assert cost_10['batch_cost_per_item'] < cost_1['batch_cost_per_item']
    assert cost_10['speedup'] > cost_1['speedup']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
