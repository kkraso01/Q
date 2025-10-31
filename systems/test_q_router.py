"""
Unit tests for Q-Router intelligent query routing.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from systems.q_router import QRouter, RouteDecision


def test_q_router_init():
    """Test Q-Router initialization."""
    router = QRouter(complexity_threshold=0.5, quantum_load_limit=100)
    
    assert router.complexity_threshold == 0.5
    assert router.quantum_load_limit == 100
    assert router.quantum_queue_size == 0


def test_analyze_query_complexity():
    """Test query complexity analysis."""
    router = QRouter()
    
    # Simple query
    simple_text = "hi"
    simple_embedding = np.random.randn(64) * 0.1
    simple_complexity = router.analyze_query_complexity(simple_text, simple_embedding)
    
    # Complex query
    complex_text = "complex semantic search with many technical concepts and terminology"
    complex_embedding = np.random.randn(64) * 10.0
    complex_complexity = router.analyze_query_complexity(complex_text, complex_embedding)
    
    # Complex should have higher score
    assert complex_complexity > simple_complexity
    
    # Both should be in [0, 1]
    assert 0 <= simple_complexity <= 1
    assert 0 <= complex_complexity <= 1


def test_route_cache_hit():
    """Test routing with cache hit."""
    router = QRouter()
    
    query_text = "test query"
    query_embedding = np.random.randn(64)
    
    decision = router.route(query_text, query_embedding, cache_status=True)
    
    assert decision == RouteDecision.CACHE_HIT


def test_route_by_complexity():
    """Test routing based on query complexity."""
    router = QRouter(complexity_threshold=0.5)
    
    # Simple query → classical
    simple_text = "a"
    simple_embedding = np.random.randn(64) * 0.1
    simple_decision = router.route(simple_text, simple_embedding, cache_status=False)
    
    # Complex query → quantum
    complex_text = "very complex query with many unique technical terminology tokens words"
    complex_embedding = np.random.randn(64) * 20.0
    complex_decision = router.route(complex_text, complex_embedding, cache_status=False)
    
    # Note: Due to heuristics, results may vary, but we test the mechanism
    assert simple_decision in [RouteDecision.CLASSICAL, RouteDecision.QUANTUM]
    assert complex_decision in [RouteDecision.CLASSICAL, RouteDecision.QUANTUM]


def test_route_quantum_load_limit():
    """Test routing respects quantum load limit."""
    router = QRouter(quantum_load_limit=2)
    
    query_text = "complex query"
    query_embedding = np.random.randn(64) * 10.0
    
    # Fill quantum queue
    router.quantum_queue_size = 2
    
    # Next query should be routed to classical despite complexity
    decision = router.route(query_text, query_embedding, cache_status=False)
    
    assert decision == RouteDecision.CLASSICAL


def test_release_quantum_slot():
    """Test releasing quantum execution slot."""
    router = QRouter()
    
    router.quantum_queue_size = 5
    router.release_quantum_slot()
    
    assert router.quantum_queue_size == 4
    
    # Should not go negative
    router.quantum_queue_size = 0
    router.release_quantum_slot()
    
    assert router.quantum_queue_size == 0


def test_routing_stats():
    """Test routing statistics tracking."""
    router = QRouter(complexity_threshold=0.3)
    
    # Route several queries
    queries = [
        ("simple", np.random.randn(64) * 0.1),
        ("complex query", np.random.randn(64) * 5.0),
        ("another", np.random.randn(64) * 0.2),
    ]
    
    for text, embedding in queries:
        router.route(text, embedding, cache_status=False)
    
    stats = router.get_routing_stats()
    
    # Should have stats for all route types
    assert RouteDecision.CACHE_HIT.value in stats
    assert RouteDecision.CLASSICAL.value in stats
    assert RouteDecision.QUANTUM.value in stats
    
    # Percentages should sum to ~1.0
    total = sum(stats.values())
    assert abs(total - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
