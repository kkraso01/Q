"""
Quantum Router - Intelligent query routing for hybrid classical-quantum pipelines.

Routes queries to optimal execution path based on query characteristics:
    - Complexity: Simple substring → classical, complex semantic → quantum
    - Cache status: Cached → direct return, uncached → full pipeline
    - Resource availability: High load → classical fallback, low load → quantum
"""

import numpy as np
from typing import Dict, Optional
from enum import Enum


class RouteDecision(Enum):
    """Query routing decisions."""
    CACHE_HIT = "cache_hit"
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"


class QRouter:
    """
    Intelligent query router for Q-Retrieval system.
    
    Routing logic:
        1. Check cache: If hit → return cached result
        2. Analyze query complexity: Simple → classical, complex → quantum
        3. Check resource load: Overloaded → classical fallback
        4. Execute optimal path
    """
    
    def __init__(
        self,
        complexity_threshold: float = 0.5,
        quantum_load_limit: int = 100
    ):
        """
        Initialize router.
        
        Args:
            complexity_threshold: Threshold for quantum vs classical (0-1)
            quantum_load_limit: Max concurrent quantum queries
        """
        self.complexity_threshold = complexity_threshold
        self.quantum_load_limit = quantum_load_limit
        self.quantum_queue_size = 0
        
        # Routing statistics
        self.route_counts = {
            RouteDecision.CACHE_HIT: 0,
            RouteDecision.CLASSICAL: 0,
            RouteDecision.QUANTUM: 0,
            RouteDecision.HYBRID: 0
        }
    
    def analyze_query_complexity(self, query_text: str, query_embedding: np.ndarray) -> float:
        """
        Estimate query complexity score.
        
        Higher complexity → more likely to benefit from quantum processing.
        
        Args:
            query_text: Query string
            query_embedding: Query vector
        
        Returns:
            Complexity score in [0, 1]
        """
        # Heuristics for complexity:
        # 1. Text length
        text_complexity = min(len(query_text) / 100.0, 1.0)
        
        # 2. Embedding entropy (high entropy → more semantic information)
        embedding_norm = np.linalg.norm(query_embedding)
        embedding_complexity = min(embedding_norm / 10.0, 1.0)
        
        # 3. Number of unique tokens (approximate)
        tokens = query_text.split()
        token_complexity = min(len(set(tokens)) / 20.0, 1.0)
        
        # Weighted combination
        complexity = (
            0.3 * text_complexity +
            0.4 * embedding_complexity +
            0.3 * token_complexity
        )
        
        return complexity
    
    def route(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        cache_status: bool = False
    ) -> RouteDecision:
        """
        Determine optimal execution path for query.
        
        Args:
            query_text: Query string
            query_embedding: Query vector
            cache_status: Whether query is in cache
        
        Returns:
            Routing decision
        """
        # Priority 1: Cache hit
        if cache_status:
            self.route_counts[RouteDecision.CACHE_HIT] += 1
            return RouteDecision.CACHE_HIT
        
        # Priority 2: Quantum resource availability
        if self.quantum_queue_size >= self.quantum_load_limit:
            self.route_counts[RouteDecision.CLASSICAL] += 1
            return RouteDecision.CLASSICAL
        
        # Priority 3: Query complexity
        complexity = self.analyze_query_complexity(query_text, query_embedding)
        
        if complexity < self.complexity_threshold:
            # Simple query → classical
            self.route_counts[RouteDecision.CLASSICAL] += 1
            return RouteDecision.CLASSICAL
        else:
            # Complex query → quantum
            self.quantum_queue_size += 1
            self.route_counts[RouteDecision.QUANTUM] += 1
            return RouteDecision.QUANTUM
    
    def release_quantum_slot(self):
        """Release quantum execution slot after query completes."""
        self.quantum_queue_size = max(0, self.quantum_queue_size - 1)
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        total = sum(self.route_counts.values())
        if total == 0:
            return {route.value: 0.0 for route in RouteDecision}
        
        return {
            route.value: count / total
            for route, count in self.route_counts.items()
        }


def demo_router():
    """Demonstration of Q-Router."""
    print("=== Q-Router Demo ===\n")
    
    router = QRouter(complexity_threshold=0.5, quantum_load_limit=10)
    
    # Test queries with varying complexity
    queries = [
        ("simple query", np.random.randn(64) * 0.5),  # Low complexity
        ("complex semantic search with many concepts", np.random.randn(64) * 5.0),  # High complexity
        ("q", np.random.randn(64) * 0.1),  # Very low complexity
    ]
    
    for query_text, query_embedding in queries:
        complexity = router.analyze_query_complexity(query_text, query_embedding)
        decision = router.route(query_text, query_embedding, cache_status=False)
        
        print(f"Query: '{query_text}'")
        print(f"  Complexity: {complexity:.3f}")
        print(f"  Route: {decision.value}\n")
    
    # Show routing stats
    print("Routing statistics:")
    stats = router.get_routing_stats()
    for route, ratio in stats.items():
        print(f"  {route}: {ratio:.2%}")


if __name__ == "__main__":
    demo_router()
