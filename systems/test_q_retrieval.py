"""
Unit tests for Q-Retrieval integrated system.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from systems.q_retrieval import QRetrieval


def test_q_retrieval_init():
    """Test Q-Retrieval initialization."""
    retriever = QRetrieval(m=32, d=64, substring_length=4)
    
    assert retriever.m == 32
    assert retriever.d == 64
    assert retriever.substring_length == 4
    assert len(retriever.documents) == 0


def test_q_retrieval_index():
    """Test document indexing."""
    retriever = QRetrieval(m=16, d=32, substring_length=4)
    
    documents = [
        {'id': 'doc0', 'text': 'quantum computing', 'embedding': np.random.randn(32)},
        {'id': 'doc1', 'text': 'classical algorithms', 'embedding': np.random.randn(32)}
    ]
    
    retriever.index(documents)
    
    assert len(retriever.documents) == 2
    assert len(retriever.doc_embeddings) == 2


def test_q_retrieval_query():
    """Test query execution."""
    retriever = QRetrieval(m=16, d=32, substring_length=4, cache_size=5)
    
    documents = [
        {'id': 'doc0', 'text': 'quantum data structures', 'embedding': np.random.randn(32)},
        {'id': 'doc1', 'text': 'classical data structures', 'embedding': np.random.randn(32)},
        {'id': 'doc2', 'text': 'machine learning algorithms', 'embedding': np.random.randn(32)}
    ]
    
    retriever.index(documents)
    
    query_text = "quantum algorithms"
    query_embedding = np.random.randn(32)
    
    results = retriever.query(query_text, query_embedding, top_k=2, use_cache=False)
    
    # Should return up to 2 results
    assert len(results) <= 2
    
    # Each result is (document, score)
    for doc, score in results:
        assert 'id' in doc
        assert 'text' in doc
        assert isinstance(score, (float, np.floating))


def test_q_retrieval_caching():
    """Test query result caching."""
    retriever = QRetrieval(m=16, d=32, cache_size=5)
    
    documents = [
        {'id': 'doc0', 'text': 'test document', 'embedding': np.random.randn(32)}
    ]
    retriever.index(documents)
    
    query_text = "test query"
    query_embedding = np.random.randn(32)
    
    # First query (cache miss)
    results1 = retriever.query(query_text, query_embedding, top_k=1, use_cache=True)
    
    # Second query (should hit cache)
    results2 = retriever.query(query_text, query_embedding, top_k=1, use_cache=True)
    
    # Results should be identical (from cache)
    assert len(results1) == len(results2)


def test_q_retrieval_stats():
    """Test retrieval statistics."""
    retriever = QRetrieval(m=16, d=32)
    
    documents = [
        {'id': f'doc{i}', 'text': f'document {i}', 'embedding': np.random.randn(32)}
        for i in range(5)
    ]
    retriever.index(documents)
    
    stats = retriever.get_stats()
    
    assert stats['n_documents'] == 5
    assert 'cache_hit_rate' in stats
    assert 'cache_size' in stats


def test_q_retrieval_empty_query():
    """Test query with no matching candidates."""
    retriever = QRetrieval(m=16, d=32, substring_length=4)
    
    documents = [
        {'id': 'doc0', 'text': 'quantum', 'embedding': np.random.randn(32)}
    ]
    retriever.index(documents)
    
    # Query with no substring matches
    query_text = "xyz"  # Very short, unlikely to match
    query_embedding = np.random.randn(32)
    
    results = retriever.query(query_text, query_embedding, top_k=1, use_cache=False)
    
    # Should handle gracefully (empty or filtered results)
    assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
