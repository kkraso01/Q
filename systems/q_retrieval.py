"""
Quantum Retrieval Stack - Integrated Q-SubSketch → Q-LSH → Q-HH → Q-KV Pipeline

This module provides a unified retrieval system combining multiple quantum data structures
for end-to-end similarity search, ranking, and caching.

Architecture:
    1. Q-SubSketch: Fast substring filtering for candidate generation
    2. Q-LSH: Similarity-based retrieval from candidate set
    3. Q-HH: Frequency-based ranking and relevance scoring
    4. Q-KV: Intelligent caching of retrieved results

Usage:
    retriever = QRetrieval(m=128, d=256)
    retriever.index(documents)
    results = retriever.query(query_text, top_k=10)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sim.q_subsketch import QSubSketch
from sim.q_lsh import QLSH
from sim.q_hh import QHH
from systems.q_kv_policy import QKVPolicy


class QRetrieval:
    """
    Integrated quantum retrieval stack.
    
    Pipeline stages:
        1. Substring filter (Q-SubSketch): Prune documents without query substrings
        2. Vector similarity (Q-LSH): Rank by embedding similarity
        3. Frequency boost (Q-HH): Re-rank by popularity/relevance signals
        4. Result cache (Q-KV): Cache frequent query-result pairs
    """
    
    def __init__(
        self,
        m: int = 128,
        k: int = 4,
        d: int = 256,
        substring_length: int = 8,
        cache_size: int = 100,
        shots: int = 512,
        noise_level: float = 0.0
    ):
        """
        Initialize quantum retrieval stack.
        
        Args:
            m: Memory size (qubits) for each component
            k: Number of hash functions
            d: Vector embedding dimension
            substring_length: Character ngram length for Q-SubSketch
            cache_size: Maximum cached query-result pairs
            shots: Measurement shots per query
            noise_level: Depolarizing noise parameter
        """
        self.m = m
        self.k = k
        self.d = d
        self.substring_length = substring_length
        self.shots = shots
        self.noise_level = noise_level
        
        # Initialize components
        self.subsketch = QSubSketch(m=m // 2, k=k)
        self.lsh = QLSH(m=m, k=k, d=d)
        self.hh = QHH(m=m // 2, k=k)
        self.cache = QKVPolicy(m=m // 4, k=k, cache_size=cache_size)
        
        # Document storage
        self.documents: List[Dict] = []
        self.doc_embeddings: List[np.ndarray] = []
    
    def index(self, documents: List[Dict]):
        """
        Index documents into retrieval system.
        
        Args:
            documents: List of dicts with keys:
                - 'text': Document text
                - 'embedding': Vector embedding (d-dimensional)
                - 'id': Document identifier
        """
        print(f"Indexing {len(documents)} documents...")
        
        for doc in documents:
            # Extract substrings for Q-SubSketch
            text = doc['text']
            for i in range(len(text) - self.substring_length + 1):
                substring = text[i:i+self.substring_length]
                self.subsketch.insert(substring.encode())
            
            # Insert embedding into Q-LSH
            embedding = doc['embedding']
            self.lsh.insert(embedding)
            
            # Track document access for Q-HH
            doc_id = doc['id'].encode()
            self.hh.insert(doc_id)
            
            # Store document
            self.documents.append(doc)
            self.doc_embeddings.append(embedding)
        
        print(f"  Indexed {len(self.documents)} documents")
    
    def query(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        use_cache: bool = True
    ) -> List[Tuple[Dict, float]]:
        """
        Query retrieval system with text and embedding.
        
        Pipeline:
            1. Check cache for query_text
            2. Q-SubSketch: Filter documents by substring presence
            3. Q-LSH: Rank candidates by embedding similarity
            4. Q-HH: Boost popular/relevant documents
            5. Cache results
        
        Args:
            query_text: Query string
            query_embedding: Query vector embedding
            top_k: Number of results to return
            use_cache: Whether to use Q-KV cache
        
        Returns:
            List of (document, score) tuples, ranked by relevance
        """
        query_key = query_text.encode()
        
        # Stage 0: Check cache
        if use_cache:
            cached_result = self.cache.get(query_key)
            if cached_result is not None:
                print("  Cache hit!")
                return cached_result
        
        # Stage 1: Substring filtering with Q-SubSketch
        print("  Stage 1: Substring filtering...")
        candidates = []
        for idx, doc in enumerate(self.documents):
            # Check if query substrings are in document
            match = False
            for i in range(len(query_text) - self.substring_length + 1):
                substring = query_text[i:i+self.substring_length]
                if self.subsketch.query(substring.encode(), shots=self.shots):
                    match = True
                    break
            if match:
                candidates.append(idx)
        
        print(f"    Candidates: {len(candidates)}/{len(self.documents)}")
        
        if len(candidates) == 0:
            print("    No candidates found!")
            return []
        
        # Stage 2: Similarity ranking with Q-LSH
        print("  Stage 2: Similarity ranking...")
        similarities = []
        for idx in candidates:
            doc_embedding = self.doc_embeddings[idx]
            sim = self.lsh.cosine_similarity_estimate(
                query_embedding,
                doc_embedding,
                shots=self.shots,
                noise_level=self.noise_level
            )
            similarities.append((idx, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_candidates = similarities[:min(top_k * 2, len(similarities))]
        
        # Stage 3: Frequency boosting with Q-HH
        print("  Stage 3: Frequency boosting...")
        final_scores = []
        for idx, sim in top_candidates:
            doc_id = self.documents[idx]['id'].encode()
            freq_est = self.hh.estimate_frequency(doc_id, shots=self.shots)
            
            # Combined score: similarity + frequency boost
            boost = np.log1p(freq_est) * 0.1  # Small boost for popular docs
            final_score = sim + boost
            
            final_scores.append((idx, final_score))
        
        # Sort by final score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        results = [(self.documents[idx], score) for idx, score in final_scores[:top_k]]
        
        # Stage 4: Cache results
        if use_cache:
            self.cache.put(query_key, results)
        
        print(f"  Retrieved {len(results)} results")
        return results
    
    def get_stats(self) -> Dict:
        """Get retrieval system statistics."""
        return {
            'n_documents': len(self.documents),
            'cache_hit_rate': self.cache.get_hit_rate(),
            'cache_size': len(self.cache.cache),
            'subsketch_qubits': self.subsketch.m,
            'lsh_qubits': self.lsh.m,
            'hh_qubits': self.hh.m,
            'cache_qubits': self.cache.m
        }


def demo_retrieval_pipeline():
    """Demonstration of Q-Retrieval pipeline."""
    print("=== Q-Retrieval Pipeline Demo ===\n")
    
    # Create synthetic documents
    documents = [
        {
            'id': 'doc0',
            'text': 'quantum computing algorithms',
            'embedding': np.random.randn(64)
        },
        {
            'id': 'doc1',
            'text': 'quantum data structures',
            'embedding': np.random.randn(64)
        },
        {
            'id': 'doc2',
            'text': 'classical algorithms and complexity',
            'embedding': np.random.randn(64)
        }
    ]
    
    # Initialize retrieval system
    retriever = QRetrieval(m=32, d=64, substring_length=4, cache_size=10)
    
    # Index documents
    retriever.index(documents)
    
    # Query
    query_text = "quantum algorithms"
    query_embedding = np.random.randn(64)
    
    results = retriever.query(query_text, query_embedding, top_k=2)
    
    print("\nResults:")
    for i, (doc, score) in enumerate(results):
        print(f"  {i+1}. {doc['id']}: {doc['text'][:50]}... (score: {score:.3f})")
    
    # Show stats
    print("\nSystem stats:")
    stats = retriever.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_retrieval_pipeline()
