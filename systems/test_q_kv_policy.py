"""Unit tests for Quantum KV-Cache Policy"""
import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from systems.q_kv_policy import QKVPolicy, LRUPolicy, LFUPolicy


def test_qkv_initialization():
    """Test Q-KV policy initialization."""
    policy = QKVPolicy(m=16, k=3, cache_size=10)
    assert policy.m == 16
    assert policy.cache_size == 10
    assert len(policy.cache) == 0


def test_qkv_put_get():
    """Test basic put and get operations."""
    policy = QKVPolicy(m=16, cache_size=10)
    policy.put(b"key1", "value1")
    assert policy.get(b"key1") == "value1"


def test_qkv_cache_miss():
    """Test cache miss."""
    policy = QKVPolicy(m=16, cache_size=10)
    assert policy.get(b"nonexistent") is None


def test_qkv_eviction():
    """Test that eviction occurs when cache is full."""
    policy = QKVPolicy(m=16, cache_size=3)
    policy.put(b"k1", "v1")
    policy.put(b"k2", "v2")
    policy.put(b"k3", "v3")
    
    initial_len = len(policy.cache)
    policy.put(b"k4", "v4")
    
    # Cache should still have size 3 (one evicted)
    assert len(policy.cache) == 3


def test_qkv_importance_estimation():
    """Test quantum importance estimation."""
    policy = QKVPolicy(m=32, k=3, cache_size=10)
    policy.put(b"key1", "value1")
    
    importance = policy.estimate_importance(b"key1", shots=256)
    assert 0.0 <= importance <= 1.0


def test_qkv_access_tracking():
    """Test that access counts are tracked."""
    policy = QKVPolicy(m=16, cache_size=10)
    policy.put(b"key1", "value1")
    
    # Access multiple times
    for _ in range(5):
        policy.get(b"key1")
    
    _, access_count = policy.cache[b"key1"]
    assert access_count == 6  # 1 put + 5 gets


def test_qkv_hit_rate():
    """Test hit rate calculation."""
    policy = QKVPolicy(m=16, cache_size=5)
    policy.put(b"k1", "v1")
    policy.put(b"k2", "v2")
    
    # Access cached keys
    policy.get(b"k1")
    policy.get(b"k2")
    # Access non-cached key
    policy.get(b"k3")
    
    hit_rate = policy.get_hit_rate()
    assert 0.0 <= hit_rate <= 1.0


def test_lru_baseline():
    """Test LRU baseline policy."""
    lru = LRUPolicy(cache_size=3)
    lru.put(b"k1", "v1")
    lru.put(b"k2", "v2")
    lru.put(b"k3", "v3")
    
    # Access k1 to make it recently used
    lru.get(b"k1")
    
    # Add k4, should evict k2 (least recently used)
    lru.put(b"k4", "v4")
    assert len(lru.cache) == 3
    assert b"k2" not in lru.cache


def test_lfu_baseline():
    """Test LFU baseline policy."""
    lfu = LFUPolicy(cache_size=3)
    lfu.put(b"k1", "v1")
    lfu.put(b"k2", "v2")
    lfu.put(b"k3", "v3")
    
    # Access k1 multiple times
    for _ in range(5):
        lfu.get(b"k1")
    
    # Add k4, should evict k2 or k3 (least frequently used)
    lfu.put(b"k4", "v4")
    assert len(lfu.cache) == 3
    assert b"k1" in lfu.cache  # Most frequent should remain


def test_policy_comparison():
    """Compare Q-KV, LRU, and LFU policies."""
    cache_size = 5
    
    qkv = QKVPolicy(m=16, cache_size=cache_size)
    lru = LRUPolicy(cache_size=cache_size)
    lfu = LFUPolicy(cache_size=cache_size)
    
    # Simulate workload
    keys = [f"key{i % 8}".encode() for i in range(20)]
    
    for key in keys:
        value = f"value_{key.decode()}"
        qkv.put(key, value)
        lru.put(key, value)
        lfu.put(key, value)
    
    # All should have same cache size
    assert len(qkv.cache) <= cache_size
    assert len(lru.cache) <= cache_size
    assert len(lfu.cache) <= cache_size


def test_qkv_clear():
    """Test cache clearing."""
    policy = QKVPolicy(m=16, cache_size=10)
    policy.put(b"k1", "v1")
    policy.put(b"k2", "v2")
    
    policy.clear()
    assert len(policy.cache) == 0
    assert len(policy._key_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
