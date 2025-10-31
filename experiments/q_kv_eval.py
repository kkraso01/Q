# Q-KV Cache Policy Evaluation Experiments
# Compares Q-KV against LRU and LFU baselines

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from systems.q_kv_policy import QKVPolicy, LRUPolicy, LFUPolicy


def generate_workload(n_keys, n_accesses, zipf_param=1.5, seed=42):
    """Generate realistic cache workload with Zipf distribution."""
    np.random.seed(seed)
    
    # Zipf distribution for key popularity
    ranks = np.arange(1, n_keys + 1)
    probabilities = 1 / (ranks ** zipf_param)
    probabilities /= probabilities.sum()
    
    # Generate access sequence
    keys = [f"key{i}".encode() for i in range(n_keys)]
    accesses = np.random.choice(keys, size=n_accesses, p=probabilities)
    
    return list(accesses)


def run_cache_size_sweep():
    """Sweep over cache sizes and measure hit rates."""
    print("Running cache size vs hit rate sweep...")
    
    n_keys = 100
    n_accesses = 1000
    workload = generate_workload(n_keys, n_accesses)
    
    cache_sizes = [10, 20, 50, 100]
    results = {'qkv': [], 'lru': [], 'lfu': []}
    
    for cache_size in cache_sizes:
        # Q-KV policy
        qkv = QKVPolicy(m=32, k=3, cache_size=cache_size)
        for key in workload:
            val = qkv.get(key)
            if val is None:
                qkv.put(key, f"value_{key.decode()}")
        qkv_hit_rate = qkv.get_hit_rate()
        results['qkv'].append(qkv_hit_rate)
        
        # LRU policy
        lru = LRUPolicy(cache_size=cache_size)
        for key in workload:
            val = lru.get(key)
            if val is None:
                lru.put(key, f"value_{key.decode()}")
        lru_hit_rate = lru.get_hit_rate()
        results['lru'].append(lru_hit_rate)
        
        # LFU policy
        lfu = LFUPolicy(cache_size=cache_size)
        for key in workload:
            val = lfu.get(key)
            if val is None:
                lfu.put(key, f"value_{key.decode()}")
        lfu_hit_rate = lfu.get_hit_rate()
        results['lfu'].append(lfu_hit_rate)
        
        print(f"  cache_size={cache_size}: Q-KV={qkv_hit_rate:.3f}, LRU={lru_hit_rate:.3f}, LFU={lfu_hit_rate:.3f}")
    
    return cache_sizes, results


def run_zipf_param_sweep():
    """Sweep over workload skewness (Zipf parameter)."""
    print("Running Zipf parameter vs hit rate sweep...")
    
    cache_size = 50
    n_keys = 100
    n_accesses = 1000
    
    zipf_params = [1.0, 1.5, 2.0, 2.5]
    results = {'qkv': [], 'lru': [], 'lfu': []}
    
    for zipf in zipf_params:
        workload = generate_workload(n_keys, n_accesses, zipf_param=zipf)
        
        # Q-KV
        qkv = QKVPolicy(m=32, cache_size=cache_size)
        for key in workload:
            if qkv.get(key) is None:
                qkv.put(key, f"value_{key.decode()}")
        results['qkv'].append(qkv.get_hit_rate())
        
        # LRU
        lru = LRUPolicy(cache_size=cache_size)
        for key in workload:
            if lru.get(key) is None:
                lru.put(key, f"value_{key.decode()}")
        results['lru'].append(lru.get_hit_rate())
        
        # LFU
        lfu = LFUPolicy(cache_size=cache_size)
        for key in workload:
            if lfu.get(key) is None:
                lfu.put(key, f"value_{key.decode()}")
        results['lfu'].append(lfu.get_hit_rate())
        
        print(f"  zipf={zipf}: Q-KV={results['qkv'][-1]:.3f}, LRU={results['lru'][-1]:.3f}, LFU={results['lfu'][-1]:.3f}")
    
    return zipf_params, results


if __name__ == "__main__":
    print("=== Q-KV Cache Policy Evaluation ===\n")
    
    cache_sizes, size_results = run_cache_size_sweep()
    print()
    
    zipf_params, zipf_results = run_zipf_param_sweep()
    print()
    
    print("Q-KV evaluation complete!")
