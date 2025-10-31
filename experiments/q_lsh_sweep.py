# Q-LSH Parameter Sweep Experiments
# Evaluates Q-LSH on similarity search tasks with various parameters

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from sim.q_lsh import QLSH


def generate_synthetic_vectors(n, d, seed=42):
    """Generate synthetic vector dataset."""
    np.random.seed(seed)
    return [np.random.randn(d) for _ in range(n)]


def run_q_lsh_accuracy_sweep():
    """Sweep over memory sizes and measure recall@k."""
    print("Running Q-LSH accuracy vs memory sweep...")
    
    d = 64  # Vector dimension
    n_vectors = 50
    k_neighbors = 10
    
    vectors = generate_synthetic_vectors(n_vectors, d)
    query = np.random.randn(d)
    
    memory_sizes = [16, 32, 64, 128]
    results = []
    
    for m in memory_sizes:
        qlsh = QLSH(m=m, k=4, d=d)
        for v in vectors:
            qlsh.insert(v)
        
        neighbors = qlsh.query_knn(query, k_neighbors=k_neighbors, shots=512)
        recall = len(neighbors) / min(k_neighbors, n_vectors)
        
        results.append({
            'memory': m,
            'recall': recall
        })
        print(f"  m={m}: recall@{k_neighbors} = {recall:.3f}")
    
    return results


def run_q_lsh_shots_sweep():
    """Sweep over shot counts and measure accuracy."""
    print("Running Q-LSH accuracy vs shots sweep...")
    
    d = 64
    qlsh = QLSH(m=32, k=4, d=d)
    
    # Insert vectors
    for i in range(10):
        v = np.random.randn(d)
        qlsh.insert(v)
    
    v1 = np.random.randn(d)
    v2 = v1 + 0.1 * np.random.randn(d)  # Similar vector
    
    shot_counts = [128, 256, 512, 1024, 2048]
    results = []
    
    for shots in shot_counts:
        sim = qlsh.cosine_similarity_estimate(v1, v2, shots=shots)
        results.append({
            'shots': shots,
            'similarity': sim
        })
        print(f"  shots={shots}: similarity = {sim:.3f}")
    
    return results


def run_q_lsh_noise_sweep():
    """Sweep over noise levels and measure robustness."""
    print("Running Q-LSH noise robustness sweep...")
    
    d = 64
    qlsh = QLSH(m=32, k=4, d=d)
    
    v1 = np.random.randn(d)
    v2 = v1 + 0.1 * np.random.randn(d)
    
    noise_levels = [0.0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    results = []
    
    for noise in noise_levels:
        sim = qlsh.cosine_similarity_estimate(v1, v2, shots=1024, noise_level=noise)
        results.append({
            'noise': noise,
            'similarity': sim
        })
        print(f"  noise={noise:.4f}: similarity = {sim:.3f}")
    
    return results


if __name__ == "__main__":
    print("=== Q-LSH Parameter Sweeps ===\n")
    
    accuracy_results = run_q_lsh_accuracy_sweep()
    print()
    
    shots_results = run_q_lsh_shots_sweep()
    print()
    
    noise_results = run_q_lsh_noise_sweep()
    print()
    
    print("Q-LSH sweeps complete!")
