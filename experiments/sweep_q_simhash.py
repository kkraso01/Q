"""
Parameter sweep for Quantum Similarity Hash (Q-SimHash).
"""
import argparse
import csv
import itertools
import numpy as np
from pathlib import Path
from sim.q_simhash import QSimHash

def random_vec(length):
    np.random.seed(42)
    return bytes(np.random.randint(0, 2, length).astype(np.uint8))

def run_single_experiment(m, k, shots, n_trials=5):
    qsh = QSimHash(m=m, k=k)
    vec1 = random_vec(8)
    vec2 = random_vec(8)
    sim_same = []
    sim_diff = []
    for _ in range(n_trials):
        sim_same.append(qsh.similarity(vec1, vec1, shots=shots))
        sim_diff.append(qsh.similarity(vec1, vec2, shots=shots))
    return {
        'sim_same_mean': np.mean(sim_same),
        'sim_diff_mean': np.mean(sim_diff)
    }

def run_sweep(output_dir='results'):
    Path(output_dir).mkdir(exist_ok=True)
    m_values = [8, 16]
    k_values = [2, 4]
    shots_values = [128, 256]
    results = []
    for m, k, shots in itertools.product(m_values, k_values, shots_values):
        metrics = run_single_experiment(m, k, shots)
        results.append({
            'm': m, 'k': k, 'shots': shots,
            **metrics
        })
    output_file = Path(output_dir) / 'q_simhash_sweep.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()
    if args.sweep:
        run_sweep()
    else:
        print(run_single_experiment(8, 2, 128))

if __name__ == '__main__':
    main()
