"""
Parameter sweep for Quantum Suffix Sketch (Q-SubSketch).
"""
import argparse
import csv
import itertools
import numpy as np
from pathlib import Path
from sim.q_subsketch import QSubSketch

def generate_text(length):
    np.random.seed(42)
    return bytes(np.random.randint(97, 123, length))  # lowercase letters

def run_single_experiment(m, k, L, stride, shots, noise, n_trials=5):
    qss = QSubSketch(m=m, k=k, L=L, stride=stride)
    text = generate_text(64)
    pattern_in = text[10:10+L]
    pattern_out = b"z" * L
    fp_rates = []
    fn_rates = []
    for _ in range(n_trials):
        exp_in = qss.query(text, pattern_in, shots=shots)
        exp_out = qss.query(text, pattern_out, shots=shots)
        fp = 1 if exp_out >= 0.5 else 0
        fn = 1 if exp_in < 0.5 else 0
        fp_rates.append(fp)
        fn_rates.append(fn)
    return {
        'fp_mean': np.mean(fp_rates),
        'fn_mean': np.mean(fn_rates)
    }

def run_sweep(output_dir='results'):
    Path(output_dir).mkdir(exist_ok=True)
    m_values = [8, 16]
    k_values = [2, 3]
    L_values = [4, 8]
    stride_values = [1]
    shots_values = [128, 256]
    noise_values = [0.0]
    results = []
    for m, k, L, stride, shots, noise in itertools.product(m_values, k_values, L_values, stride_values, shots_values, noise_values):
        metrics = run_single_experiment(m, k, L, stride, shots, noise)
        results.append({
            'm': m, 'k': k, 'L': L, 'stride': stride, 'shots': shots, 'noise': noise,
            **metrics
        })
    output_file = Path(output_dir) / 'q_subsketch_sweep.csv'
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
        print(run_single_experiment(8, 2, 4, 1, 128, 0.0))

if __name__ == '__main__':
    main()
