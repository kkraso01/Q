"""
Experimental sweep utilities for QAM performance evaluation.

Runs parameter grid searches and collects metrics for analysis.
"""
import argparse
import csv
import itertools
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))


from sim.qam import QAM, create_noise_model
from sim.classical_filters import CuckooFilter, XORFilter, VacuumFilter


def generate_random_items(n, length=8):
    """Generate n random byte strings."""
    np.random.seed(42)
    return [bytes(np.random.randint(0, 256, length)) for _ in range(n)]


def classical_bloom_filter(items, query_item, m, k):
    """
    Classical Bloom filter baseline.
    
    Returns false positive rate approximation.
    """
    from sim.utils import make_hash_functions
    
    # Build bit array
    bit_array = [False] * m
    hash_fns = make_hash_functions(k)
    
    # Insert items
    for item in items:
        item_hash = hash(item)
        for h in hash_fns:
            idx = h(item_hash) % m
            bit_array[idx] = True
    
    # Query
    query_hash = hash(query_item)
    for h in hash_fns:
        idx = h(query_hash) % m
        if not bit_array[idx]:
            return False  # Definitely not in set
    
    return True  # Probably in set


def run_single_experiment(m, k, set_size, shots, noise_rate, theta=np.pi/4, n_trials=10):
    """
    Run single parameter configuration experiment.
    
    Returns metrics averaged over n_trials.
    """
    np.random.seed(42)
    
    qam = QAM(m=m, k=k, theta=theta)
    noise_model = create_noise_model(noise_rate) if noise_rate > 0 else None
    
    # Generate dataset
    all_items = generate_random_items(set_size * 3, length=8)
    

    qam_fp_rates = []
    qam_fn_rates = []
    classical_fp_rates = []
    classical_fn_rates = []
    cuckoo_fp_rates = []
    cuckoo_fn_rates = []
    xor_fp_rates = []
    xor_fn_rates = []
    vacuum_fp_rates = []
    vacuum_fn_rates = []
    

    for trial in range(n_trials):
        trial_seed = 42 + trial
        np.random.seed(trial_seed)

        # Split into inserted and test sets
        inserted = all_items[:set_size]
        test_positive = all_items[set_size:set_size + 10]  # In set
        test_negative = all_items[set_size + 10:set_size + 20]  # Not in set

        # QAM metrics
        qam_fps = 0
        qam_fns = 0
        for item in test_negative:
            exp = qam.query(inserted, item, shots=shots, noise_model=noise_model)
            if exp >= 0.5:  # Threshold
                qam_fps += 1

        for item in test_positive:
            exp = qam.query(inserted, item, shots=shots, noise_model=noise_model)
            if exp < 0.5:
                qam_fns += 1

        qam_fp_rates.append(qam_fps / len(test_negative))
        qam_fn_rates.append(qam_fns / len(test_positive))

        # Classical Bloom filter baseline
        classical_fps = 0
        classical_fns = 0
        for item in test_negative:
            if classical_bloom_filter(inserted, item, m, k):
                classical_fps += 1
        for item in test_positive:
            if not classical_bloom_filter(inserted, item, m, k):
                classical_fns += 1
        classical_fp_rates.append(classical_fps / len(test_negative))
        classical_fn_rates.append(classical_fns / len(test_positive))

        # Cuckoo filter baseline
        cuckoo = CuckooFilter(m)
        for x in inserted:
            cuckoo.insert(x)
        cuckoo_fps = sum(cuckoo.contains(x) for x in test_negative)
        cuckoo_fns = sum(not cuckoo.contains(x) for x in test_positive)
        cuckoo_fp_rates.append(cuckoo_fps / len(test_negative))
        cuckoo_fn_rates.append(cuckoo_fns / len(test_positive))

        # XOR filter baseline
        xor = XORFilter(m, k)
        for x in inserted:
            xor.insert(x)
        xor_fps = sum(xor.contains(x) for x in test_negative)
        xor_fns = sum(not xor.contains(x) for x in test_positive)
        xor_fp_rates.append(xor_fps / len(test_negative))
        xor_fn_rates.append(xor_fns / len(test_positive))

        # Vacuum filter baseline
        vacuum = VacuumFilter(m, k)
        for x in inserted:
            vacuum.insert(x)
        vacuum_fps = sum(vacuum.contains(x) for x in test_negative)
        vacuum_fns = sum(not vacuum.contains(x) for x in test_positive)
        vacuum_fp_rates.append(vacuum_fps / len(test_negative))
        vacuum_fn_rates.append(vacuum_fns / len(test_positive))
    
    return {
        'qam_fp_mean': np.mean(qam_fp_rates),
        'qam_fp_std': np.std(qam_fp_rates),
        'qam_fn_mean': np.mean(qam_fn_rates),
        'qam_fn_std': np.std(qam_fn_rates),
        'classical_fp_mean': np.mean(classical_fp_rates),
        'classical_fp_std': np.std(classical_fp_rates),
        'classical_fn_mean': np.mean(classical_fn_rates),
        'classical_fn_std': np.std(classical_fn_rates),
        'cuckoo_fp_mean': np.mean(cuckoo_fp_rates),
        'cuckoo_fp_std': np.std(cuckoo_fp_rates),
        'cuckoo_fn_mean': np.mean(cuckoo_fn_rates),
        'cuckoo_fn_std': np.std(cuckoo_fn_rates),
        'xor_fp_mean': np.mean(xor_fp_rates),
        'xor_fp_std': np.std(xor_fp_rates),
        'xor_fn_mean': np.mean(xor_fn_rates),
        'xor_fn_std': np.std(xor_fn_rates),
        'vacuum_fp_mean': np.mean(vacuum_fp_rates),
        'vacuum_fp_std': np.std(vacuum_fp_rates),
        'vacuum_fn_mean': np.mean(vacuum_fn_rates),
        'vacuum_fn_std': np.std(vacuum_fn_rates),
        'load_factor': set_size / m
    }


def run_parameter_sweep(output_dir='results'):
    """Run full parameter grid search."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Parameter ranges
    m_values = [16, 32, 64]
    k_values = [2, 3, 4]
    set_sizes = [32, 64, 128]
    shot_values = [128, 256, 512, 1024]
    noise_values = [0.0, 0.001, 0.01]
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(output_dir) / f'qam_sweep_{timestamp}.csv'
    
    results = []
    total = len(m_values) * len(k_values) * len(set_sizes) * len(shot_values) * len(noise_values)
    count = 0
    
    print(f"Running {total} experiments...")
    
    for m, k, set_size, shots, noise in itertools.product(
        m_values, k_values, set_sizes, shot_values, noise_values
    ):
        count += 1
        print(f"[{count}/{total}] m={m}, k={k}, |S|={set_size}, shots={shots}, ε={noise}")
        
        try:
            metrics = run_single_experiment(m, k, set_size, shots, noise)
            
            results.append({
                'm': m,
                'k': k,
                'set_size': set_size,
                'shots': shots,
                'noise': noise,
                **metrics
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
    
    # Save results
    if results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to {output_file}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Run QAM parameter sweeps')
    parser.add_argument('--m', type=int, default=32, help='Number of qubits')
    parser.add_argument('--k', type=int, default=3, help='Number of hash functions')
    parser.add_argument('--set-size', type=int, default=64, help='Set size')
    parser.add_argument('--shots', type=int, default=512, help='Measurement shots')
    parser.add_argument('--noise', type=float, default=0.0, help='Noise rate')
    parser.add_argument('--sweep', action='store_true', help='Run full parameter sweep')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    if args.sweep:
        run_parameter_sweep(args.output_dir)
    else:
        print(f"Running single experiment: m={args.m}, k={args.k}, |S|={args.set_size}, shots={args.shots}, ε={args.noise}")
        metrics = run_single_experiment(args.m, args.k, args.set_size, args.shots, args.noise)
        
        print("\nResults:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()
