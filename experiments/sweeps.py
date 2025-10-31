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


def run_batch_query_experiment(m, k, set_size, shots, batch_sizes=[1, 16, 64], noise_rate=0.0, theta=np.pi/4, n_trials=10, output_dir='results'):
    """
    Run batch query experiments for QAM.
    For each batch size, run n_trials and record error and amortized cost.
    Saves results as CSV in output_dir.
    """
    np.random.seed(42)
    from sim.qam import QAM, create_noise_model
    Path(output_dir).mkdir(exist_ok=True)
    all_items = generate_random_items(set_size * 3, length=8)
    noise_model = create_noise_model(noise_rate) if noise_rate > 0 else None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(output_dir) / f'qam_batch_query_{timestamp}.csv'
    results = []
    for batch in batch_sizes:
        for trial in range(n_trials):
            trial_seed = 42 + trial
            np.random.seed(trial_seed)
            inserted = all_items[:set_size]
            # For each batch, sample batch query items (half in set, half not)
            n_in = batch // 2
            n_out = batch - n_in
            in_items = all_items[set_size:set_size + n_in]
            out_items = all_items[set_size + 10:set_size + 10 + n_out]
            query_items = list(in_items) + list(out_items)
            qam = QAM(m=m, k=k, theta=theta)
            batch_results = qam.batch_query(inserted, query_items, shots=shots, noise_model=noise_model)
            # Compute error rates
            fp = sum(1 for i, (item, exp, is_member) in enumerate(batch_results[n_in:]) if is_member)
            fn = sum(1 for i, (item, exp, is_member) in enumerate(batch_results[:n_in]) if not is_member)
            fp_rate = fp / n_out if n_out > 0 else 0.0
            fn_rate = fn / n_in if n_in > 0 else 0.0
            # Amortized cost: total shots / batch size
            amortized_shots = shots / batch
            results.append({
                'm': m,
                'k': k,
                'set_size': set_size,
                'shots': shots,
                'batch_size': batch,
                'noise': noise_rate,
                'trial': trial,
                'fp_rate': fp_rate,
                'fn_rate': fn_rate,
                'amortized_shots': amortized_shots
            })
    # Save results
    if results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Batch query results saved to {output_file}")
    return output_file


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


def run_heatmap_sweep(m=32, k=3, set_size=64, shot_values=[128, 256, 512, 1024], noise_values=[0.0, 0.001, 0.005, 0.01, 0.02], theta=np.pi/4, n_trials=10, output_dir='results'):
    """
    Run shots × noise heatmap sweep for QAM.
    Sweeps over shots and noise, records error rates, saves as CSV.
    """
    np.random.seed(42)
    from sim.qam import QAM, create_noise_model
    Path(output_dir).mkdir(exist_ok=True)
    all_items = generate_random_items(set_size * 3, length=8)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(output_dir) / f'qam_heatmap_{timestamp}.csv'
    results = []
    total = len(shot_values) * len(noise_values)
    count = 0
    print(f"Running {total} heatmap experiments...")
    for shots in shot_values:
        for noise in noise_values:
            count += 1
            print(f"[{count}/{total}] shots={shots}, ε={noise}")
            fp_rates = []
            fn_rates = []
            for trial in range(n_trials):
                trial_seed = 42 + trial
                np.random.seed(trial_seed)
                inserted = all_items[:set_size]
                test_positive = all_items[set_size:set_size + 10]
                test_negative = all_items[set_size + 10:set_size + 20]
                qam = QAM(m=m, k=k, theta=theta)
                noise_model = create_noise_model(noise) if noise > 0 else None
                # Compute FP
                fps = 0
                for item in test_negative:
                    exp = qam.query(inserted, item, shots=shots, noise_model=noise_model)
                    if exp >= 0.5:
                        fps += 1
                fp_rate = fps / len(test_negative)
                # Compute FN
                fns = 0
                for item in test_positive:
                    exp = qam.query(inserted, item, shots=shots, noise_model=noise_model)
                    if exp < 0.5:
                        fns += 1
                fn_rate = fns / len(test_positive)
                fp_rates.append(fp_rate)
                fn_rates.append(fn_rate)
            results.append({
                'm': m,
                'k': k,
                'set_size': set_size,
                'shots': shots,
                'noise': noise,
                'fp_mean': np.mean(fp_rates),
                'fp_std': np.std(fp_rates),
                'fn_mean': np.mean(fn_rates),
                'fn_std': np.std(fn_rates)
            })
    # Save results
    if results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Heatmap results saved to {output_file}")
    return output_file


def run_q_subsketch_evaluation(text_file=None, m=32, k=3, L_values=[4, 8, 16, 32], shots=512, n_trials=5, output_dir='results'):
    """
    Evaluate Q-SubSketch on real text corpus.
    If no text_file provided, generates synthetic text.
    Computes AUC vs substring length and saves results.
    """
    np.random.seed(42)
    from sim.q_subsketch import QSubSketch
    from sklearn.metrics import roc_auc_score
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load or generate text
    if text_file and Path(text_file).exists():
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            corpus = f.read().encode('utf-8')
        print(f"Loaded text corpus: {len(corpus)} bytes")
    else:
        # Generate synthetic text (code-like corpus)
        print("Generating synthetic code-like corpus...")
        corpus = b"def main():\n    for i in range(100):\n        print(i)\n" * 100
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(output_dir) / f'q_subsketch_{timestamp}.csv'
    results = []
    total = len(L_values)
    count = 0
    print(f"Running {total} Q-SubSketch experiments...")
    
    for L in L_values:
        count += 1
        print(f"[{count}/{total}] L={L}")
        auc_scores = []
        for trial in range(n_trials):
            trial_seed = 42 + trial
            np.random.seed(trial_seed)
            # Extract a chunk of text
            chunk_start = np.random.randint(0, max(1, len(corpus) - 1000))
            chunk_end = min(chunk_start + 1000, len(corpus))
            text_chunk = corpus[chunk_start:chunk_end]
            
            # Ensure chunk is long enough
            if len(text_chunk) < L + 50:
                continue
            
            # Sample positive (in text) and negative (not in text) substrings
            n_test = 20
            positive_patterns = []
            for _ in range(n_test // 2):
                start = np.random.randint(0, len(text_chunk) - L)
                positive_patterns.append(text_chunk[start:start+L])
            
            negative_patterns = []
            for _ in range(n_test // 2):
                # Random bytes not in text
                neg = bytes(np.random.randint(0, 256, L))
                negative_patterns.append(neg)
            
            # Query with Q-SubSketch
            qss = QSubSketch(m=m, k=k, L=L, stride=1)
            y_true = [1] * len(positive_patterns) + [0] * len(negative_patterns)
            y_scores = []
            for pat in positive_patterns:
                exp = qss.query(text_chunk, pat, shots=shots)
                y_scores.append(exp)
            for pat in negative_patterns:
                exp = qss.query(text_chunk, pat, shots=shots)
                y_scores.append(exp)
            
            # Compute AUC
            try:
                auc = roc_auc_score(y_true, y_scores)
                auc_scores.append(auc)
            except:
                pass
        
        if auc_scores:
            results.append({
                'm': m,
                'k': k,
                'L': L,
                'shots': shots,
                'auc_mean': np.mean(auc_scores),
                'auc_std': np.std(auc_scores)
            })
    
    # Save results
    if results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Q-SubSketch results saved to {output_file}")
    return output_file


def run_topology_sweep(m=32, k=3, set_size=64, shots=512, topologies=['none', 'linear', 'ring', 'all-to-all'], noise_rate=0.0, theta=np.pi/4, n_trials=10, output_dir='results'):
    """
    Run topology sweep for QAM.
    Compare different entanglement topologies: none, linear, ring, all-to-all.
    Records error rates and circuit depth for each topology.
    """
    np.random.seed(42)
    from sim.qam import QAM, create_noise_model
    Path(output_dir).mkdir(exist_ok=True)
    all_items = generate_random_items(set_size * 3, length=8)
    noise_model = create_noise_model(noise_rate) if noise_rate > 0 else None
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = Path(output_dir) / f'qam_topology_{timestamp}.csv'
    results = []
    total = len(topologies)
    count = 0
    print(f"Running {total} topology experiments...")
    for topo in topologies:
        count += 1
        print(f"[{count}/{total}] topology={topo}")
        fp_rates = []
        fn_rates = []
        depths = []
        for trial in range(n_trials):
            trial_seed = 42 + trial
            np.random.seed(trial_seed)
            inserted = all_items[:set_size]
            test_positive = all_items[set_size:set_size + 10]
            test_negative = all_items[set_size + 10:set_size + 20]
            qam = QAM(m=m, k=k, theta=theta, topology=topo)
            # Compute circuit depth
            qc = qam.build_insert_circuit(inserted)
            depth = qc.depth()
            depths.append(depth)
            # Compute FP
            fps = 0
            for item in test_negative:
                exp = qam.query(inserted, item, shots=shots, noise_model=noise_model)
                if exp >= 0.5:
                    fps += 1
            fp_rate = fps / len(test_negative)
            # Compute FN
            fns = 0
            for item in test_positive:
                exp = qam.query(inserted, item, shots=shots, noise_model=noise_model)
                if exp < 0.5:
                    fns += 1
            fn_rate = fns / len(test_positive)
            fp_rates.append(fp_rate)
            fn_rates.append(fn_rate)
        results.append({
            'm': m,
            'k': k,
            'set_size': set_size,
            'shots': shots,
            'noise': noise_rate,
            'topology': topo,
            'fp_mean': np.mean(fp_rates),
            'fp_std': np.std(fp_rates),
            'fn_mean': np.mean(fn_rates),
            'fn_std': np.std(fn_rates),
            'depth_mean': np.mean(depths),
            'depth_std': np.std(depths)
        })
    # Save results
    if results:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"Topology results saved to {output_file}")
    return output_file


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
    parser.add_argument('--batch', action='store_true', help='Run batch query experiment')
    parser.add_argument('--heatmap', action='store_true', help='Run shots × noise heatmap sweep')
    parser.add_argument('--topology', action='store_true', help='Run topology comparison sweep')
    parser.add_argument('--q-subsketch', action='store_true', help='Run Q-SubSketch evaluation')
    parser.add_argument('--text-file', type=str, default=None, help='Text file for Q-SubSketch evaluation')
    
    args = parser.parse_args()
    
    if args.q_subsketch:
        run_q_subsketch_evaluation(text_file=args.text_file, m=args.m, k=args.k, shots=args.shots, output_dir=args.output_dir)
    elif args.topology:
        run_topology_sweep(args.m, args.k, args.set_size, args.shots, output_dir=args.output_dir)
    elif args.heatmap:
        run_heatmap_sweep(args.m, args.k, args.set_size, output_dir=args.output_dir)
    elif args.batch:
        run_batch_query_experiment(args.m, args.k, args.set_size, args.shots, batch_sizes=[1, 16, 64], noise_rate=args.noise, output_dir=args.output_dir)
    elif args.sweep:
        run_parameter_sweep(args.output_dir)
    else:
        print(f"Running single experiment: m={args.m}, k={args.k}, |S|={args.set_size}, shots={args.shots}, ε={args.noise}")
        metrics = run_single_experiment(args.m, args.k, args.set_size, args.shots, args.noise)
        print("\nResults:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")


if __name__ == '__main__':
    main()
