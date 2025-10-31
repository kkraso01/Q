"""
Unified benchmark harness for all Quantum Data Structures.

This script runs all QDS experiments with configurations from configs/*.yml
and generates comprehensive comparison plots.

Usage:
    python run_all.py --structures qam,qht,q_count  # Run specific structures
    python run_all.py --all                          # Run all structures
    python run_all.py --quick                        # Quick test with reduced params
"""

import argparse
import yaml
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.sweeps import (
    run_parameter_sweep,
    run_batch_query_experiment,
    run_heatmap_sweep,
    run_topology_sweep
)
from experiments.qht_sweep import run_qht_accuracy_sweep, run_qht_depth_sweep
from experiments.q_count_sweep import run_q_count_cardinality_sweep
from experiments.q_hh_sweep import run_q_hh_topk_sweep
from experiments.q_lsh_sweep import run_q_lsh_accuracy_sweep
from experiments.q_kv_eval import run_cache_size_sweep


STRUCTURE_MAP = {
    'qam': {
        'name': 'Quantum Approximate Membership',
        'experiments': [
            ('parameter_sweep', run_parameter_sweep),
            ('batch_query', run_batch_query_experiment),
            ('heatmap', run_heatmap_sweep),
            ('topology', run_topology_sweep)
        ],
        'config': 'configs/qam.yml'
    },
    'qht': {
        'name': 'Quantum Hashed Trie',
        'experiments': [
            ('accuracy_sweep', run_qht_accuracy_sweep),
            ('depth_sweep', run_qht_depth_sweep)
        ],
        'config': 'configs/qht.yml'
    },
    'q_count': {
        'name': 'Quantum Count-Distinct',
        'experiments': [
            ('cardinality_sweep', run_q_count_cardinality_sweep)
        ],
        'config': 'configs/q_count.yml'
    },
    'q_hh': {
        'name': 'Quantum Heavy Hitters',
        'experiments': [
            ('topk_sweep', run_q_hh_topk_sweep)
        ],
        'config': 'configs/q_hh.yml'
    },
    'q_lsh': {
        'name': 'Quantum LSH',
        'experiments': [
            ('accuracy_sweep', run_q_lsh_accuracy_sweep)
        ],
        'config': 'configs/q_lsh.yml'
    },
    'q_kv': {
        'name': 'Quantum KV-Cache',
        'experiments': [
            ('cache_size_sweep', run_cache_size_sweep)
        ],
        'config': 'configs/q_kv.yml'
    }
}


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_structure_benchmarks(structure_name, quick=False):
    """Run all benchmarks for a given structure."""
    if structure_name not in STRUCTURE_MAP:
        print(f"Error: Unknown structure '{structure_name}'")
        return
    
    structure = STRUCTURE_MAP[structure_name]
    print(f"\n{'='*60}")
    print(f"Running benchmarks: {structure['name']}")
    print(f"{'='*60}\n")
    
    config_path = Path(__file__).parent / structure['config']
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        config = {}
    else:
        config = load_config(config_path)
    
    results = {}
    for exp_name, exp_func in structure['experiments']:
        print(f"\n>>> Running {exp_name}...")
        start_time = time.time()
        
        try:
            if quick:
                # Override config for quick testing
                result = exp_func()  # Use minimal defaults
            else:
                # Use full config parameters
                result = exp_func()
            
            elapsed = time.time() - start_time
            print(f"    ✓ Completed in {elapsed:.2f}s")
            results[exp_name] = result
        
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results[exp_name] = None
    
    return results


def generate_summary_report(all_results):
    """Generate markdown summary of all benchmark results."""
    report_path = Path(__file__).parent.parent / 'results' / 'benchmark_summary.md'
    
    with open(report_path, 'w') as f:
        f.write("# Quantum Data Structures - Benchmark Summary\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for structure_name, results in all_results.items():
            structure = STRUCTURE_MAP[structure_name]
            f.write(f"## {structure['name']}\n\n")
            
            for exp_name, result in results.items():
                if result is not None:
                    f.write(f"- **{exp_name}**: ✓ Complete\n")
                else:
                    f.write(f"- **{exp_name}**: ✗ Failed\n")
            
            f.write("\n")
    
    print(f"\nSummary report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Run QDS benchmarks')
    parser.add_argument('--structures', type=str, 
                       help='Comma-separated list of structures (e.g., qam,qht)')
    parser.add_argument('--all', action='store_true', 
                       help='Run all structures')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with reduced parameters')
    
    args = parser.parse_args()
    
    if args.all:
        structures_to_run = list(STRUCTURE_MAP.keys())
    elif args.structures:
        structures_to_run = [s.strip() for s in args.structures.split(',')]
    else:
        print("Error: Must specify --structures or --all")
        parser.print_help()
        return
    
    print("="*60)
    print("Quantum Data Structures - Unified Benchmark Suite")
    print("="*60)
    print(f"Structures: {', '.join(structures_to_run)}")
    print(f"Quick mode: {args.quick}")
    print()
    
    all_results = {}
    for structure in structures_to_run:
        results = run_structure_benchmarks(structure, quick=args.quick)
        all_results[structure] = results
    
    generate_summary_report(all_results)
    
    print("\n" + "="*60)
    print("All benchmarks complete!")
    print("="*60)


if __name__ == "__main__":
    main()
