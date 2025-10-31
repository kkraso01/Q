"""
Generate all required figures for the QDS paper.

Runs all experiments and generates 6-10+ reproducible figures.
"""
import subprocess
import sys
from pathlib import Path
import time

def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR in {description}:")
        print(result.stderr)
        return None
    
    print(f"SUCCESS: {description}")
    return result.stdout

def main():
    """Generate all figures."""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("QUANTUM DATA STRUCTURES - FIGURE GENERATION")
    print("=" * 60)
    
    experiments = [
        {
            'name': 'Standard Parameter Sweep',
            'cmd': 'python experiments/sweeps.py --sweep',
            'description': 'Generates: accuracy vs memory, shots, noise, load factor'
        },
        {
            'name': 'Batch Query Experiment',
            'cmd': 'python experiments/sweeps.py --batch --shots 512',
            'description': 'Generates: error vs amortized cost for batches [1, 16, 64]'
        },
        {
            'name': 'Heatmap: Shots × Noise',
            'cmd': 'python experiments/sweeps.py --heatmap',
            'description': 'Generates: 2D heatmap of FP/FN vs shots × noise'
        },
        {
            'name': 'Topology Comparison',
            'cmd': 'python experiments/sweeps.py --topology --shots 512',
            'description': 'Generates: accuracy and depth vs topology (none, linear, ring, all-to-all)'
        },
        {
            'name': 'Q-SubSketch Evaluation',
            'cmd': 'python experiments/sweeps.py --q-subsketch',
            'description': 'Generates: AUC vs substring length'
        }
    ]
    
    csv_files = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n[{i}/{len(experiments)}] {exp['name']}")
        print(f"Description: {exp['description']}")
        
        start_time = time.time()
        output = run_command(exp['cmd'], exp['name'])
        elapsed = time.time() - start_time
        
        if output:
            print(f"Completed in {elapsed:.1f} seconds")
            # Extract CSV filename from output
            for line in output.split('\n'):
                if 'saved to' in line.lower() or '.csv' in line:
                    # Try to extract path
                    parts = line.split()
                    for part in parts:
                        if part.endswith('.csv'):
                            csv_files.append(part)
        else:
            print(f"FAILED after {elapsed:.1f} seconds")
    
    print("\n\n" + "="*60)
    print("GENERATING PLOTS FROM RESULTS")
    print("="*60)
    
    # Find all CSV files in results directory
    csv_files = list(results_dir.glob('*.csv'))
    
    if not csv_files:
        print("ERROR: No CSV files found in results directory!")
        return
    
    print(f"\nFound {len(csv_files)} CSV files:")
    for csv in csv_files:
        print(f"  - {csv.name}")
    
    # Generate plots for each CSV
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Plotting {csv_file.name}")
        cmd = f'python experiments/plotting.py --results "{csv_file}"'
        run_command(cmd, f"Plot generation for {csv_file.name}")
    
    print("\n\n" + "="*60)
    print("SUMMARY: GENERATED FIGURES")
    print("="*60)
    
    # List all generated figures
    figures = list(results_dir.glob('*.png')) + list(results_dir.glob('*.svg'))
    
    print(f"\nTotal figures generated: {len(figures)}")
    print("\nFigures:")
    for fig in sorted(figures):
        print(f"  ✓ {fig.name}")
    
    print("\n" + "="*60)
    print("FIGURE GENERATION COMPLETE!")
    print("="*60)
    print(f"\nAll results saved in: {results_dir.absolute()}")
    print("\nExpected figures (6-10+):")
    print("  1. accuracy_vs_memory.png")
    print("  2. accuracy_vs_shots.png")
    print("  3. accuracy_vs_noise.png")
    print("  4. accuracy_vs_load_factor.png")
    print("  5. batch_query_error_vs_amortized_cost.png")
    print("  6. heatmap_shots_noise.png")
    print("  7. topology_comparison.png")
    print("  8. q_subsketch_auc.png")
    print("\n" + "="*60)

if __name__ == '__main__':
    main()
