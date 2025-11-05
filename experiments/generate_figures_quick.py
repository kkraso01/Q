"""
Quick figure generation with reduced parameters for paper submission.
Uses smaller qubit counts to avoid memory issues.
"""
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    """Run command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    """Generate paper figures with reduced parameters."""
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("QUICK FIGURE GENERATION FOR PAPER")
    print("Using m=32 to avoid memory issues")
    print("="*60)
    
    commands = [
        # 1-4: Standard sweep (accuracy vs memory/shots/noise/load)
        "python experiments/sweeps.py --sweep --output-dir results",
        
        # 5: Batch query (smaller m to fit in memory)
        "python experiments/sweeps.py --batch --m 32 --k 3 --set-size 16 --shots 256 --output-dir results",
        
        # 6: Heatmap
        "python experiments/sweeps.py --heatmap --m 32 --k 3 --set-size 16 --output-dir results",
        
        # 7: Topology (smaller m)
        "python experiments/sweeps.py --topology --m 32 --k 3 --set-size 16 --shots 256 --output-dir results",
        
        # 8: Q-SubSketch
        "python experiments/sweeps.py --q-subsketch --m 32 --k 3 --shots 256 --output-dir results"
    ]
    
    print("\nWill run 5 experiments (generates 8+ figures)...\n")
    
    success = 0
    for i, cmd in enumerate(commands, 1):
        print(f"\n\n{'#'*60}")
        print(f"EXPERIMENT {i}/5")
        print(f"{'#'*60}")
        if run_cmd(cmd):
            success += 1
            print(f"✓ Experiment {i} completed")
        else:
            print(f"✗ Experiment {i} failed")
    
    print(f"\n\n{'='*60}")
    print(f"COMPLETED: {success}/5 experiments successful")
    print(f"{'='*60}")
    
    # List generated files
    figures = list(results_dir.glob('*.png'))
    csvs = list(results_dir.glob('*.csv'))
    
    print(f"\nGenerated {len(csvs)} CSV files and {len(figures)} figures")
    print(f"\nFigures in results/:")
    for fig in sorted(figures):
        print(f"  ✓ {fig.name}")
    
    if len(figures) == 0:
        print("\n⚠ No figures generated. Check errors above.")
        print("Try: python experiments/plotting.py --help")

if __name__ == '__main__':
    main()
