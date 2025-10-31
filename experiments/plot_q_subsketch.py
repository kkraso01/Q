"""
Plotting for Q-SubSketch sweep results.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_fp_vs_L(csv_file, output_dir='results'):
    df = pd.read_csv(csv_file)
    grouped = df.groupby('L').agg({'fp_mean': 'mean', 'fn_mean': 'mean'})
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(grouped.index, grouped['fp_mean'], marker='o', label='False Positive Rate')
    ax.plot(grouped.index, grouped['fn_mean'], marker='s', label='False Negative Rate')
    ax.set_xlabel('Substring Length L')
    ax.set_ylabel('Error Rate')
    ax.set_title('Q-SubSketch: Error Rate vs Substring Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(output_dir).mkdir(exist_ok=True)
    out = Path(output_dir) / 'q_subsketch_fp_vs_L.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True)
    args = parser.parse_args()
    plot_fp_vs_L(args.results)

if __name__ == '__main__':
    main()
