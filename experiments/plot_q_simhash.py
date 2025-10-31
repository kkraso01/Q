"""
Plotting for Q-SimHash sweep results.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_similarity(csv_file, output_dir='results'):
    df = pd.read_csv(csv_file)
    grouped = df.groupby('shots').agg({'sim_same_mean': 'mean', 'sim_diff_mean': 'mean'})
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(grouped.index, grouped['sim_same_mean'], marker='o', label='Identical Vectors')
    ax.plot(grouped.index, grouped['sim_diff_mean'], marker='s', label='Random Vectors')
    ax.set_xlabel('Shots')
    ax.set_ylabel('Measured Similarity')
    ax.set_title('Q-SimHash: Similarity vs Shots')
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(output_dir).mkdir(exist_ok=True)
    out = Path(output_dir) / 'q_simhash_similarity_vs_shots.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True)
    args = parser.parse_args()
    plot_similarity(args.results)

if __name__ == '__main__':
    main()
