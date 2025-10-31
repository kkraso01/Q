def plot_batch_query_error_vs_amortized_cost(df, output_dir='results'):
    """Plot error (FP, FN) vs amortized cost for different batch sizes."""
    Path(output_dir).mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    # Group by batch size and amortized shots
    grouped = df.groupby(['batch_size', 'amortized_shots']).agg({
        'fp_rate': 'mean',
        'fn_rate': 'mean'
    }).reset_index()
    batch_sizes = sorted(df['batch_size'].unique())
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    for i, batch in enumerate(batch_sizes):
        sub = grouped[grouped['batch_size'] == batch]
        ax.plot(sub['amortized_shots'], sub['fp_rate'], marker='o', label=f'FP, batch={batch}', color=colors[i%len(colors)])
        ax.plot(sub['amortized_shots'], sub['fn_rate'], marker='s', linestyle='--', label=f'FN, batch={batch}', color=colors[i%len(colors)])
    ax.set_xlabel('Amortized Shots per Query', fontsize=12)
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title('Batch Query: Error vs Amortized Cost', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    output_file = Path(output_dir) / 'batch_query_error_vs_amortized_cost.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
"""
Plotting utilities for QAM experimental results.

Generates publication-quality plots for accuracy, variance, and robustness analysis.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_accuracy_vs_memory(df, output_dir='results'):
    """Plot false positive rate vs memory (qubits) for QAM and classical."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by m, average over other parameters
    grouped = df.groupby('m').agg({
        'qam_fp_mean': 'mean',
        'classical_fp_mean': 'mean',
        'qam_fp_std': 'mean',
        'classical_fp_std': 'mean'
    })
    
    m_vals = grouped.index
    
    ax.errorbar(m_vals, grouped['qam_fp_mean'], yerr=grouped['qam_fp_std'], 
                label='QAM', marker='o', capsize=5, linewidth=2)
    ax.errorbar(m_vals, grouped['classical_fp_mean'], yerr=grouped['classical_fp_std'],
                label='Classical Bloom', marker='s', capsize=5, linewidth=2)
    
    ax.set_xlabel('Memory (qubits/bits)', fontsize=12)
    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_title('Accuracy vs Memory', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    output_file = Path(output_dir) / 'accuracy_vs_memory.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_accuracy_vs_shots(df, output_dir='results'):
    """Plot accuracy vs number of measurement shots."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter for noise=0 to isolate shot variance
    df_clean = df[df['noise'] == 0.0]
    grouped = df_clean.groupby('shots').agg({
        'qam_fp_mean': 'mean',
        'qam_fp_std': 'mean'
    })
    
    shots_vals = grouped.index
    
    ax.errorbar(shots_vals, grouped['qam_fp_mean'], yerr=grouped['qam_fp_std'],
                marker='o', capsize=5, linewidth=2, color='#2E86AB')
    
    ax.set_xlabel('Number of Shots', fontsize=12)
    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_title('QAM Accuracy vs Shots (No Noise)', fontsize=14, fontweight='bold')
    ax.set_xscale('log', base=2)
    ax.grid(True, alpha=0.3)
    
    output_file = Path(output_dir) / 'accuracy_vs_shots.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_accuracy_vs_noise(df, output_dir='results'):
    """Plot robustness: accuracy vs noise level."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    grouped = df.groupby('noise').agg({
        'qam_fp_mean': 'mean',
        'qam_fp_std': 'mean'
    })
    
    noise_vals = grouped.index
    
    ax.errorbar(noise_vals, grouped['qam_fp_mean'], yerr=grouped['qam_fp_std'],
                marker='o', capsize=5, linewidth=2, color='#A23B72')
    
    ax.set_xlabel('Noise Rate (ε)', fontsize=12)
    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_title('QAM Robustness vs Noise', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    output_file = Path(output_dir) / 'accuracy_vs_noise.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_load_factor_analysis(df, output_dir='results'):
    """Plot false positive rate vs load factor (|S|/m)."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by load factor
    grouped = df.groupby('load_factor').agg({
        'qam_fp_mean': 'mean',
        'classical_fp_mean': 'mean',
        'qam_fp_std': 'mean',
        'classical_fp_std': 'mean'
    })
    
    load_vals = grouped.index
    
    ax.errorbar(load_vals, grouped['qam_fp_mean'], yerr=grouped['qam_fp_std'],
                label='QAM', marker='o', capsize=5, linewidth=2)
    ax.errorbar(load_vals, grouped['classical_fp_mean'], yerr=grouped['classical_fp_std'],
                label='Classical Bloom', marker='s', capsize=5, linewidth=2)
    
    ax.set_xlabel('Load Factor (|S|/m)', fontsize=12)
    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_title('Accuracy vs Load Factor', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    output_file = Path(output_dir) / 'accuracy_vs_load_factor.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_heatmap_shots_noise(df, output_dir='results'):
    """Plot 2D heatmap of error vs shots × noise."""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Create pivot tables for FP and FN
    fp_pivot = df.pivot_table(values='fp_mean', index='noise', columns='shots', aggfunc='mean')
    fn_pivot = df.pivot_table(values='fn_mean', index='noise', columns='shots', aggfunc='mean')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # FP heatmap
    im1 = ax1.imshow(fp_pivot.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax1.set_xticks(range(len(fp_pivot.columns)))
    ax1.set_xticklabels(fp_pivot.columns)
    ax1.set_yticks(range(len(fp_pivot.index)))
    ax1.set_yticklabels([f'{x:.3f}' for x in fp_pivot.index])
    ax1.set_xlabel('Shots', fontsize=12)
    ax1.set_ylabel('Noise Rate (ε)', fontsize=12)
    ax1.set_title('False Positive Rate: Shots × Noise', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('FP Rate', fontsize=11)
    
    # FN heatmap
    im2 = ax2.imshow(fn_pivot.values, cmap='YlGnBu', aspect='auto', interpolation='nearest')
    ax2.set_xticks(range(len(fn_pivot.columns)))
    ax2.set_xticklabels(fn_pivot.columns)
    ax2.set_yticks(range(len(fn_pivot.index)))
    ax2.set_yticklabels([f'{x:.3f}' for x in fn_pivot.index])
    ax2.set_xlabel('Shots', fontsize=12)
    ax2.set_ylabel('Noise Rate (ε)', fontsize=12)
    ax2.set_title('False Negative Rate: Shots × Noise', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('FN Rate', fontsize=11)
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'heatmap_shots_noise.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_topology_comparison(df, output_dir='results'):
    """Plot FP rate and circuit depth vs topology."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    topologies = df['topology'].unique()
    
    # FP rate vs topology
    fp_means = [df[df['topology'] == t]['fp_mean'].mean() for t in topologies]
    fp_stds = [df[df['topology'] == t]['fp_std'].mean() for t in topologies]
    
    ax1.bar(topologies, fp_means, yerr=fp_stds, capsize=5, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ax1.set_xlabel('Topology', fontsize=12)
    ax1.set_ylabel('False Positive Rate', fontsize=12)
    ax1.set_title('Accuracy vs Topology', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Circuit depth vs topology
    depth_means = [df[df['topology'] == t]['depth_mean'].mean() for t in topologies]
    depth_stds = [df[df['topology'] == t]['depth_std'].mean() for t in topologies]
    
    ax2.bar(topologies, depth_means, yerr=depth_stds, capsize=5, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    ax2.set_xlabel('Topology', fontsize=12)
    ax2.set_ylabel('Circuit Depth', fontsize=12)
    ax2.set_title('Circuit Depth vs Topology', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = Path(output_dir) / 'topology_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_q_subsketch_auc(df, output_dir='results'):
    """Plot AUC vs substring length for Q-SubSketch."""
    Path(output_dir).mkdir(exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    L_vals = sorted(df['L'].unique())
    auc_means = [df[df['L'] == L]['auc_mean'].mean() for L in L_vals]
    auc_stds = [df[df['L'] == L]['auc_std'].mean() for L in L_vals]
    
    ax.errorbar(L_vals, auc_means, yerr=auc_stds, marker='o', capsize=5, linewidth=2, color='#2E86AB')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random baseline (AUC=0.5)')
    ax.set_xlabel('Substring Length (L)', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('Q-SubSketch: AUC vs Substring Length', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    output_file = Path(output_dir) / 'q_subsketch_auc.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_all(csv_file, output_dir='results'):
    """Generate all plots from CSV results."""
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} results from {csv_file}")
    print("\nGenerating plots...")
    # Detect CSV type and generate appropriate plots
    if 'batch_size' in df.columns and 'amortized_shots' in df.columns:
        # Batch query CSV
        plot_batch_query_error_vs_amortized_cost(df, output_dir)
    elif 'topology' in df.columns and 'depth_mean' in df.columns:
        # Topology CSV
        plot_topology_comparison(df, output_dir)
    elif 'auc_mean' in df.columns and 'L' in df.columns:
        # Q-SubSketch CSV
        plot_q_subsketch_auc(df, output_dir)
    elif 'fp_mean' in df.columns and 'fn_mean' in df.columns and 'shots' in df.columns and 'noise' in df.columns and len(df['shots'].unique()) > 1 and len(df['noise'].unique()) > 1:
        # Heatmap CSV
        plot_heatmap_shots_noise(df, output_dir)
    else:
        # Standard sweep CSV
        plot_accuracy_vs_memory(df, output_dir)
        plot_accuracy_vs_shots(df, output_dir)
        plot_accuracy_vs_noise(df, output_dir)
        plot_load_factor_analysis(df, output_dir)
    print("\nAll plots generated!")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from QAM sweep results')
    parser.add_argument('--results', type=str, required=True, help='Path to CSV results file')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    plot_all(args.results, args.output_dir)


if __name__ == '__main__':
    main()
