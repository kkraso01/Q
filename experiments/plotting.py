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


def plot_all(csv_file, output_dir='results'):
    """Generate all plots from CSV results."""
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} results from {csv_file}")
    print("\nGenerating plots...")
    
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
