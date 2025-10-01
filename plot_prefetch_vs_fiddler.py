#!/usr/bin/env python3
"""
Visualization script for prefetch vs Fiddler benchmark results.
Creates comprehensive plots showing speedup of each prefetch config vs baseline.
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict


def load_results(results_dir):
    """Load benchmark results from JSON file."""
    json_path = os.path.join(results_dir, "benchmark_results.json")
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_heatmap(results, output_dir):
    """Create heatmap of speedup for each prefetch config vs batch size."""
    # Organize data
    batch_sizes = sorted(set(r['batch_size'] for r in results))
    prefetch_configs = sorted(set(r['num_experts_prefetch'] for r in results if r['config'] != 'Baseline'))

    # Create speedup matrix
    speedup_matrix = np.zeros((len(prefetch_configs), len(batch_sizes)))

    for i, num_experts in enumerate(prefetch_configs):
        for j, batch_size in enumerate(batch_sizes):
            matching = [r for r in results
                       if r['batch_size'] == batch_size
                       and r['num_experts_prefetch'] == num_experts]
            if matching:
                speedup_matrix[i, j] = matching[0]['speedup_vs_baseline']

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto', vmin=0.8, vmax=1.5)

    # Set ticks
    ax.set_xticks(np.arange(len(batch_sizes)))
    ax.set_yticks(np.arange(len(prefetch_configs)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticklabels(prefetch_configs)

    # Labels
    ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Experts to Prefetch', fontsize=12, fontweight='bold')
    ax.set_title('Speedup of Prefetch vs Fiddler Baseline\n(use_fiddler_mode=True: CPU exec for batch<8, GPU exec for batch>=8)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Speedup (×)', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    # Add text annotations
    for i in range(len(prefetch_configs)):
        for j in range(len(batch_sizes)):
            text = ax.text(j, i, f'{speedup_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)

    # Add vertical line to show CPU/GPU transition
    ax.axvline(x=2.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(2.5, len(prefetch_configs) + 0.5, 'CPU → GPU',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='blue')

    plt.tight_layout()

    output_path = os.path.join(output_dir, "speedup_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Heatmap saved to: {output_path}")
    plt.close()


def plot_by_batch_size(results, output_dir):
    """Create line plots showing speedup vs prefetch config for each batch size."""
    batch_sizes = sorted(set(r['batch_size'] for r in results))

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, batch_size in enumerate(batch_sizes):
        ax = axes[idx]

        # Get data for this batch size
        batch_results = [r for r in results if r['batch_size'] == batch_size and r['config'] != 'Baseline']
        batch_results = sorted(batch_results, key=lambda x: x['num_experts_prefetch'])

        num_experts = [r['num_experts_prefetch'] for r in batch_results]
        speedups = [r['speedup_vs_baseline'] for r in batch_results]
        hit_rates = [r['hit_rate'] if r['hit_rate'] is not None else 0 for r in batch_results]

        # Plot speedup
        color = 'tab:blue' if batch_size < 8 else 'tab:red'
        exec_mode = 'CPU' if batch_size < 8 else 'GPU'

        ax.plot(num_experts, speedups, marker='o', linewidth=2, markersize=6, color=color, label='Speedup')
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Number of Experts Prefetched', fontweight='bold')
        ax.set_ylabel('Speedup vs Baseline', fontweight='bold', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_title(f'Batch Size = {batch_size} ({exec_mode} execution)', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Plot hit rate on secondary axis
        ax2 = ax.twinx()
        ax2.plot(num_experts, hit_rates, marker='s', linewidth=2, markersize=4,
                color='tab:green', alpha=0.6, linestyle='--', label='Hit Rate')
        ax2.set_ylabel('Hit Rate (%)', fontweight='bold', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        # Find best configuration
        best_idx = np.argmax(speedups)
        best_num_experts = num_experts[best_idx]
        best_speedup = speedups[best_idx]
        ax.plot(best_num_experts, best_speedup, marker='*', markersize=15,
               color='gold', markeredgecolor='black', markeredgewidth=1.5, zorder=10)

    plt.suptitle('Speedup vs Number of Experts Prefetched (by Batch Size)\nuse_fiddler_mode=True',
                 fontsize=16, fontweight='bold', y=1.02)

    # Create custom legend
    speedup_patch = mpatches.Patch(color='tab:blue', label='Speedup (CPU exec)')
    speedup_gpu_patch = mpatches.Patch(color='tab:red', label='Speedup (GPU exec)')
    hitrate_patch = mpatches.Patch(color='tab:green', label='Hit Rate')
    best_marker = plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                             markersize=12, markeredgecolor='black', label='Best Config')
    fig.legend(handles=[speedup_patch, speedup_gpu_patch, hitrate_patch, best_marker],
              loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout()

    output_path = os.path.join(output_dir, "speedup_by_batch_size.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Line plots saved to: {output_path}")
    plt.close()


def plot_best_config_per_batch(results, output_dir):
    """Plot showing the best prefetch configuration for each batch size."""
    batch_sizes = sorted(set(r['batch_size'] for r in results))

    best_configs = []
    best_speedups = []
    best_hit_rates = []

    for batch_size in batch_sizes:
        batch_results = [r for r in results if r['batch_size'] == batch_size and r['config'] != 'Baseline']
        if batch_results:
            best = max(batch_results, key=lambda x: x['speedup_vs_baseline'])
            best_configs.append(best['num_experts_prefetch'])
            best_speedups.append(best['speedup_vs_baseline'])
            best_hit_rates.append(best['hit_rate'] if best['hit_rate'] is not None else 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Best number of experts and speedup
    colors = ['tab:blue' if bs < 8 else 'tab:red' for bs in batch_sizes]

    ax1_twin = ax1.twinx()
    bars = ax1.bar(range(len(batch_sizes)), best_configs, color=colors, alpha=0.7, edgecolor='black')
    line = ax1_twin.plot(range(len(batch_sizes)), best_speedups, marker='o', color='green',
                         linewidth=3, markersize=10, label='Speedup')

    ax1.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Best # Experts to Prefetch', fontweight='bold', fontsize=12)
    ax1_twin.set_ylabel('Speedup vs Baseline', fontweight='bold', fontsize=12, color='green')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1_twin.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_xticks(range(len(batch_sizes)))
    ax1.set_xticklabels(batch_sizes)
    ax1.set_title('Best Prefetch Configuration per Batch Size', fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bs, config, speedup) in enumerate(zip(batch_sizes, best_configs, best_speedups)):
        ax1.text(i, config + 0.3, f'{config}', ha='center', va='bottom', fontweight='bold')
        ax1_twin.text(i, speedup + 0.01, f'{speedup:.2f}x', ha='center', va='bottom',
                     fontsize=9, color='green', fontweight='bold')

    # Add CPU/GPU labels
    ax1.text(1, max(best_configs) * 0.9, 'CPU', ha='center', fontsize=12,
            fontweight='bold', color='white', bbox=dict(boxstyle='round', facecolor='tab:blue', alpha=0.8))
    ax1.text(5, max(best_configs) * 0.9, 'GPU', ha='center', fontsize=12,
            fontweight='bold', color='white', bbox=dict(boxstyle='round', facecolor='tab:red', alpha=0.8))

    # Plot 2: Execution time comparison
    baseline_times = []
    best_times = []

    for batch_size in batch_sizes:
        baseline = [r for r in results if r['batch_size'] == batch_size and r['config'] == 'Baseline']
        batch_results = [r for r in results if r['batch_size'] == batch_size and r['config'] != 'Baseline']

        if baseline and batch_results:
            baseline_times.append(baseline[0]['median_time'])
            best = max(batch_results, key=lambda x: x['speedup_vs_baseline'])
            best_times.append(best['median_time'])

    x = np.arange(len(batch_sizes))
    width = 0.35

    bars1 = ax2.bar(x - width/2, baseline_times, width, label='Baseline (Fiddler)',
                   color='lightgray', edgecolor='black')
    bars2 = ax2.bar(x + width/2, best_times, width, label='Best Prefetch Config',
                   color=['tab:blue' if bs < 8 else 'tab:red' for bs in batch_sizes],
                   alpha=0.7, edgecolor='black')

    ax2.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Execution Time (seconds)', fontweight='bold', fontsize=12)
    ax2.set_title('Execution Time: Baseline vs Best Prefetch Configuration', fontweight='bold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Best Prefetch Configuration Analysis\nuse_fiddler_mode=True',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "best_config_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Best config analysis saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot prefetch vs Fiddler benchmark results')
    parser.add_argument('results_dir', help='Directory containing benchmark_results.json')
    args = parser.parse_args()

    print("=" * 80)
    print("PLOTTING PREFETCH vs FIDDLER BENCHMARK RESULTS")
    print("=" * 80)

    # Load results
    results = load_results(args.results_dir)
    print(f"\nLoaded {len(results)} result entries from {args.results_dir}")

    # Create plots
    print("\nGenerating visualizations...")
    plot_heatmap(results, args.results_dir)
    plot_by_batch_size(results, args.results_dir)
    plot_best_config_per_batch(results, args.results_dir)

    print("\n" + "=" * 80)
    print(f"All plots saved to: {args.results_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
