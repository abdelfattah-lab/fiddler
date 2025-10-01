#!/usr/bin/env python3
"""
Plot batch size benchmark results

Creates visualizations comparing:
- Baseline GPU vs Fiddler mode
- Prefetch GPU vs Fiddler mode
- Overall speedups across batch sizes
"""

import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    """Load benchmark results from JSON file."""
    results_file = os.path.join(results_dir, 'batch_size_results.json')
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_batch_size_comparison(results, output_dir):
    """Create comprehensive plots for batch size comparison."""

    # Extract data
    batch_sizes = []
    baseline_times = []
    baseline_fiddler_times = []
    prefetch_times = []
    prefetch_fiddler_times = []

    # Collect all batch sizes tested
    all_batch_sizes = set()
    for config_results in results.values():
        for r in config_results:
            all_batch_sizes.add(r['batch_size'])
    batch_sizes = sorted(all_batch_sizes)

    # Organize results by batch size
    for batch_size in batch_sizes:
        baseline = next((r for r in results.get('baseline', []) if r['batch_size'] == batch_size), None)
        baseline_fiddler = next((r for r in results.get('baseline_fiddler', []) if r['batch_size'] == batch_size), None)
        prefetch = next((r for r in results.get('prefetch', []) if r['batch_size'] == batch_size), None)
        prefetch_fiddler = next((r for r in results.get('prefetch_fiddler', []) if r['batch_size'] == batch_size), None)

        baseline_times.append(baseline['avg_time'] if baseline else None)
        baseline_fiddler_times.append(baseline_fiddler['avg_time'] if baseline_fiddler else None)
        prefetch_times.append(prefetch['avg_time'] if prefetch else None)
        prefetch_fiddler_times.append(prefetch_fiddler['avg_time'] if prefetch_fiddler else None)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Fiddler Mode vs GPU Prefetch: Batch Size Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Absolute execution times
    ax1 = axes[0, 0]
    x = np.arange(len(batch_sizes))
    width = 0.2

    # Filter out None values for plotting
    def filter_none(values, x_vals):
        return [v if v is not None else 0 for v in values], [x_vals[i] for i, v in enumerate(values) if v is not None], [v for v in values if v is not None]

    if any(t for t in baseline_times if t is not None):
        _, x_valid, vals = filter_none(baseline_times, x)
        ax1.bar(np.array(x_valid) - 1.5*width, vals, width, label='Baseline (GPU)', color='#1f77b4')
    if any(t for t in baseline_fiddler_times if t is not None):
        _, x_valid, vals = filter_none(baseline_fiddler_times, x)
        ax1.bar(np.array(x_valid) - 0.5*width, vals, width, label='Baseline (Fiddler)', color='#ff7f0e')
    if any(t for t in prefetch_times if t is not None):
        _, x_valid, vals = filter_none(prefetch_times, x)
        ax1.bar(np.array(x_valid) + 0.5*width, vals, width, label='Prefetch (GPU, 7 experts)', color='#2ca02c')
    if any(t for t in prefetch_fiddler_times if t is not None):
        _, x_valid, vals = filter_none(prefetch_fiddler_times, x)
        ax1.bar(np.array(x_valid) + 1.5*width, vals, width, label='Prefetch (Fiddler)', color='#d62728')

    ax1.set_xlabel('Batch Size', fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontweight='bold')
    ax1.set_title('Execution Time by Batch Size')
    ax1.set_xticks(x)
    ax1.set_xticklabels(batch_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Speedup of Fiddler mode vs GPU (for baseline)
    ax2 = axes[0, 1]
    baseline_fiddler_speedup = []
    for i, batch_size in enumerate(batch_sizes):
        if baseline_times[i] and baseline_fiddler_times[i]:
            speedup = baseline_times[i] / baseline_fiddler_times[i]
            baseline_fiddler_speedup.append(speedup)
        else:
            baseline_fiddler_speedup.append(None)

    valid_idx = [i for i, s in enumerate(baseline_fiddler_speedup) if s is not None]
    valid_batch_sizes = [batch_sizes[i] for i in valid_idx]
    valid_speedups = [baseline_fiddler_speedup[i] for i in valid_idx]

    ax2.plot(valid_batch_sizes, valid_speedups, marker='o', linewidth=2, markersize=8, color='#ff7f0e')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Break-even')
    ax2.set_xlabel('Batch Size', fontweight='bold')
    ax2.set_ylabel('Speedup', fontweight='bold')
    ax2.set_title('Baseline: Fiddler Mode Speedup vs GPU')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels(batch_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Speedup of Prefetch vs Baseline (comparing best strategies)
    ax3 = axes[1, 0]

    # For each batch size, compare the best strategy for prefetch vs best for baseline
    prefetch_best_speedup = []
    for i, batch_size in enumerate(batch_sizes):
        # Choose best baseline time
        baseline_best = min([t for t in [baseline_times[i], baseline_fiddler_times[i]] if t is not None], default=None)
        # Choose best prefetch time
        prefetch_best = min([t for t in [prefetch_times[i], prefetch_fiddler_times[i]] if t is not None], default=None)

        if baseline_best and prefetch_best:
            speedup = baseline_best / prefetch_best
            prefetch_best_speedup.append(speedup)
        else:
            prefetch_best_speedup.append(None)

    valid_idx = [i for i, s in enumerate(prefetch_best_speedup) if s is not None]
    valid_batch_sizes = [batch_sizes[i] for i in valid_idx]
    valid_speedups = [prefetch_best_speedup[i] for i in valid_idx]

    ax3.plot(valid_batch_sizes, valid_speedups, marker='s', linewidth=2, markersize=8, color='#2ca02c')
    ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Break-even')
    ax3.set_xlabel('Batch Size', fontweight='bold')
    ax3.set_ylabel('Speedup', fontweight='bold')
    ax3.set_title('Prefetch Speedup vs Baseline (Best Strategies)')
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(batch_sizes)
    ax3.set_xticklabels(batch_sizes)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Throughput (tokens/second per batch item)
    ax4 = axes[1, 1]

    # Assume 3 tokens generated per prompt
    tokens_per_prompt = 3

    def calc_throughput(times, batch_sizes):
        throughput = []
        for i, time_val in enumerate(times):
            if time_val:
                # tokens per second = (batch_size * tokens_per_prompt) / time
                tps = (batch_sizes[i] * tokens_per_prompt) / time_val
                throughput.append(tps)
            else:
                throughput.append(None)
        return throughput

    baseline_throughput = calc_throughput(baseline_times, batch_sizes)
    baseline_fiddler_throughput = calc_throughput(baseline_fiddler_times, batch_sizes)
    prefetch_throughput = calc_throughput(prefetch_times, batch_sizes)
    prefetch_fiddler_throughput = calc_throughput(prefetch_fiddler_times, batch_sizes)

    if any(baseline_throughput):
        valid_idx = [i for i, t in enumerate(baseline_throughput) if t is not None]
        ax4.plot([batch_sizes[i] for i in valid_idx], [baseline_throughput[i] for i in valid_idx],
                 marker='o', linewidth=2, label='Baseline (GPU)', color='#1f77b4')

    if any(baseline_fiddler_throughput):
        valid_idx = [i for i, t in enumerate(baseline_fiddler_throughput) if t is not None]
        ax4.plot([batch_sizes[i] for i in valid_idx], [baseline_fiddler_throughput[i] for i in valid_idx],
                 marker='^', linewidth=2, label='Baseline (Fiddler)', color='#ff7f0e')

    if any(prefetch_throughput):
        valid_idx = [i for i, t in enumerate(prefetch_throughput) if t is not None]
        ax4.plot([batch_sizes[i] for i in valid_idx], [prefetch_throughput[i] for i in valid_idx],
                 marker='s', linewidth=2, label='Prefetch (GPU)', color='#2ca02c')

    if any(prefetch_fiddler_throughput):
        valid_idx = [i for i, t in enumerate(prefetch_fiddler_throughput) if t is not None]
        ax4.plot([batch_sizes[i] for i in valid_idx], [prefetch_fiddler_throughput[i] for i in valid_idx],
                 marker='D', linewidth=2, label='Prefetch (Fiddler)', color='#d62728')

    ax4.set_xlabel('Batch Size', fontweight='bold')
    ax4.set_ylabel('Throughput (tokens/sec)', fontweight='bold')
    ax4.set_title('Throughput Scaling with Batch Size')
    ax4.set_xscale('log', base=2)
    ax4.set_xticks(batch_sizes)
    ax4.set_xticklabels(batch_sizes)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = os.path.join(output_dir, 'batch_size_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {plot_file}")

    plt.close()

    # Create a second figure with strategy recommendations
    fig2, ax = plt.subplots(figsize=(12, 8))

    # For each batch size, determine which strategy is best
    strategies = []
    for i, batch_size in enumerate(batch_sizes):
        times_dict = {
            'Baseline (GPU)': baseline_times[i],
            'Baseline (Fiddler)': baseline_fiddler_times[i],
            'Prefetch (GPU)': prefetch_times[i],
            'Prefetch (Fiddler)': prefetch_fiddler_times[i]
        }
        # Remove None values
        times_dict = {k: v for k, v in times_dict.items() if v is not None}

        if times_dict:
            best_strategy = min(times_dict, key=times_dict.get)
            best_time = times_dict[best_strategy]
            strategies.append((batch_size, best_strategy, best_time))

    # Create color map
    color_map = {
        'Baseline (GPU)': '#1f77b4',
        'Baseline (Fiddler)': '#ff7f0e',
        'Prefetch (GPU)': '#2ca02c',
        'Prefetch (Fiddler)': '#d62728'
    }

    # Plot bars colored by best strategy
    for batch_size, strategy, time_val in strategies:
        idx = batch_sizes.index(batch_size)
        ax.bar(idx, 1.0, color=color_map[strategy], alpha=0.7, edgecolor='black', linewidth=2)
        ax.text(idx, 0.5, f'{strategy}\n{time_val:.3f}s', ha='center', va='center',
                fontsize=9, fontweight='bold')

    ax.set_xlabel('Batch Size', fontweight='bold', fontsize=12)
    ax.set_ylabel('Best Strategy', fontweight='bold', fontsize=12)
    ax.set_title('Optimal Strategy by Batch Size', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels(batch_sizes)
    ax.set_yticks([])

    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, fc=color, alpha=0.7, edgecolor='black')
                       for color in color_map.values()]
    ax.legend(legend_elements, color_map.keys(), loc='upper left', fontsize=10)

    plt.tight_layout()

    # Save strategy plot
    strategy_plot_file = os.path.join(output_dir, 'optimal_strategy_by_batch_size.png')
    plt.savefig(strategy_plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Strategy plot saved to {strategy_plot_file}")

    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_batch_size_results.py <results_directory>")
        sys.exit(1)

    results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist")
        sys.exit(1)

    print(f"Loading results from {results_dir}...")
    results = load_results(results_dir)

    print("Creating plots...")
    plot_batch_size_comparison(results, results_dir)

    print(f"\n{'='*80}")
    print("Plotting complete!")
    print(f"{'='*80}")
