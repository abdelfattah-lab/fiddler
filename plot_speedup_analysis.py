#!/usr/bin/env python3
"""Plot speedup analysis from expert loading benchmark results."""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot speedup analysis from benchmark results")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory containing benchmark results (auto-detects latest if not specified)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for plot (default: <results-dir>/speedup_analysis.png)")
    return parser.parse_args()


def find_latest_results_dir():
    """Find the most recent expert_loading_benchmark directory."""
    benchmark_dirs = sorted(Path(".").glob("expert_loading_benchmark_*"))
    if not benchmark_dirs:
        raise FileNotFoundError("No expert_loading_benchmark_* directories found")
    return str(benchmark_dirs[-1])


def load_results(results_dir):
    """Load benchmark results from JSON file."""
    json_path = os.path.join(results_dir, "expert_loading_results.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results file not found: {json_path}")

    with open(json_path) as f:
        return json.load(f)


def calculate_speedups(results):
    """Calculate speedup relative to baseline for each configuration."""
    # Find baseline time
    baseline_time = None
    for result in results:
        if "Baseline" in result["config"]:
            baseline_time = result["total_time_mean"]
            break

    if baseline_time is None:
        raise ValueError("Baseline configuration not found in results")

    # Calculate speedups
    speedup_data = []
    for result in results:
        config = result["config"]
        total_time = result["total_time_mean"]
        speedup = baseline_time / total_time

        # Extract prefetch count if applicable
        prefetch_count = None
        if "Prefetch-" in config:
            try:
                prefetch_count = int(config.split("Prefetch-")[1].split()[0])
            except:
                pass

        speedup_data.append({
            "config": config,
            "speedup": speedup,
            "total_time": total_time,
            "prefetch_count": prefetch_count,
            "decode_hit_rate": result.get("decode_hit_rate_mean", 0.0)
        })

    return baseline_time, speedup_data


def plot_speedup_analysis(baseline_time, speedup_data, output_path):
    """Create comprehensive speedup analysis plot."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Sort data for plotting
    configs = [d["config"] for d in speedup_data]
    speedups = [d["speedup"] for d in speedup_data]
    times = [d["total_time"] for d in speedup_data]
    hit_rates = [d["decode_hit_rate"] * 100 for d in speedup_data]

    # Color mapping
    colors = []
    for config in configs:
        if "Baseline" in config:
            colors.append("#1f77b4")  # Blue
        elif "Oracle" in config:
            colors.append("#2ca02c")  # Green
        else:  # Prefetch variants
            colors.append("#ff7f0e")  # Orange

    # Plot 1: Speedup comparison (bar chart)
    bars = ax1.bar(range(len(configs)), speedups, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    ax1.set_ylabel('Speedup vs Baseline', fontsize=12, fontweight='bold')
    ax1.set_title('Expert Loading Strategy Speedup', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()

    # Add speedup values on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 2: Time vs Hit Rate scatter with labels
    # Separate data by type
    baseline_idx = [i for i, c in enumerate(configs) if "Baseline" in c]
    oracle_idx = [i for i, c in enumerate(configs) if "Oracle" in c]
    prefetch_idx = [i for i, c in enumerate(configs) if "Prefetch-" in c]

    # Plot baseline
    if baseline_idx:
        ax2.scatter([hit_rates[i] for i in baseline_idx],
                   [times[i] for i in baseline_idx],
                   color='#1f77b4', s=150, marker='o', label='Baseline', zorder=3)

    # Plot oracle
    if oracle_idx:
        ax2.scatter([hit_rates[i] for i in oracle_idx],
                   [times[i] for i in oracle_idx],
                   color='#2ca02c', s=150, marker='s', label='Oracle', zorder=3)

    # Plot prefetch variants
    if prefetch_idx:
        prefetch_hit_rates = [hit_rates[i] for i in prefetch_idx]
        prefetch_times = [times[i] for i in prefetch_idx]
        prefetch_configs = [configs[i] for i in prefetch_idx]

        ax2.scatter(prefetch_hit_rates, prefetch_times,
                   color='#ff7f0e', s=150, marker='o', label='Prefetch', zorder=3)

        # Add labels for prefetch points
        for hr, t, cfg in zip(prefetch_hit_rates, prefetch_times, prefetch_configs):
            # Extract number from config
            try:
                num = cfg.split("Prefetch-")[1].split()[0]
                ax2.annotate(f'k={num}', (hr, t),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, fontweight='bold')
            except:
                pass

    ax2.set_xlabel('Decode Hit Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Time (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance vs Hit Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Add baseline time reference line
    ax2.axhline(y=baseline_time, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.text(ax2.get_xlim()[1] * 0.95, baseline_time, f'Baseline: {baseline_time:.2f}s',
            ha='right', va='bottom', fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Speedup analysis plot saved to: {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SPEEDUP ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'Time (s)':<12} {'Speedup':<12} {'Hit Rate':<12}")
    print("-"*80)
    for data in speedup_data:
        print(f"{data['config']:<30} {data['total_time']:<12.3f} "
              f"{data['speedup']:<12.2f}x {data['decode_hit_rate']*100:<12.1f}%")
    print("="*80)

    # Find optimal configuration
    prefetch_data = [d for d in speedup_data if "Prefetch-" in d["config"]]
    if prefetch_data:
        optimal = max(prefetch_data, key=lambda x: x["speedup"])
        print(f"\n🏆 Optimal Prefetch Configuration: {optimal['config']}")
        print(f"   Speedup: {optimal['speedup']:.2f}x | Time: {optimal['total_time']:.3f}s | Hit Rate: {optimal['decode_hit_rate']*100:.1f}%")

    oracle_data = [d for d in speedup_data if "Oracle" in d["config"]]
    if oracle_data:
        oracle = oracle_data[0]
        print(f"\n🎯 Oracle (Upper Bound): {oracle['speedup']:.2f}x speedup")
        if prefetch_data:
            gap_closed = (optimal['speedup'] - 1.0) / (oracle['speedup'] - 1.0) * 100
            print(f"   Prefetching closes {gap_closed:.1f}% of gap to oracle performance")
    print()


def main():
    args = parse_args()

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        print("📁 Auto-detecting latest benchmark results directory...")
        results_dir = find_latest_results_dir()
        print(f"   Found: {results_dir}")

    # Load results
    print(f"📖 Loading results from: {results_dir}")
    results = load_results(results_dir)
    print(f"   Loaded {len(results)} configurations")

    # Calculate speedups
    baseline_time, speedup_data = calculate_speedups(results)
    print(f"   Baseline time: {baseline_time:.3f}s")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(results_dir, "speedup_analysis.png")

    # Generate plot
    plot_speedup_analysis(baseline_time, speedup_data, output_path)


if __name__ == "__main__":
    main()
