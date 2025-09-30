#!/usr/bin/env python3
"""
Benchmark script to test different prefetch configurations (0-16 experts).
Runs the model with different num_experts_to_prefetch settings and measures performance.
"""

import os
import sys
import time
import torch
import json
import csv
import matplotlib.pyplot as plt
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch
from fiddler.qwen import FiddlerQwen


class Args:
    """Arguments for Qwen model."""
    def __init__(self):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1


def run_baseline():
    """Run baseline model without prefetching."""
    print("\n" + "="*80)
    print("Running BASELINE (FiddlerQwen - no prefetch)")
    print("="*80)

    args = Args()
    model = FiddlerQwen(args)

    # Warmup run
    print("Warmup...")
    model.generate("The capital of France is", output_token=10)

    # Actual benchmark run
    print("\nBenchmark run...")
    start_time = time.time()
    prefill_time, decode_time, hit_rate = model.generate(
        "The capital of France is",
        output_token=20
    )
    total_time = time.time() - start_time

    print(f"✅ Baseline completed: {total_time:.3f}s")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return total_time, hit_rate


def run_prefetch_config(num_experts, pattern_file_exists):
    """Run model with specific prefetch configuration."""
    print("\n" + "="*80)
    print(f"Running PREFETCH with {num_experts} expert(s)")
    print("="*80)

    args = Args()

    # For num_experts=0, we still need to go through prediction mode
    # but we won't actually load any experts
    if num_experts == 0:
        # Make sure pattern file exists for prediction mode
        if not pattern_file_exists:
            print("⚠️  Skipping config with 0 experts - need pattern file first")
            return None, None

    model = FiddlerQwenWithPrefetch(args, num_experts_to_prefetch=num_experts)

    # If in collection mode (no pattern file), run once to collect patterns
    if model.collection_mode:
        print("📊 Collection mode - running to collect expert usage patterns...")
        model.generate("The capital of France is", output_token=20)
        print("✅ Patterns collected and saved")

        # Reload model in prediction mode
        del model
        torch.cuda.empty_cache()
        model = FiddlerQwenWithPrefetch(args, num_experts_to_prefetch=num_experts)

    # Warmup run
    print("Warmup...")
    model.generate("The capital of France is", output_token=10)

    # Actual benchmark run
    print("\nBenchmark run...")
    start_time = time.time()
    prefill_time, decode_time, hit_rate = model.generate(
        "The capital of France is",
        output_token=20
    )
    total_time = time.time() - start_time

    print(f"✅ Config completed: {total_time:.3f}s, Hit rate: {hit_rate:.1%}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return total_time, hit_rate


def plot_results(results, output_dir):
    """Plot speedup vs number of experts prefetched."""
    configs = [r['num_experts'] for r in results]
    speedups = [r['speedup'] for r in results]
    hit_rates = [r['hit_rate'] for r in results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Speedup vs Config
    ax1.plot(configs, speedups, 'bo-', linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Baseline (1.0x)')
    ax1.set_xlabel('Number of Experts Prefetched', fontsize=12)
    ax1.set_ylabel('Speedup vs Baseline', fontsize=12)
    ax1.set_title('Prefetch Performance vs Configuration', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Annotate best configuration
    best_idx = speedups.index(max(speedups))
    ax1.annotate(f'Best: {configs[best_idx]} experts\n{speedups[best_idx]:.2f}x speedup',
                xy=(configs[best_idx], speedups[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 2: Hit Rate vs Config
    ax2.plot(configs, [hr * 100 for hr in hit_rates], 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Experts Prefetched', fontsize=12)
    ax2.set_ylabel('Prefetch Hit Rate (%)', fontsize=12)
    ax2.set_title('Prefetch Hit Rate vs Configuration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, 'prefetch_speedup_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot saved to: {plot_path}")

    # Also display if in interactive mode
    # plt.show()
    plt.close()


def main():
    """Main benchmark function."""
    print("\n" + "="*80)
    print("PREFETCH CONFIGURATION BENCHMARK")
    print("="*80)
    print("This script will:")
    print("1. Run baseline model (no prefetch)")
    print("2. Test prefetch with 0-16 experts")
    print("3. Generate performance comparison plots")
    print("="*80)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"prefetch_benchmark_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Check if pattern file exists
    pattern_file = "expert_usage_patterns_qwen.json"
    pattern_file_exists = os.path.exists(pattern_file)

    # Run baseline
    baseline_time, baseline_hit_rate = run_baseline()

    # Test different configurations
    results = []
    configs_to_test = list(range(0, 17))  # 0 to 16 experts

    for num_experts in configs_to_test:
        prefetch_time, hit_rate = run_prefetch_config(num_experts, pattern_file_exists)

        # Skip if we couldn't run this config
        if prefetch_time is None:
            # After first run, pattern file will exist
            pattern_file_exists = True
            continue

        # Calculate speedup
        speedup = baseline_time / prefetch_time if prefetch_time > 0 else 0.0

        result = {
            'num_experts': num_experts,
            'time': prefetch_time,
            'speedup': speedup,
            'hit_rate': hit_rate
        }
        results.append(result)

        print(f"\n📈 Config {num_experts}: {speedup:.3f}x speedup")

        # Pattern file now exists after first prefetch run
        pattern_file_exists = True

    # Save results to CSV
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['num_experts', 'time', 'speedup', 'hit_rate'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 Results saved to: {csv_path}")

    # Save summary JSON
    summary = {
        'baseline_time': baseline_time,
        'baseline_hit_rate': baseline_hit_rate,
        'timestamp': timestamp,
        'configs': results
    }

    json_path = os.path.join(output_dir, 'benchmark_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Summary saved to: {json_path}")

    # Generate plots
    if results:
        plot_results(results, output_dir)

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Baseline time: {baseline_time:.3f}s")
    print(f"\nBest configurations:")

    # Sort by speedup
    sorted_results = sorted(results, key=lambda x: x['speedup'], reverse=True)
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. {result['num_experts']} experts: {result['speedup']:.3f}x speedup "
              f"({result['time']:.3f}s, {result['hit_rate']:.1%} hit rate)")

    print("="*80)


if __name__ == "__main__":
    main()
