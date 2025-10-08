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
    prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
        "The capital of France is",
        output_token=20
    )
    total_time = time.time() - start_time

    print(f"✅ Baseline completed: {total_time:.3f}s (Prefill: {prefill_time:.3f}s, Decode: {decode_time:.3f}s/token)")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return prefill_time, decode_time, total_time, prefill_hit_rate, decode_hit_rate


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
            return None, None, None, None, None

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
    prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
        "The capital of France is",
        output_token=20
    )
    total_time = time.time() - start_time

    print(f"✅ Config completed: {total_time:.3f}s (Prefill: {prefill_time:.3f}s, Decode: {decode_time:.3f}s/token)")
    print(f"   Hit rates - Prefill: {prefill_hit_rate:.1%}, Decode: {decode_hit_rate:.1%}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return prefill_time, decode_time, total_time, prefill_hit_rate, decode_hit_rate


def plot_results(results, output_dir, output_tokens):
    """Plot speedup vs number of experts prefetched."""
    configs = [r['num_experts'] for r in results]
    speedups = [r['speedup'] for r in results]
    prefill_speedups = [r['prefill_speedup'] for r in results]
    decode_speedups = [r['decode_speedup_per_token'] for r in results]
    prefill_hit_rates = [r['prefill_hit_rate'] for r in results]
    decode_hit_rates = [r['decode_hit_rate'] for r in results]

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    # Plot 1: Overall Speedup vs Config
    ax1.plot(configs, speedups, 'bo-', linewidth=2, markersize=8, label='Total')
    ax1.plot(configs, prefill_speedups, 'go-', linewidth=2, markersize=8, label='Prefill')
    ax1.plot(configs, decode_speedups, 'ro-', linewidth=2, markersize=8, label='Decode (per-token)')
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
    ax1.set_xlabel('Number of Experts Prefetched', fontsize=12)
    ax1.set_ylabel('Speedup vs Baseline', fontsize=12)
    ax1.set_title('Prefetch Performance vs Configuration', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Annotate best configurations
    best_idx = speedups.index(max(speedups))
    ax1.annotate(f'Best Total: {configs[best_idx]} experts\n{speedups[best_idx]:.2f}x',
                xy=(configs[best_idx], speedups[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 2: Prefill vs Decode Speedup Comparison
    x = range(len(configs))
    width = 0.35
    ax2.bar([i - width/2 for i in x], prefill_speedups, width, label='Prefill', color='green', alpha=0.7)
    ax2.bar([i + width/2 for i in x], decode_speedups, width, label='Decode (per-token)', color='red', alpha=0.7)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Experts Prefetched', fontsize=12)
    ax2.set_ylabel('Speedup vs Baseline', fontsize=12)
    ax2.set_title('Prefill vs Decode Speedup (per-token)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # Plot 3: Hit Rate vs Config
    ax3.plot(configs, [hr * 100 for hr in prefill_hit_rates], 'go-', linewidth=2, markersize=8, label='Prefill')
    ax3.plot(configs, [hr * 100 for hr in decode_hit_rates], 'ro-', linewidth=2, markersize=8, label='Decode')
    ax3.set_xlabel('Number of Experts Prefetched', fontsize=12)
    ax3.set_ylabel('Prefetch Hit Rate (%)', fontsize=12)
    ax3.set_title('Prefetch Hit Rate vs Configuration', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

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
    baseline_prefill_time, baseline_decode_time, baseline_total_time, baseline_prefill_hit_rate, baseline_decode_hit_rate = run_baseline()

    # Test different configurations
    results = []
    configs_to_test = list(range(0, 17))  # 0 to 16 experts

    for num_experts in configs_to_test:
        prefill_time, decode_time, total_time, prefill_hit_rate, decode_hit_rate = run_prefetch_config(num_experts, pattern_file_exists)

        # Skip if we couldn't run this config
        if prefill_time is None:
            # After first run, pattern file will exist
            pattern_file_exists = True
            continue

        # Calculate speedups
        total_speedup = baseline_total_time / total_time if total_time > 0 else 0.0
        prefill_speedup = baseline_prefill_time / prefill_time if prefill_time > 0 else 0.0
        decode_speedup = baseline_decode_time / decode_time if decode_time > 0 else 0.0

        result = {
            'num_experts': num_experts,
            'prefill_time': prefill_time,
            'decode_time_per_token': decode_time,
            'total_time': total_time,
            'speedup': total_speedup,
            'prefill_speedup': prefill_speedup,
            'decode_speedup_per_token': decode_speedup,
            'prefill_hit_rate': prefill_hit_rate,
            'decode_hit_rate': decode_hit_rate
        }
        results.append(result)

        print(f"\n📈 Config {num_experts}: {total_speedup:.3f}x total speedup (Prefill: {prefill_speedup:.3f}x, Decode/token: {decode_speedup:.3f}x)")

        # Pattern file now exists after first prefetch run
        pattern_file_exists = True

    # Save results to CSV
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['num_experts', 'prefill_time', 'decode_time_per_token', 'total_time',
                                                 'prefill_speedup', 'decode_speedup_per_token', 'speedup',
                                                 'prefill_hit_rate', 'decode_hit_rate'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 Results saved to: {csv_path}")

    # Save summary JSON
    summary = {
        'baseline_prefill_time': baseline_prefill_time,
        'baseline_decode_time_per_token': baseline_decode_time,
        'baseline_total_time': baseline_total_time,
        'baseline_prefill_hit_rate': baseline_prefill_hit_rate,
        'baseline_decode_hit_rate': baseline_decode_hit_rate,
        'output_tokens': configs_to_test[0] if configs_to_test else 20,  # Store output_tokens used
        'timestamp': timestamp,
        'configs': results
    }

    json_path = os.path.join(output_dir, 'benchmark_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Summary saved to: {json_path}")

    # Generate plots
    if results:
        plot_results(results, output_dir, 20)  # Pass output_tokens

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"Baseline timing:")
    print(f"  Prefill: {baseline_prefill_time:.3f}s")
    print(f"  Decode:  {baseline_decode_time:.3f}s/token")
    print(f"  Total:   {baseline_total_time:.3f}s")
    print(f"\nBest configurations by total speedup:")

    # Sort by total speedup
    sorted_results = sorted(results, key=lambda x: x['speedup'], reverse=True)
    for i, result in enumerate(sorted_results[:5]):
        print(f"{i+1}. {result['num_experts']} experts: {result['speedup']:.3f}x total speedup "
              f"(Prefill: {result['prefill_speedup']:.3f}x, Decode/token: {result['decode_speedup_per_token']:.3f}x)")
        print(f"   Hit rates - Prefill: {result['prefill_hit_rate']:.1%}, Decode: {result['decode_hit_rate']:.1%}")

    print(f"\nBest configurations by prefill speedup:")
    sorted_by_prefill = sorted(results, key=lambda x: x['prefill_speedup'], reverse=True)
    for i, result in enumerate(sorted_by_prefill[:3]):
        print(f"{i+1}. {result['num_experts']} experts: {result['prefill_speedup']:.3f}x prefill speedup")

    print(f"\nBest configurations by decode speedup (per-token):")
    sorted_by_decode = sorted(results, key=lambda x: x['decode_speedup_per_token'], reverse=True)
    for i, result in enumerate(sorted_by_decode[:3]):
        print(f"{i+1}. {result['num_experts']} experts: {result['decode_speedup_per_token']:.3f}x decode speedup per token")

    print("="*80)


if __name__ == "__main__":
    main()
