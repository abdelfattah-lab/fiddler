#!/usr/bin/env python3
"""
Benchmark comparing all-GPU mode vs prefetch modes.

Tests:
- All-GPU mode (all experts on GPU)
- Prefetch mode with 0, 4, 7, and 16 experts
- Baseline on-demand loading

Generates performance comparison plots and CSV results.
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen import FiddlerQwen
from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch


def benchmark_configuration(model_class, args_dict, num_runs=5, output_tokens=20, **kwargs):
    """Benchmark a specific configuration."""
    print(f"\n{'='*80}")
    print(f"Benchmarking {kwargs.get('name', 'Configuration')}")
    print(f"{'='*80}")

    # Create model
    args = argparse.Namespace(**args_dict)
    if model_class == FiddlerQwen:
        model = FiddlerQwen(args)
    else:
        model = model_class(args, **{k: v for k, v in kwargs.items() if k != 'name'})

    # Warm-up run
    print("\nWarm-up run...")
    model.generate(text="The capital of France is", output_token=output_tokens)

    # Benchmark runs
    times = []
    print(f"\nRunning {num_runs} benchmark iterations...")
    for i in range(num_runs):
        print(f"  Run {i+1}/{num_runs}...", end=" ", flush=True)
        start = time.time()
        model.generate(text="The capital of France is", output_token=output_tokens)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"{elapsed:.3f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nResults:")
    print(f"  Average time: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"  Min time: {np.min(times):.3f}s")
    print(f"  Max time: {np.max(times):.3f}s")

    return {
        'name': kwargs.get('name', 'Configuration'),
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': np.min(times),
        'max_time': np.max(times),
        'all_times': times
    }


def main():
    """Run all benchmarks."""
    # Configuration
    args_dict = {
        'model': 'Qwen/Qwen1.5-MoE-A2.7B',
        'dtype': 'bfloat16',
        'device': 'cuda',
        'cpu_offload': 0,
        'max_experts_gpu': 0,
        'beam_width': 1,
        'use_fiddler_mode': False,
        'fiddler_batch_threshold': 8
    }

    num_runs = 5
    output_tokens = 20

    results = []

    # 1. Baseline (on-demand loading)
    print("\n" + "="*80)
    print("BENCHMARK 1: Baseline (on-demand CPU→GPU loading)")
    print("="*80)
    result = benchmark_configuration(
        FiddlerQwen,
        args_dict,
        num_runs=num_runs,
        output_tokens=output_tokens,
        name="Baseline (on-demand)"
    )
    results.append(result)

    # 2. All-GPU mode
    print("\n" + "="*80)
    print("BENCHMARK 2: All-GPU mode (all experts on GPU)")
    print("="*80)
    result = benchmark_configuration(
        FiddlerQwenWithPrefetch,
        args_dict,
        num_runs=num_runs,
        output_tokens=output_tokens,
        name="All-GPU mode",
        all_gpu_mode=True
    )
    results.append(result)

    # 3. Prefetch with 0 experts (should match baseline)
    print("\n" + "="*80)
    print("BENCHMARK 3: Prefetch (0 experts)")
    print("="*80)
    result = benchmark_configuration(
        FiddlerQwenWithPrefetch,
        args_dict,
        num_runs=num_runs,
        output_tokens=output_tokens,
        name="Prefetch (0 experts)",
        num_experts_to_prefetch=0
    )
    results.append(result)

    # 4. Prefetch with 4 experts
    print("\n" + "="*80)
    print("BENCHMARK 4: Prefetch (4 experts)")
    print("="*80)
    result = benchmark_configuration(
        FiddlerQwenWithPrefetch,
        args_dict,
        num_runs=num_runs,
        output_tokens=output_tokens,
        name="Prefetch (4 experts)",
        num_experts_to_prefetch=4
    )
    results.append(result)

    # 5. Prefetch with 7 experts (optimal from previous benchmarks)
    print("\n" + "="*80)
    print("BENCHMARK 5: Prefetch (7 experts - optimal)")
    print("="*80)
    result = benchmark_configuration(
        FiddlerQwenWithPrefetch,
        args_dict,
        num_runs=num_runs,
        output_tokens=output_tokens,
        name="Prefetch (7 experts)",
        num_experts_to_prefetch=7
    )
    results.append(result)

    # 6. Prefetch with 16 experts
    print("\n" + "="*80)
    print("BENCHMARK 6: Prefetch (16 experts)")
    print("="*80)
    result = benchmark_configuration(
        FiddlerQwenWithPrefetch,
        args_dict,
        num_runs=num_runs,
        output_tokens=output_tokens,
        name="Prefetch (16 experts)",
        num_experts_to_prefetch=16
    )
    results.append(result)

    # Calculate speedups vs baseline
    baseline_time = results[0]['avg_time']
    for result in results:
        result['speedup_vs_baseline'] = baseline_time / result['avg_time']

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Configuration':<25} {'Avg Time (s)':<15} {'Speedup vs Baseline':<20}")
    print("-" * 80)
    for result in results:
        print(f"{result['name']:<25} {result['avg_time']:>6.3f} ± {result['std_time']:.3f}   {result['speedup_vs_baseline']:>6.2f}x")

    # Save results
    import time as time_module
    timestamp = int(time_module.time())
    results_dir = f"all_gpu_benchmark_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    # Save JSON
    json_path = os.path.join(results_dir, "benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'datetime': datetime.now().isoformat(),
            'config': args_dict,
            'num_runs': num_runs,
            'output_tokens': output_tokens,
            'results': results
        }, f, indent=2)

    # Save CSV
    csv_path = os.path.join(results_dir, "benchmark_results.csv")
    with open(csv_path, 'w') as f:
        f.write("Configuration,Avg Time (s),Std Time (s),Min Time (s),Max Time (s),Speedup vs Baseline\n")
        for result in results:
            f.write(f"{result['name']},{result['avg_time']:.4f},{result['std_time']:.4f},"
                   f"{result['min_time']:.4f},{result['max_time']:.4f},{result['speedup_vs_baseline']:.4f}\n")

    print(f"\n✅ Results saved to {results_dir}/")
    print(f"   - benchmark_results.json")
    print(f"   - benchmark_results.csv")

    # Create plot
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Execution time
        names = [r['name'] for r in results]
        times = [r['avg_time'] for r in results]
        stds = [r['std_time'] for r in results]

        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
        bars1 = ax1.bar(range(len(names)), times, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax1.set_xlabel('Configuration', fontsize=12)
        ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
        ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, time, std) in enumerate(zip(bars1, times, stds)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + std,
                    f'{time:.3f}s',
                    ha='center', va='bottom', fontsize=9)

        # Plot 2: Speedup
        speedups = [r['speedup_vs_baseline'] for r in results]
        bars2 = ax2.bar(range(len(names)), speedups, color=colors, alpha=0.8)
        ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Baseline (1.0x)')
        ax2.set_xlabel('Configuration', fontsize=12)
        ax2.set_ylabel('Speedup vs Baseline', fontsize=12)
        ax2.set_title('Speedup vs Baseline', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend()

        # Add value labels on bars
        for i, (bar, speedup) in enumerate(zip(bars2, speedups)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{speedup:.2f}x',
                    ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(results_dir, "comparison_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   - comparison_plot.png")
        print(f"\n📊 Plot saved successfully!")

    except ImportError:
        print("\n⚠️  matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"\n⚠️  Error generating plot: {e}")


if __name__ == "__main__":
    main()
