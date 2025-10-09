#!/usr/bin/env python3
"""
Comprehensive Benchmark: Fiddler CPU Offloading vs Prefetching

This script benchmarks all configurations to prove that Fiddler+Prefetch
achieves better performance than either technique alone.

Configurations tested:
1. Baseline: No prefetch, no CPU offload
2. Prefetch: Varying num_experts (0-16), no CPU offload
3. Fiddler: No prefetch, varying cost parameters
4. Fiddler+Prefetch: Varying num_experts and cost parameters
"""

import torch
import argparse
import sys
import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch


class Args:
    def __init__(self):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.dtype = torch.bfloat16
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1


def run_single_config(config_name, num_experts, enable_cpu_offload, latency_cpu, latency_gpu):
    """Run a single configuration and return results."""
    print(f"\n{'='*80}")
    print(f"Running: {config_name}")
    print(f"  num_experts_to_prefetch={num_experts}")
    print(f"  enable_cpu_offload={enable_cpu_offload}")
    if enable_cpu_offload:
        print(f"  latency_cpu={latency_cpu:.4f}, latency_gpu={latency_gpu:.4f}")
    print(f"{'='*80}")

    args = Args()

    try:
        # Load model
        model = FiddlerQwenWithPrefetch(
            args,
            num_experts_to_prefetch=num_experts,
            enable_cpu_offload=enable_cpu_offload,
            latency_cpu=latency_cpu if enable_cpu_offload else None,
            latency_gpu=latency_gpu if enable_cpu_offload else None
        )

        # Generate
        test_prompt = "The capital of France is ______.\n"

        start_time = time.time()
        prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
            text=test_prompt,
            output_token=20
        )
        total_time = time.time() - start_time

        # Get stats
        prefetch_stats = model.get_prefetch_stats()
        cpu_stats = model.get_cpu_offload_stats() if enable_cpu_offload else None

        result = {
            'config_name': config_name,
            'num_experts': num_experts,
            'enable_cpu_offload': enable_cpu_offload,
            'latency_cpu': latency_cpu if enable_cpu_offload else None,
            'latency_gpu': latency_gpu if enable_cpu_offload else None,
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': prefill_time + decode_time,
            'wall_time': total_time,
            'prefill_hit_rate': prefill_hit_rate,
            'decode_hit_rate': decode_hit_rate,
            'overall_hit_rate': prefetch_stats['overall_hit_rate'],
        }

        # Add CPU offloading stats if enabled
        if cpu_stats:
            result.update({
                'cpu_expert_count': cpu_stats['cpu_expert_count'],
                'gpu_expert_count': cpu_stats['gpu_expert_count'],
                'cpu_percentage': cpu_stats['cpu_percentage'],
                'gpu_percentage': cpu_stats['gpu_percentage']
            })
        else:
            result.update({
                'cpu_expert_count': 0,
                'gpu_expert_count': 0,
                'cpu_percentage': 0,
                'gpu_percentage': 0
            })

        # Clean up
        del model
        torch.cuda.empty_cache()

        print(f"✅ Completed: {config_name}")
        print(f"   Total time: {result['total_time']:.3f}s")
        print(f"   Prefill: {prefill_time:.3f}s, Decode: {decode_time:.3f}s")

        return result

    except Exception as e:
        print(f"❌ Error running {config_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Fiddler vs Prefetch benchmark')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated timestamp)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: test fewer configurations')
    args = parser.parse_args()

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"fiddler_benchmark_{timestamp}"
    else:
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    print("\n" + "="*80)
    print("COMPREHENSIVE FIDDLER vs PREFETCH BENCHMARK")
    print("="*80)

    # Define configurations to test
    configs = []

    # 1. Baseline
    configs.append({
        'name': 'Baseline',
        'num_experts': 0,
        'enable_cpu_offload': False,
        'latency_cpu': None,
        'latency_gpu': None
    })

    # 2. Prefetch only (varying expert counts)
    if args.quick:
        prefetch_counts = [4, 8, 16]
    else:
        prefetch_counts = [0, 2, 4, 6, 8, 10, 12, 14, 16]

    for n in prefetch_counts:
        if n == 0:
            continue  # Skip 0, already tested in baseline
        configs.append({
            'name': f'Prefetch-{n}',
            'num_experts': n,
            'enable_cpu_offload': False,
            'latency_cpu': None,
            'latency_gpu': None
        })

    # 3. Fiddler only (varying cost parameters)
    if args.quick:
        cost_params = [(0.1, 10.0)]
    else:
        cost_params = [
            (0.01, 5.0),   # Low CPU cost, low GPU cost (favor CPU)
            (0.05, 5.0),   # Medium CPU cost, low GPU cost
            (0.1, 10.0),   # Medium CPU cost, medium GPU cost
            (0.2, 20.0),   # High CPU cost, high GPU cost (favor GPU)
        ]

    for latency_cpu, latency_gpu in cost_params:
        configs.append({
            'name': f'Fiddler-c{latency_cpu:.2f}-g{latency_gpu:.1f}',
            'num_experts': 0,
            'enable_cpu_offload': True,
            'latency_cpu': latency_cpu,
            'latency_gpu': latency_gpu
        })

    # 4. Fiddler+Prefetch (best combinations)
    if args.quick:
        combined_configs = [(8, 0.1, 10.0)]
    else:
        combined_configs = [
            (4, 0.05, 5.0),
            (8, 0.05, 5.0),
            (8, 0.1, 10.0),
            (16, 0.1, 10.0),
        ]

    for num_exp, lat_cpu, lat_gpu in combined_configs:
        configs.append({
            'name': f'Fiddler+Prefetch-{num_exp}-c{lat_cpu:.2f}-g{lat_gpu:.1f}',
            'num_experts': num_exp,
            'enable_cpu_offload': True,
            'latency_cpu': lat_cpu,
            'latency_gpu': lat_gpu
        })

    print(f"\nTotal configurations to test: {len(configs)}")

    # Run all configurations
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing: {config['name']}")
        result = run_single_config(
            config['name'],
            config['num_experts'],
            config['enable_cpu_offload'],
            config['latency_cpu'],
            config['latency_gpu']
        )
        if result:
            results.append(result)

    if not results:
        print("❌ No successful runs!")
        return 1

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

    json_path = os.path.join(output_dir, 'benchmark_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {json_path}")

    # Calculate speedups
    baseline_time = df[df['config_name'] == 'Baseline']['total_time'].iloc[0]
    df['speedup'] = baseline_time / df['total_time']
    df['prefill_speedup'] = df[df['config_name'] == 'Baseline']['prefill_time'].iloc[0] / df['prefill_time']
    df['decode_speedup'] = df[df['config_name'] == 'Baseline']['decode_time'].iloc[0] / df['decode_time']

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<30} {'Total(s)':<10} {'Speedup':<10} {'Prefill%':<12} {'Decode%':<12}")
    print("-" * 80)
    for _, row in df.iterrows():
        print(f"{row['config_name']:<30} "
              f"{row['total_time']:<10.3f} "
              f"{row['speedup']:<10.3f}x "
              f"{row['prefill_hit_rate']*100:<12.1f} "
              f"{row['decode_hit_rate']*100:<12.1f}")

    # Find best configurations
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS BY SPEEDUP")
    print("="*80)
    top5 = df.nlargest(5, 'speedup')
    for _, row in top5.iterrows():
        print(f"\n{row['config_name']}:")
        print(f"  Speedup: {row['speedup']:.3f}x")
        print(f"  Total time: {row['total_time']:.3f}s (Prefill: {row['prefill_time']:.3f}s, Decode: {row['decode_time']:.3f}s)")
        print(f"  Hit rates: Prefill {row['prefill_hit_rate']*100:.1f}%, Decode {row['decode_hit_rate']*100:.1f}%")
        if row['enable_cpu_offload']:
            print(f"  CPU/GPU split: {row['cpu_percentage']:.1f}% CPU, {row['gpu_percentage']:.1f}% GPU")

    # Plot results
    create_plots(df, output_dir)

    print(f"\n✅ Benchmark complete! Results saved to {output_dir}")
    return 0


def create_plots(df, output_dir):
    """Create visualization plots."""
    print("\n📊 Creating plots...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Speedup comparison
    ax = axes[0, 0]

    # Separate configurations by type
    baseline = df[df['config_name'] == 'Baseline']
    prefetch_only = df[df['config_name'].str.startswith('Prefetch-')]
    fiddler_only = df[df['config_name'].str.startswith('Fiddler-') & ~df['config_name'].str.contains('\+')]
    fiddler_prefetch = df[df['config_name'].str.contains('Fiddler\+')]

    x_pos = 0
    colors = {'Baseline': 'gray', 'Prefetch': 'blue', 'Fiddler': 'green', 'Fiddler+Prefetch': 'red'}
    labels_added = set()

    for name, color in [('Baseline', 'gray')]:
        if len(baseline) > 0:
            ax.bar(x_pos, baseline['speedup'].iloc[0], color=color, label='Baseline')
            ax.text(x_pos, baseline['speedup'].iloc[0] + 0.05, f"{baseline['speedup'].iloc[0]:.2f}x",
                    ha='center', va='bottom', fontsize=8)
            x_pos += 1

    for subset, color, label in [
        (prefetch_only, 'blue', 'Prefetch'),
        (fiddler_only, 'green', 'Fiddler'),
        (fiddler_prefetch, 'red', 'Fiddler+Prefetch')
    ]:
        if len(subset) > 0:
            for _, row in subset.iterrows():
                ax.bar(x_pos, row['speedup'], color=color, label=label if label not in labels_added else '')
                labels_added.add(label)
                x_pos += 1

    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='Baseline (1.0x)')
    ax.set_ylabel('Speedup vs Baseline')
    ax.set_title('Speedup Comparison Across All Configurations')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Prefetch configurations comparison
    ax = axes[0, 1]
    if len(prefetch_only) > 0:
        prefetch_nums = prefetch_only['num_experts'].values
        prefetch_speedups = prefetch_only['speedup'].values
        ax.plot(prefetch_nums, prefetch_speedups, 'o-', color='blue', label='Prefetch-only')
        ax.set_xlabel('Number of Experts to Prefetch')
        ax.set_ylabel('Speedup vs Baseline')
        ax.set_title('Prefetch Performance vs Expert Count')
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 3. Hit rates
    ax = axes[1, 0]
    x = range(len(df))
    width = 0.35
    ax.bar([i - width/2 for i in x], df['prefill_hit_rate'] * 100, width, label='Prefill', alpha=0.8)
    ax.bar([i + width/2 for i in x], df['decode_hit_rate'] * 100, width, label='Decode', alpha=0.8)
    ax.set_ylabel('Hit Rate (%)')
    ax.set_title('Prefetch Hit Rates by Configuration')
    ax.set_xticks([])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. CPU/GPU distribution for Fiddler configs
    ax = axes[1, 1]
    fiddler_configs = df[df['enable_cpu_offload'] == True]
    if len(fiddler_configs) > 0:
        x_fiddler = range(len(fiddler_configs))
        ax.bar(x_fiddler, fiddler_configs['cpu_percentage'], label='CPU %', alpha=0.8)
        ax.bar(x_fiddler, fiddler_configs['gpu_percentage'],
               bottom=fiddler_configs['cpu_percentage'], label='GPU %', alpha=0.8)
        ax.set_ylabel('Percentage')
        ax.set_title('CPU/GPU Expert Distribution (Fiddler Configs)')
        ax.set_xticks([])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'benchmark_analysis.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"💾 Plots saved to: {plot_path}")
    plt.close()


if __name__ == '__main__':
    sys.exit(main())
