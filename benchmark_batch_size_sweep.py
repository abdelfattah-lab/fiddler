#!/usr/bin/env python3
"""
Batch size sweep benchmark to find configurations where Fiddler+Prefetch
outperforms Fiddler alone.

Tests 4 configurations across multiple batch sizes:
1. Baseline (no prefetch, no Fiddler)
2. Prefetch (8 experts)
3. Fiddler (CPU offloading, no prefetch)
4. Fiddler+Prefetch (8 experts + CPU offloading)
"""

import os
import sys
import time
import torch
import json
import csv
import numpy as np
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


def run_config(config_name, batch_size, num_experts_to_prefetch=0, enable_cpu_offload=False):
    """Run a single configuration with specified batch size."""
    print(f"\n{'='*80}")
    print(f"Running {config_name} | Batch Size: {batch_size}")
    print(f"{'='*80}")

    args = Args()

    # Create model based on configuration
    if config_name == "Baseline":
        model = FiddlerQwen(args)
    else:
        model = FiddlerQwenWithPrefetch(
            args,
            num_experts_to_prefetch=num_experts_to_prefetch,
            enable_cpu_offload=enable_cpu_offload
        )

        # If in collection mode, run once to collect patterns
        if hasattr(model, 'collection_mode') and model.collection_mode:
            print("📊 Collection mode - collecting patterns...")
            # Use batch size 1 for collection to keep patterns simple
            model.generate("The capital of France is", output_token=20)
            print("✅ Patterns collected")

            # Reload in prediction mode
            del model
            torch.cuda.empty_cache()
            model = FiddlerQwenWithPrefetch(
                args,
                num_experts_to_prefetch=num_experts_to_prefetch,
                enable_cpu_offload=enable_cpu_offload
            )

    # Create batched input - repeat the same prompt batch_size times
    prompts = ["The capital of France is"] * batch_size

    # Warmup
    print("Warmup...")
    warmup_prompts = ["The capital of France is"] * min(batch_size, 2)
    if batch_size == 1:
        model.generate(warmup_prompts[0], output_token=5)
    else:
        # For batch_size > 1, we need to tokenize the batch manually
        inputs = model.tokenizer(warmup_prompts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.model.device)
        attention_mask = inputs.attention_mask.to(model.model.device)
        # Quick forward pass for warmup
        with torch.no_grad():
            _ = model.model.model.embed_tokens(input_ids)

    # Benchmark run
    print("Benchmark run...")
    torch.cuda.synchronize()
    start_time = time.time()

    if batch_size == 1:
        # Single input - use generate method
        prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
            prompts[0],
            output_token=20
        )
    else:
        # Batched input - need to handle manually
        inputs = model.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(model.model.device)
        attention_mask = inputs.attention_mask.to(model.model.device)

        # Reset stats
        if hasattr(model, 'expert_fetch_count'):
            model.expert_fetch_count = 0
            model.expert_hit_count = 0
            model.cnt_expert_hit = 0
            model.cnt_expert_all = 0
            model.prefill_hit_count = 0
            model.prefill_total = 0
            model.decode_hit_count = 0
            model.decode_total = 0

        # Generate tokens autoregressively
        generated = input_ids
        prefill_time = 0.0
        decode_time = 0.0

        for i in range(20):  # Generate 20 tokens
            with torch.no_grad():
                is_decode = (i > 0)

                # Time this forward pass
                torch.cuda.synchronize()
                step_start = time.time()

                outputs = model.model(
                    input_ids=generated,
                    attention_mask=attention_mask,
                    use_cache=False
                )

                torch.cuda.synchronize()
                step_time = time.time() - step_start

                if is_decode:
                    decode_time += step_time
                else:
                    prefill_time = step_time

                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=-1)

                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=attention_mask.device)
                ], dim=-1)

        # Calculate hit rates
        if hasattr(model, 'prefill_total') and model.prefill_total > 0:
            prefill_hit_rate = model.prefill_hit_count / model.prefill_total
        else:
            prefill_hit_rate = 0.0

        if hasattr(model, 'decode_total') and model.decode_total > 0:
            decode_hit_rate = model.decode_hit_count / model.decode_total
        else:
            decode_hit_rate = 0.0

    torch.cuda.synchronize()
    total_time = time.time() - start_time

    print(f"✅ Completed: {total_time:.3f}s (Prefill: {prefill_time:.3f}s, Decode: {decode_time:.3f}s)")
    print(f"   Hit rates - Prefill: {prefill_hit_rate:.1%}, Decode: {decode_hit_rate:.1%}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return {
        'config': config_name,
        'batch_size': batch_size,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'total_time': total_time,
        'prefill_hit_rate': prefill_hit_rate,
        'decode_hit_rate': decode_hit_rate,
        'tokens_per_second': (20 * batch_size) / decode_time if decode_time > 0 else 0,
        'throughput': batch_size / total_time if total_time > 0 else 0
    }


def plot_results(results, output_dir):
    """Create comprehensive visualizations of the batch size sweep."""

    # Organize data by configuration
    configs = {}
    for r in results:
        config = r['config']
        if config not in configs:
            configs[config] = []
        configs[config].append(r)

    # Sort each config by batch size
    for config in configs:
        configs[config] = sorted(configs[config], key=lambda x: x['batch_size'])

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Color scheme for each configuration
    colors = {
        'Baseline': '#1f77b4',
        'Prefetch': '#ff7f0e',
        'Fiddler': '#2ca02c',
        'Fiddler+Prefetch': '#d62728'
    }

    # Plot 1: Total Time vs Batch Size
    ax1 = fig.add_subplot(gs[0, 0])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        total_times = [d['total_time'] for d in data]
        ax1.plot(batch_sizes, total_times, 'o-', label=config_name,
                color=colors.get(config_name, 'gray'), linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Total Time (s)', fontsize=11, fontweight='bold')
    ax1.set_title('Total Time vs Batch Size', fontsize=12, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Throughput (batches/sec) vs Batch Size
    ax2 = fig.add_subplot(gs[0, 1])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        throughput = [d['throughput'] for d in data]
        ax2.plot(batch_sizes, throughput, 'o-', label=config_name,
                color=colors.get(config_name, 'gray'), linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Throughput (batches/s)', fontsize=11, fontweight='bold')
    ax2.set_title('Throughput vs Batch Size', fontsize=12, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Tokens per Second vs Batch Size
    ax3 = fig.add_subplot(gs[0, 2])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        tokens_per_sec = [d['tokens_per_second'] for d in data]
        ax3.plot(batch_sizes, tokens_per_sec, 'o-', label=config_name,
                color=colors.get(config_name, 'gray'), linewidth=2, markersize=8)
    ax3.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Tokens/Second', fontsize=11, fontweight='bold')
    ax3.set_title('Token Generation Throughput vs Batch Size', fontsize=12, fontweight='bold')
    ax3.set_xscale('log', base=2)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: Speedup vs Baseline (by batch size)
    ax4 = fig.add_subplot(gs[1, 0])
    baseline_data = configs.get('Baseline', [])
    baseline_by_batch = {d['batch_size']: d['total_time'] for d in baseline_data}

    for config_name, data in configs.items():
        if config_name == 'Baseline':
            continue
        batch_sizes = [d['batch_size'] for d in data]
        speedups = [baseline_by_batch.get(d['batch_size'], 1.0) / d['total_time']
                   if d['batch_size'] in baseline_by_batch else 1.0 for d in data]
        ax4.plot(batch_sizes, speedups, 'o-', label=config_name,
                color=colors.get(config_name, 'gray'), linewidth=2, markersize=8)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0x)')
    ax4.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Speedup vs Baseline', fontsize=11, fontweight='bold')
    ax4.set_title('Speedup vs Baseline by Batch Size', fontsize=12, fontweight='bold')
    ax4.set_xscale('log', base=2)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Plot 5: Fiddler vs Fiddler+Prefetch Direct Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    fiddler_data = configs.get('Fiddler', [])
    fiddler_prefetch_data = configs.get('Fiddler+Prefetch', [])

    if fiddler_data and fiddler_prefetch_data:
        fiddler_batch_sizes = [d['batch_size'] for d in fiddler_data]
        fiddler_times = [d['total_time'] for d in fiddler_data]

        fp_batch_sizes = [d['batch_size'] for d in fiddler_prefetch_data]
        fp_times = [d['total_time'] for d in fiddler_prefetch_data]

        ax5.plot(fiddler_batch_sizes, fiddler_times, 'o-', label='Fiddler',
                color=colors['Fiddler'], linewidth=2, markersize=8)
        ax5.plot(fp_batch_sizes, fp_times, 's-', label='Fiddler+Prefetch',
                color=colors['Fiddler+Prefetch'], linewidth=2, markersize=8)

        # Mark crossover point if it exists
        fiddler_by_batch = {d['batch_size']: d['total_time'] for d in fiddler_data}
        fp_by_batch = {d['batch_size']: d['total_time'] for d in fiddler_prefetch_data}

        for bs in sorted(set(fiddler_batch_sizes) & set(fp_batch_sizes)):
            if fp_by_batch[bs] < fiddler_by_batch[bs]:
                ax5.axvline(x=bs, color='green', linestyle=':', alpha=0.5)
                ax5.annotate(f'F+P wins\nat BS={bs}',
                           xy=(bs, min(fp_by_batch[bs], fiddler_by_batch[bs])),
                           xytext=(10, -20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='green'))
                break

    ax5.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Total Time (s)', fontsize=11, fontweight='bold')
    ax5.set_title('🎯 Fiddler vs Fiddler+Prefetch', fontsize=12, fontweight='bold')
    ax5.set_xscale('log', base=2)
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # Plot 6: Relative Performance (Fiddler+Prefetch / Fiddler)
    ax6 = fig.add_subplot(gs[1, 2])
    if fiddler_data and fiddler_prefetch_data:
        common_batch_sizes = sorted(set(fiddler_batch_sizes) & set(fp_batch_sizes))
        relative_perf = []

        for bs in common_batch_sizes:
            fiddler_time = fiddler_by_batch[bs]
            fp_time = fp_by_batch[bs]
            # Speedup of Fiddler+Prefetch relative to Fiddler
            # > 1.0 means Fiddler+Prefetch is faster
            speedup = fiddler_time / fp_time
            relative_perf.append(speedup)

        bars = ax6.bar(range(len(common_batch_sizes)), relative_perf,
                       color=['green' if x > 1.0 else 'red' for x in relative_perf],
                       alpha=0.7, edgecolor='black', linewidth=1.5)
        ax6.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Equal Performance')
        ax6.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Speedup (Fiddler+Prefetch / Fiddler)', fontsize=11, fontweight='bold')
        ax6.set_title('🏆 Fiddler+Prefetch Speedup vs Fiddler', fontsize=12, fontweight='bold')
        ax6.set_xticks(range(len(common_batch_sizes)))
        ax6.set_xticklabels(common_batch_sizes)
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.legend()

        # Annotate bars with values
        for i, (bs, val) in enumerate(zip(common_batch_sizes, relative_perf)):
            ax6.text(i, val + 0.02, f'{val:.2f}x', ha='center', va='bottom', fontweight='bold')

    # Plot 7: Prefill Time Breakdown
    ax7 = fig.add_subplot(gs[2, 0])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        prefill_times = [d['prefill_time'] for d in data]
        ax7.plot(batch_sizes, prefill_times, 'o-', label=config_name,
                color=colors.get(config_name, 'gray'), linewidth=2, markersize=8)
    ax7.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Prefill Time (s)', fontsize=11, fontweight='bold')
    ax7.set_title('Prefill Phase Time vs Batch Size', fontsize=12, fontweight='bold')
    ax7.set_xscale('log', base=2)
    ax7.grid(True, alpha=0.3)
    ax7.legend()

    # Plot 8: Decode Time Breakdown
    ax8 = fig.add_subplot(gs[2, 1])
    for config_name, data in configs.items():
        batch_sizes = [d['batch_size'] for d in data]
        decode_times = [d['decode_time'] for d in data]
        ax8.plot(batch_sizes, decode_times, 'o-', label=config_name,
                color=colors.get(config_name, 'gray'), linewidth=2, markersize=8)
    ax8.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Decode Time (s)', fontsize=11, fontweight='bold')
    ax8.set_title('Decode Phase Time vs Batch Size', fontsize=12, fontweight='bold')
    ax8.set_xscale('log', base=2)
    ax8.grid(True, alpha=0.3)
    ax8.legend()

    # Plot 9: Hit Rates
    ax9 = fig.add_subplot(gs[2, 2])
    for config_name, data in configs.items():
        if config_name == 'Baseline':
            continue
        batch_sizes = [d['batch_size'] for d in data]
        decode_hit_rates = [d['decode_hit_rate'] * 100 for d in data]
        ax9.plot(batch_sizes, decode_hit_rates, 'o-', label=f"{config_name} (Decode)",
                color=colors.get(config_name, 'gray'), linewidth=2, markersize=8)
    ax9.set_xlabel('Batch Size', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Decode Hit Rate (%)', fontsize=11, fontweight='bold')
    ax9.set_title('Decode Phase Hit Rate vs Batch Size', fontsize=12, fontweight='bold')
    ax9.set_xscale('log', base=2)
    ax9.set_ylim([0, 105])
    ax9.grid(True, alpha=0.3)
    ax9.legend()

    # Add overall title
    fig.suptitle('Batch Size Sweep: Finding Where Fiddler+Prefetch Wins',
                fontsize=16, fontweight='bold', y=0.995)

    # Save plot
    plot_path = os.path.join(output_dir, 'batch_size_sweep_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comprehensive plot saved to: {plot_path}")
    plt.close()


def main():
    """Main benchmark function."""
    print("\n" + "="*80)
    print("BATCH SIZE SWEEP BENCHMARK")
    print("="*80)
    print("Goal: Find batch sizes where Fiddler+Prefetch > Fiddler")
    print("="*80)
    print("\nConfigurations to test:")
    print("  1. Baseline (no prefetch, no CPU offload)")
    print("  2. Prefetch (8 experts, no CPU offload)")
    print("  3. Fiddler (no prefetch, CPU offload)")
    print("  4. Fiddler+Prefetch (8 experts, CPU offload)")
    print("\nBatch sizes: 1, 2, 4, 8, 16, 32")
    print("="*80)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"batch_size_sweep_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Delete pattern file to ensure clean collection
    pattern_file = "expert_usage_patterns_qwen.json"
    if os.path.exists(pattern_file):
        os.remove(pattern_file)
        print(f"🗑️  Deleted existing pattern file: {pattern_file}")

    # Define configurations to test
    configurations = [
        {"name": "Baseline", "num_experts": 0, "cpu_offload": False},
        {"name": "Prefetch", "num_experts": 8, "cpu_offload": False},
        {"name": "Fiddler", "num_experts": 0, "cpu_offload": True},
        {"name": "Fiddler+Prefetch", "num_experts": 8, "cpu_offload": True},
    ]

    # Batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32]

    # Run all benchmarks
    all_results = []

    for config in configurations:
        for batch_size in batch_sizes:
            try:
                result = run_config(
                    config['name'],
                    batch_size,
                    num_experts_to_prefetch=config['num_experts'],
                    enable_cpu_offload=config['cpu_offload']
                )
                all_results.append(result)

                print(f"\n📈 {config['name']} @ BS={batch_size}: "
                      f"{result['total_time']:.3f}s "
                      f"({result['tokens_per_second']:.1f} tokens/s)")

            except Exception as e:
                print(f"\n❌ Error running {config['name']} @ BS={batch_size}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save results to CSV
    csv_path = os.path.join(output_dir, 'batch_size_results.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['config', 'batch_size', 'prefill_time', 'decode_time', 'total_time',
                     'prefill_hit_rate', 'decode_hit_rate', 'tokens_per_second', 'throughput']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n💾 Results saved to: {csv_path}")

    # Save summary JSON
    summary = {
        'timestamp': timestamp,
        'batch_sizes': batch_sizes,
        'configurations': [c['name'] for c in configurations],
        'results': all_results
    }

    json_path = os.path.join(output_dir, 'batch_size_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"💾 Summary saved to: {json_path}")

    # Generate visualizations
    if all_results:
        plot_results(all_results, output_dir)

    # Print analysis
    print("\n" + "="*80)
    print("ANALYSIS: Where does Fiddler+Prefetch win?")
    print("="*80)

    # Group by batch size and compare Fiddler vs Fiddler+Prefetch
    fiddler_results = {r['batch_size']: r for r in all_results if r['config'] == 'Fiddler'}
    fp_results = {r['batch_size']: r for r in all_results if r['config'] == 'Fiddler+Prefetch'}

    print(f"\n{'Batch Size':<12} {'Fiddler':<12} {'F+Prefetch':<12} {'Speedup':<12} {'Winner':<12}")
    print("-" * 60)

    for bs in sorted(set(fiddler_results.keys()) & set(fp_results.keys())):
        fiddler_time = fiddler_results[bs]['total_time']
        fp_time = fp_results[bs]['total_time']
        speedup = fiddler_time / fp_time
        winner = "🏆 F+Prefetch" if fp_time < fiddler_time else "Fiddler"

        print(f"{bs:<12} {fiddler_time:<12.3f} {fp_time:<12.3f} {speedup:<12.3f} {winner:<12}")

    print("="*80)


if __name__ == "__main__":
    main()
