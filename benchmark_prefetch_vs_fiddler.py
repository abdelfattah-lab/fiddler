#!/usr/bin/env python3
"""
Comprehensive benchmark comparing different prefetch configurations vs Fiddler baseline.
Tests across multiple batch sizes with CPU offloading enabled (use_fiddler_mode=True).

This benchmark compares:
- Fiddler baseline (use_fiddler_mode=True)
- Prefetch with 0-16 experts (use_fiddler_mode=True)

Both will automatically use CPU execution for small batches (< fiddler_batch_threshold=8)
and GPU execution for larger batches, allowing us to see behavior in both modes.

Batch sizes: 1, 2, 4, 8, 16, 32, 64
"""

import os
import sys
import argparse
import torch
import time
import json
import csv
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen import FiddlerQwen
from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch


def generate_diverse_prompts(num_prompts):
    """Generate diverse prompts to ensure varied expert activation patterns."""
    # Categories designed to trigger different expert specializations
    categories = [
        # Factual/Knowledge
        "What is the capital of {}?",
        "Explain the concept of {} in simple terms.",
        "Who invented the {}?",
        "When did {} occur?",

        # Creative/Language
        "Write a poem about {}.",
        "Create a story involving {}.",
        "Describe {} in a poetic way.",
        "Compose a haiku about {}.",

        # Technical/Logical
        "How does {} work?",
        "What are the steps to {}?",
        "Explain the algorithm for {}.",
        "Analyze the implementation of {}.",

        # Mathematical
        "What is the formula for {}?",
        "Solve this equation: {}",
        "Find the derivative of {}.",
        "Compute the value of {}.",

        # Conversational
        "Tell me about your thoughts on {}.",
        "What do you think about {}?",
        "Can you help me understand {}?",
        "I'm curious about {}.",
    ]

    topics = [
        "France", "gravity", "telephone", "World War II",
        "sunset", "courage", "time", "silence",
        "quantum computing", "neural networks", "encryption", "sorting",
        "calculus", "geometry", "probability", "statistics",
        "artificial intelligence", "climate change", "evolution", "relativity",
        "democracy", "economics", "psychology", "philosophy",
        "music theory", "literature", "art history", "archaeology",
        "medicine", "biology", "chemistry", "physics",
        "computer science", "mathematics", "engineering", "astronomy",
        "language learning", "cooking", "gardening", "photography",
        "sports", "travel", "history", "culture",
        "technology", "science", "education", "health",
        "environment", "society", "politics", "law",
        "business", "finance", "marketing", "management",
        "blockchain", "machine learning", "data science", "robotics",
        "renewable energy", "space exploration", "genetics", "neuroscience",
        "ancient civilizations", "mythology", "religions", "ethics",
        "quantum physics", "string theory", "dark matter", "black holes",
    ]

    prompts = []
    for i in range(num_prompts):
        category = categories[i % len(categories)]
        topic = topics[i % len(topics)]
        prompts.append(category.format(topic))

    return prompts


def benchmark_configuration(model_class, args, config_name, batch_size, prompts, warmup_runs=1, test_runs=3):
    """Benchmark a specific configuration."""
    print(f"\n  Testing {config_name} with batch_size={batch_size}...")

    # Create model instance
    model = model_class(args)

    # Select prompts for this batch size
    test_prompts = prompts[:batch_size]

    # Warmup runs
    for _ in range(warmup_runs):
        try:
            _ = model.generate(test_prompts, output_token=20)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"    ⚠️  Warmup failed: {e}")
            del model
            torch.cuda.empty_cache()
            return None

    # Timed runs
    times = []
    for run in range(test_runs):
        try:
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            output = model.generate(test_prompts, output_token=20)

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"    Run {run + 1}/{test_runs}: {elapsed:.3f}s")

        except Exception as e:
            print(f"    ⚠️  Run {run + 1} failed: {e}")
            del model
            torch.cuda.empty_cache()
            return None

    # Get statistics if available
    stats = {}
    if hasattr(model, 'cnt_expert_hit'):
        total = model.cnt_expert_all if hasattr(model, 'cnt_expert_all') else 0
        hits = model.cnt_expert_hit
        hit_rate = (hits / total * 100) if total > 0 else 0
        stats['hit_rate'] = hit_rate
        stats['total_experts_accessed'] = total
        stats['hits'] = hits

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Return median time and stats
    median_time = sorted(times)[len(times) // 2]
    return {
        'median_time': median_time,
        'all_times': times,
        'stats': stats
    }


def run_comprehensive_benchmark():
    """Run comprehensive benchmark across all configurations and batch sizes."""
    print("=" * 80)
    print("COMPREHENSIVE PREFETCH vs FIDDLER BENCHMARK")
    print("=" * 80)

    # Configuration
    model_name = "Qwen/Qwen1.5-MoE-A2.7B"
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    prefetch_configs = list(range(0, 17))  # 0 to 16 experts

    # Generate diverse prompts (use max batch size worth of prompts)
    max_batch = max(batch_sizes)
    prompts = generate_diverse_prompts(max_batch)

    print(f"\nModel: {model_name}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Prefetch configs: 0-16 experts")
    print(f"Generated {len(prompts)} diverse prompts")
    print(f"Mode: use_fiddler_mode=True (CPU for batch<8, GPU for batch>=8)")

    # Results storage
    results = []

    # Store baseline times for all batch sizes
    baseline_times = {}

    # First, test baseline for all batch sizes (only load model once)
    print(f"\n{'='*80}")
    print(f"TESTING BASELINE (All batch sizes)")
    print(f"{'='*80}")

    baseline_args = argparse.Namespace(
        model=model_name,
        beam_width=1,
        cpu_offload=1,
        max_experts_gpu=0,
        use_fiddler_mode=True,
    )

    print("\nLoading baseline model...")
    baseline_model = FiddlerQwen(baseline_args)

    for batch_size in batch_sizes:
        execution_mode = "CPU" if batch_size < 8 else "GPU"
        print(f"\n  Batch {batch_size} ({execution_mode})...")

        test_prompts = prompts[:batch_size]

        # Run benchmark
        times = []
        for run in range(3):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = baseline_model.generate(test_prompts, output_token=20)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
            print(f"    Run {run+1}/3: {times[-1]:.3f}s")

        median_time = sorted(times)[len(times) // 2]
        baseline_times[batch_size] = median_time

        # Get hit rate
        hit_rate = (baseline_model.cnt_expert_hit / baseline_model.cnt_expert_all * 100) if baseline_model.cnt_expert_all > 0 else 0

        result_entry = {
            'batch_size': batch_size,
            'config': 'Baseline',
            'num_experts_prefetch': 0,
            'median_time': median_time,
            'baseline_time': median_time,
            'speedup_vs_baseline': 1.0,
            'hit_rate': hit_rate,
            'execution_mode': execution_mode
        }
        results.append(result_entry)
        print(f"    ✓ Median: {median_time:.3f}s, Hit rate: {hit_rate:.1f}%")

    del baseline_model
    torch.cuda.empty_cache()
    print("\n✅ Baseline complete for all batch sizes")

    # Now test each prefetch configuration for all batch sizes
    for config_idx, num_experts in enumerate(prefetch_configs, start=1):
        print(f"\n{'='*80}")
        print(f"TESTING PREFETCH CONFIG [{config_idx}/{len(prefetch_configs)}]: {num_experts} experts")
        print(f"{'='*80}")

        prefetch_args = argparse.Namespace(
            model=model_name,
            beam_width=1,
            cpu_offload=1,
            max_experts_gpu=0,
            use_fiddler_mode=True,
        )

        print(f"\nLoading model with {num_experts} experts to prefetch...")
        prefetch_model = FiddlerQwenWithPrefetch(prefetch_args, num_experts_to_prefetch=num_experts)

        for batch_size in batch_sizes:
            execution_mode = "CPU" if batch_size < 8 else "GPU"
            print(f"\n  Batch {batch_size} ({execution_mode})...")

            test_prompts = prompts[:batch_size]

            # Run benchmark
            times = []
            for run in range(3):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = prefetch_model.generate(test_prompts, output_token=20)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                print(f"    Run {run+1}/3: {times[-1]:.3f}s")

            median_time = sorted(times)[len(times) // 2]
            baseline_time = baseline_times[batch_size]
            speedup = baseline_time / median_time if median_time > 0 else 0

            # Get hit rate
            hit_rate = (prefetch_model.cnt_expert_hit / prefetch_model.cnt_expert_all * 100) if prefetch_model.cnt_expert_all > 0 else 0

            result_entry = {
                'batch_size': batch_size,
                'config': f'Prefetch-{num_experts}',
                'num_experts_prefetch': num_experts,
                'median_time': median_time,
                'baseline_time': baseline_time,
                'speedup_vs_baseline': speedup,
                'hit_rate': hit_rate,
                'execution_mode': execution_mode
            }
            results.append(result_entry)
            print(f"    ✓ Median: {median_time:.3f}s, Speedup: {speedup:.2f}x, Hit rate: {hit_rate:.1f}%")

        del prefetch_model
        torch.cuda.empty_cache()
        print(f"\n✅ Prefetch-{num_experts} complete for all batch sizes")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"prefetch_vs_fiddler_benchmark_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✅ Results saved to: {csv_path}")

    # Save to JSON for easier processing
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {json_path}")

    # Create summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("PREFETCH vs FIDDLER BENCHMARK SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Batch sizes tested: {batch_sizes}\n")
        f.write(f"Prefetch configs tested: {prefetch_configs}\n")
        f.write(f"Total tests: {len(results)}\n")
        f.write(f"\nMode: use_fiddler_mode=True\n")
        f.write(f"  - Batch < 8: CPU execution\n")
        f.write(f"  - Batch >= 8: GPU execution\n")
        f.write("\n" + "=" * 80 + "\n")

        # Best configuration for each batch size
        f.write("\nBest Configuration for Each Batch Size:\n")
        f.write("-" * 80 + "\n")
        for bs in batch_sizes:
            bs_results = [r for r in results if r['batch_size'] == bs and r['config'] != 'Baseline']
            if bs_results:
                best = max(bs_results, key=lambda x: x['speedup_vs_baseline'] or 0)
                exec_mode = "CPU" if bs < 8 else "GPU"
                f.write(f"\nBatch {bs:2d} ({exec_mode}): {best['config']:15s} - "
                       f"{best['speedup_vs_baseline']:.3f}x speedup - "
                       f"{best['median_time']:.3f}s - "
                       f"Hit rate: {best['hit_rate']:.1f}%\n")

    print(f"✅ Summary saved to: {summary_path}")
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results in: {output_dir}/")
    print(f"{'='*80}")

    return output_dir


if __name__ == "__main__":
    output_dir = run_comprehensive_benchmark()
