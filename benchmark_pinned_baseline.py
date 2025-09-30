#!/usr/bin/env python3
"""
Benchmark script to compare baseline performance with and without pinned memory.
This will help quantify the speedup from pinned memory alone.
"""

import os
import sys
import time
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen import FiddlerQwen


class Args:
    """Arguments for Qwen model."""
    def __init__(self):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1


def run_baseline_with_pinned():
    """Run baseline model with pinned memory (current version)."""
    print("\n" + "="*80)
    print("Running BASELINE with PINNED MEMORY")
    print("="*80)

    args = Args()
    model = FiddlerQwen(args)

    # Warmup run
    print("Warmup...")
    model.generate("The capital of France is", output_token=10)

    # Actual benchmark run (5 runs for stability)
    print("\nBenchmark runs (5 iterations)...")
    times = []
    for i in range(5):
        print(f"\nRun {i+1}/5...")
        start_time = time.time()
        prefill_time, decode_time, hit_rate = model.generate(
            "The capital of France is",
            output_token=20
        )
        total_time = time.time() - start_time
        times.append(total_time)
        print(f"  Time: {total_time:.3f}s")

    avg_time = sum(times) / len(times)
    print(f"\n✅ Average time with pinned memory: {avg_time:.3f}s")
    print(f"   Times: {[f'{t:.3f}' for t in times]}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return avg_time, times


def main():
    """Run benchmark and save results."""
    print("Starting pinned memory baseline benchmark...")
    print(f"Model: Qwen/Qwen1.5-MoE-A2.7B")
    print(f"Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")

    # Run baseline with pinned memory
    avg_time_pinned, times_pinned = run_baseline_with_pinned()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"pinned_baseline_benchmark_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)

    results = {
        "timestamp": timestamp,
        "model": "Qwen/Qwen1.5-MoE-A2.7B",
        "baseline_with_pinned": {
            "avg_time": avg_time_pinned,
            "times": times_pinned
        }
    }

    # Save JSON results
    json_path = os.path.join(results_dir, "benchmark_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_dir}/")
    print(f"\nBaseline with pinned memory: {avg_time_pinned:.3f}s")
    print(f"\nNote: To measure speedup, compare against the previous baseline")
    print(f"without pinned memory from earlier benchmarks.")


if __name__ == "__main__":
    main()
