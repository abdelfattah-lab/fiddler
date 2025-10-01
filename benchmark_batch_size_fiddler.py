#!/usr/bin/env python3
"""
Benchmark Fiddler Mode vs GPU Prefetch across different batch sizes

Compares:
1. FiddlerQwen baseline with CPU fallback (Fiddler mode)
2. FiddlerQwenWithPrefetch (7-expert optimal config) with CPU fallback

Tests batch sizes: 1, 2, 4, 8, 16, 32, 64
Uses diverse prompts to ensure different experts are activated
"""

import os
import sys
import time
import torch
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen import FiddlerQwen
from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch


# Diverse prompts that should activate different experts
DIVERSE_PROMPTS = [
    "The capital of France is",
    "Write a Python function to sort a list:",
    "Explain quantum mechanics in simple terms:",
    "What is the meaning of life?",
    "Translate 'hello' to Spanish:",
    "Calculate 15 * 23 =",
    "Who invented the telephone?",
    "Describe the water cycle:",
    "Write a haiku about mountains:",
    "What causes seasons on Earth?",
    "How do airplanes fly?",
    "Explain photosynthesis:",
    "What is artificial intelligence?",
    "Name three planets in our solar system:",
    "How does a computer work?",
    "What is democracy?",
    "Describe the process of making bread:",
    "What is the speed of light?",
    "How do vaccines work?",
    "What is DNA?",
    "Explain gravity:",
    "What causes rain?",
    "How do magnets work?",
    "What is electricity?",
    "Describe the human heart:",
    "What is evolution?",
    "How does the internet work?",
    "What is climate change?",
    "Explain the concept of time:",
    "What is a black hole?",
    "How do batteries work?",
    "What is sound?",
    "Describe a rainbow:",
    "What is renewable energy?",
    "How do plants grow?",
    "What is the Big Bang theory?",
    "Explain cellular respiration:",
    "What is a volcano?",
    "How do eyes see?",
    "What is the greenhouse effect?",
    "Describe an ecosystem:",
    "What is atomic structure?",
    "How do rockets work?",
    "What is plate tectonics?",
    "Explain the food chain:",
    "What is nuclear fusion?",
    "How do antibiotics work?",
    "What is osmosis?",
    "Describe the carbon cycle:",
    "What is entropy?",
    "How do satellites orbit?",
    "What is gene expression?",
    "Explain chemical bonding:",
    "What is a neural network?",
    "How does GPS work?",
    "What is natural selection?",
    "Describe the nitrogen cycle:",
    "What is a semiconductor?",
    "How do enzymes work?",
    "What is mitosis?",
    "Explain electromagnetic waves:",
    "What is protein synthesis?",
    "How do holograms work?",
    "What is thermodynamics?",
]


class Args:
    def __init__(self, use_fiddler_mode=False, fiddler_batch_threshold=8):
        self.model = 'Qwen/Qwen1.5-MoE-A2.7B'
        self.beam_width = 1
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.use_fiddler_mode = use_fiddler_mode
        self.fiddler_batch_threshold = fiddler_batch_threshold


def benchmark_batch_size(model, batch_size, output_tokens=3, warmup=1, iterations=3):
    """
    Benchmark a model with a specific batch size.

    Args:
        model: The model to benchmark
        batch_size: Number of prompts to process in parallel
        output_tokens: Number of tokens to generate
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations

    Returns:
        dict with timing statistics
    """
    # Select diverse prompts for this batch
    prompts = DIVERSE_PROMPTS[:batch_size]

    # Warmup
    for _ in range(warmup):
        try:
            # Tokenize batch
            inputs = model.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)

            with torch.no_grad():
                model.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=output_tokens,
                    do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id,
                    use_cache=True
                )

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  Warmup failed for batch_size={batch_size}: {e}")
            return None

    # Benchmark iterations
    times = []
    for _ in range(iterations):
        try:
            # Tokenize batch
            inputs = model.tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)

            # Reset statistics
            if hasattr(model, 'cnt_expert_hit'):
                model.cnt_expert_hit = 0
                model.cnt_expert_all = 0
            if hasattr(model, 'metrics'):
                model.metrics = type(model.metrics)()

            start_time = time.time()
            with torch.no_grad():
                outputs = model.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=output_tokens,
                    do_sample=False,
                    pad_token_id=model.tokenizer.eos_token_id,
                    use_cache=True
                )
            elapsed = time.time() - start_time
            times.append(elapsed)

            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"⚠️  Benchmark failed for batch_size={batch_size}: {e}")
            return None

    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    # Get hit rate if available
    hit_rate = None
    if hasattr(model, 'metrics') and hasattr(model.metrics, 'get_hit_rate'):
        hit_rate = model.metrics.get_hit_rate()
    elif hasattr(model, 'cnt_expert_all') and model.cnt_expert_all > 0:
        hit_rate = model.cnt_expert_hit / model.cnt_expert_all

    return {
        'batch_size': batch_size,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'hit_rate': hit_rate,
        'times': times
    }


def run_benchmark_suite():
    """Run full benchmark suite comparing baseline and prefetch with Fiddler mode."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"fiddler_batch_benchmark_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("BENCHMARK: Fiddler Mode vs GPU Prefetch across Batch Sizes")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = {
        'baseline': [],
        'baseline_fiddler': [],
        'prefetch': [],
        'prefetch_fiddler': []
    }

    # Test configurations
    configs = [
        ('baseline', 'FiddlerQwen (GPU only)', False, 8),
        ('baseline_fiddler', 'FiddlerQwen (Fiddler mode, threshold=8)', True, 8),
        ('prefetch', 'FiddlerQwenWithPrefetch (GPU, 7 experts)', False, 8),
        ('prefetch_fiddler', 'FiddlerQwenWithPrefetch (Fiddler mode, 7 experts, threshold=8)', True, 8),
    ]

    for config_name, config_desc, use_fiddler, threshold in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {config_desc}")
        print(f"{'='*80}\n")

        # Initialize model
        args = Args(use_fiddler_mode=use_fiddler, fiddler_batch_threshold=threshold)

        if 'prefetch' in config_name:
            print("Loading FiddlerQwenWithPrefetch (7 experts)...")
            model = FiddlerQwenWithPrefetch(args, num_experts_to_prefetch=7)

            # Run pattern collection if needed
            if not os.path.exists('expert_usage_patterns_qwen.json'):
                print("Running pattern collection...")
                model.generate("The capital of France is", output_token=5)
                model.save_expert_patterns()
        else:
            print("Loading FiddlerQwen baseline...")
            model = FiddlerQwen(args)

        print(f"Fiddler mode: {model.use_fiddler_mode}, Threshold: {model.fiddler_batch_threshold}")

        # Test each batch size
        for batch_size in batch_sizes:
            print(f"\nTesting batch_size={batch_size}...", end=" ", flush=True)
            result = benchmark_batch_size(model, batch_size, output_tokens=3, warmup=1, iterations=3)

            if result:
                results[config_name].append(result)
                hit_info = f", hit_rate={result['hit_rate']:.2%}" if result['hit_rate'] is not None else ""
                print(f"✅ {result['avg_time']:.3f}s (min={result['min_time']:.3f}s, max={result['max_time']:.3f}s{hit_info})")
            else:
                print("❌ Failed")

        # Clean up
        del model
        torch.cuda.empty_cache()
        print(f"\n{'='*80}\n")

    # Save results
    results_file = os.path.join(output_dir, 'batch_size_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")

    # Generate summary
    summary_file = os.path.join(output_dir, 'summary.txt')
    with open(summary_file, 'w') as f:
        f.write("Batch Size Benchmark Summary\n")
        f.write("="*80 + "\n\n")

        for batch_size in batch_sizes:
            f.write(f"\nBatch Size: {batch_size}\n")
            f.write("-" * 40 + "\n")

            # Find results for this batch size
            baseline = next((r for r in results['baseline'] if r['batch_size'] == batch_size), None)
            baseline_fiddler = next((r for r in results['baseline_fiddler'] if r['batch_size'] == batch_size), None)
            prefetch = next((r for r in results['prefetch'] if r['batch_size'] == batch_size), None)
            prefetch_fiddler = next((r for r in results['prefetch_fiddler'] if r['batch_size'] == batch_size), None)

            if baseline:
                f.write(f"Baseline (GPU):             {baseline['avg_time']:.3f}s\n")
            if baseline_fiddler:
                f.write(f"Baseline (Fiddler):         {baseline_fiddler['avg_time']:.3f}s")
                if baseline:
                    speedup = baseline['avg_time'] / baseline_fiddler['avg_time']
                    f.write(f"  (speedup: {speedup:.2f}x)\n")
                else:
                    f.write("\n")
            if prefetch:
                f.write(f"Prefetch (GPU, 7 experts):  {prefetch['avg_time']:.3f}s")
                if baseline:
                    speedup = baseline['avg_time'] / prefetch['avg_time']
                    f.write(f"  (speedup: {speedup:.2f}x)\n")
                else:
                    f.write("\n")
            if prefetch_fiddler:
                f.write(f"Prefetch (Fiddler):         {prefetch_fiddler['avg_time']:.3f}s")
                if baseline:
                    speedup = baseline['avg_time'] / prefetch_fiddler['avg_time']
                    f.write(f"  (speedup: {speedup:.2f}x)\n")
                else:
                    f.write("\n")

    print(f"✅ Summary saved to {summary_file}")
    print(f"\n{'='*80}")
    print(f"Benchmark complete! Results in: {output_dir}")
    print(f"{'='*80}")

    return output_dir


if __name__ == "__main__":
    output_dir = run_benchmark_suite()
    print(f"\nNext step: Create plots from results in {output_dir}")
