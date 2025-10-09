#!/usr/bin/env python3
"""
Test Correctness of All 4 Fiddler Configurations

This script tests:
1. Baseline: num_experts_to_prefetch=0, enable_cpu_offload=False
2. Prefetch: num_experts_to_prefetch=8, enable_cpu_offload=False
3. Fiddler: num_experts_to_prefetch=0, enable_cpu_offload=True
4. Fiddler+Prefetch: num_experts_to_prefetch=8, enable_cpu_offload=True

All configurations should produce identical outputs.
"""

import torch
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch


def run_config(config_name, num_experts, enable_cpu_offload, latency_cpu, latency_gpu):
    """Run a single configuration and return results."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"  num_experts_to_prefetch={num_experts}, enable_cpu_offload={enable_cpu_offload}")
    if enable_cpu_offload:
        print(f"  latency_cpu={latency_cpu}, latency_gpu={latency_gpu}")
    print(f"{'='*80}")

    # Setup args
    class Args:
        def __init__(self):
            self.model = "Qwen/Qwen1.5-MoE-A2.7B"
            self.dtype = torch.bfloat16
            self.cpu_offload = 0
            self.max_experts_gpu = 0
            self.beam_width = 1

    args = Args()

    # Load model
    model = FiddlerQwenWithPrefetch(
        args,
        num_experts_to_prefetch=num_experts,
        enable_cpu_offload=enable_cpu_offload,
        latency_cpu=latency_cpu,
        latency_gpu=latency_gpu
    )

    # Generate
    test_prompt = "The capital of France is ______.\n"
    print(f"\nPrompt: {test_prompt}")

    prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
        text=test_prompt,
        output_token=20
    )

    # Get generated text
    generated_text = model.last_generated_text

    # Get stats
    prefetch_stats = model.get_prefetch_stats()
    cpu_stats = model.get_cpu_offload_stats() if enable_cpu_offload else None

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {
        'config_name': config_name,
        'generated_text': generated_text,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'total_time': prefill_time + decode_time,
        'prefill_hit_rate': prefill_hit_rate,
        'decode_hit_rate': decode_hit_rate,
        'prefetch_stats': prefetch_stats,
        'cpu_stats': cpu_stats
    }


def main():
    parser = argparse.ArgumentParser(description='Test all Fiddler configurations for correctness')
    parser.add_argument('--latency-cpu', type=float, default=0.05,
                        help='CPU cost per token (ms) - from profiling')
    parser.add_argument('--latency-gpu', type=float, default=5.0,
                        help='GPU transfer cost (ms) - from profiling')
    parser.add_argument('--num-prefetch', type=int, default=8,
                        help='Number of experts to prefetch')
    args = parser.parse_args()

    print("="*80)
    print("TESTING FIDDLER CONFIGURATIONS FOR CORRECTNESS")
    print("="*80)
    print(f"\nCost model parameters:")
    print(f"  latency_cpu = {args.latency_cpu} ms/token")
    print(f"  latency_gpu = {args.latency_gpu} ms")
    print(f"  num_prefetch = {args.num_prefetch} experts")

    # Test all 4 configurations
    configs = [
        {
            'name': 'Baseline',
            'num_experts': 0,
            'enable_cpu_offload': False,
            'latency_cpu': None,
            'latency_gpu': None
        },
        {
            'name': 'Prefetch',
            'num_experts': args.num_prefetch,
            'enable_cpu_offload': False,
            'latency_cpu': None,
            'latency_gpu': None
        },
        {
            'name': 'Fiddler',
            'num_experts': 0,
            'enable_cpu_offload': True,
            'latency_cpu': args.latency_cpu,
            'latency_gpu': args.latency_gpu
        },
        {
            'name': 'Fiddler+Prefetch',
            'num_experts': args.num_prefetch,
            'enable_cpu_offload': True,
            'latency_cpu': args.latency_cpu,
            'latency_gpu': args.latency_gpu
        }
    ]

    results = []
    for config in configs:
        result = run_config(
            config['name'],
            config['num_experts'],
            config['enable_cpu_offload'],
            config['latency_cpu'],
            config['latency_gpu']
        )
        results.append(result)

    # Verify correctness
    print("\n" + "="*80)
    print("CORRECTNESS VERIFICATION")
    print("="*80)

    baseline_text = results[0]['generated_text']
    all_match = True

    for i, result in enumerate(results):
        config_name = result['config_name']
        generated_text = result['generated_text']

        matches = generated_text == baseline_text
        all_match = all_match and matches

        status = "✅ MATCH" if matches else "❌ MISMATCH"
        print(f"\n{config_name}: {status}")
        print(f"  Text: {generated_text[:100]}...")

        if not matches:
            print(f"  Expected: {baseline_text[:100]}...")
            print(f"  Got:      {generated_text[:100]}...")

    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<20} {'Total (s)':<12} {'Prefill (s)':<12} {'Decode (s)':<12} {'Speedup':<10}")
    print("-" * 80)

    baseline_total = results[0]['total_time']

    for result in results:
        speedup = baseline_total / result['total_time']
        print(f"{result['config_name']:<20} "
              f"{result['total_time']:<12.3f} "
              f"{result['prefill_time']:<12.3f} "
              f"{result['decode_time']:<12.3f} "
              f"{speedup:<10.3f}x")

    # Hit rate summary
    print("\n" + "="*80)
    print("HIT RATE SUMMARY")
    print("="*80)

    print(f"\n{'Configuration':<20} {'Prefill Hit %':<15} {'Decode Hit %':<15}")
    print("-" * 80)

    for result in results:
        print(f"{result['config_name']:<20} "
              f"{result['prefill_hit_rate']*100:<15.1f} "
              f"{result['decode_hit_rate']*100:<15.1f}")

    # CPU offloading summary
    print("\n" + "="*80)
    print("CPU OFFLOADING SUMMARY")
    print("="*80)

    for result in results:
        if result['cpu_stats'] and result['cpu_stats']['enabled']:
            stats = result['cpu_stats']
            print(f"\n{result['config_name']}:")
            print(f"  CPU experts: {stats['cpu_expert_count']} ({stats['cpu_percentage']:.1f}%)")
            print(f"  GPU experts: {stats['gpu_expert_count']} ({stats['gpu_percentage']:.1f}%)")
            print(f"  CPU exec time: {stats['cpu_execution_time']:.3f}s")
            print(f"  GPU exec time: {stats['gpu_execution_time']:.3f}s")

    # Final result
    print("\n" + "="*80)
    if all_match:
        print("✅ SUCCESS: All configurations produce identical outputs!")
    else:
        print("❌ FAILURE: Configurations produce different outputs!")
    print("="*80)

    return 0 if all_match else 1


if __name__ == '__main__':
    sys.exit(main())
