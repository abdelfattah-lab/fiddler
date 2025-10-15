#!/usr/bin/env python3
"""Profile Fiddler-only at batch size 8."""

import sys
sys.path.insert(0, 'src')

import torch
from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch


class Args:
    def __init__(self):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1


def get_test_prompts(batch_size):
    """Get diverse test prompts."""
    all_prompts = [
        "The capital of France is",
        "The theory of relativity was",
        "In quantum mechanics, the",
        "The human brain contains",
        "Climate change affects the",
        "Machine learning algorithms can",
        "The history of civilization began",
        "Photosynthesis is the process by which",
    ]
    return all_prompts[:batch_size]


def main():
    print("="*80)
    print("PROFILING: Fiddler-only (Batch Size = 8)")
    print("="*80)

    args = Args()
    batch_size = 8

    # Create model: Fiddler only (no prefetch)
    print("Loading model: Fiddler-only...")
    model = FiddlerQwenWithPrefetch(
        args,
        num_experts_to_prefetch=0,  # No prefetch
        enable_cpu_offload=True     # Fiddler enabled
    )

    prompts = get_test_prompts(batch_size)
    print(f"Testing with {batch_size} prompts")

    # Warm-up run
    print("Warm-up run...")
    _ = model.generate(prompts, output_token=20)

    # Profiled run
    print("Starting profiled run...")
    torch.cuda.nvtx.range_push("Fiddler_BS8_Generation")

    prefill_time, decode_time, prefill_hr, decode_hr = model.generate(prompts, output_token=20)

    torch.cuda.nvtx.range_pop()

    # Print results
    total_time = prefill_time + decode_time
    print(f"\nResults:")
    print(f"  Prefill time: {prefill_time:.3f}s (Hit rate: {prefill_hr*100:.1f}%)")
    print(f"  Decode time: {decode_time:.3f}s (Hit rate: {decode_hr*100:.1f}%)")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Throughput: {batch_size * 20 / total_time:.1f} tok/s")

    print("\n✅ Profiling complete")


if __name__ == '__main__':
    main()
