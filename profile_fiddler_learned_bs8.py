#!/usr/bin/env python3
"""Profile Fiddler+Learned-Prefetch at batch size 8."""

import sys
sys.path.insert(0, 'src')

import torch
from fiddler.qwen_with_learned_prefetch import FiddlerQwenWithLearnedPrefetch


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
    print("PROFILING: Fiddler+Learned-Prefetch (Batch Size = 8)")
    print("="*80)

    args = Args()
    batch_size = 8

    # Create model: Fiddler + Learned Prefetch
    print("Loading model: Fiddler+Learned-Prefetch...")
    model = FiddlerQwenWithLearnedPrefetch(
        args,
        num_experts_to_prefetch=8,                      # Prefetch 8 experts
        enable_cpu_offload=True,                        # Fiddler enabled
        predictor_path='predictor_checkpoints/best_model.pt'
    )

    prompts = get_test_prompts(batch_size)
    print(f"Testing with {batch_size} prompts")

    # Warm-up run
    print("Warm-up run...")
    _ = model.generate(prompts, output_token=20)

    # Profiled run
    print("Starting profiled run...")
    torch.cuda.nvtx.range_push("Fiddler+Learned_BS8_Generation")

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
