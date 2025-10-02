#!/usr/bin/env python3
"""Quick test for all-GPU mode correctness."""

import os
import sys
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch


def main():
    """Test all-GPU mode."""
    args = argparse.Namespace(
        model='Qwen/Qwen1.5-MoE-A2.7B',
        dtype='bfloat16',
        device='cuda',
        cpu_offload=0,
        max_experts_gpu=0,
        beam_width=1,
        use_fiddler_mode=False,
        fiddler_batch_threshold=8
    )

    print("Testing all-GPU mode...")
    model = FiddlerQwenWithPrefetch(args, all_gpu_mode=True)

    print("\nGenerating text...")
    result = model.generate(text="The capital of France is", output_token=20)

    print(f"\nGeneration completed successfully!")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
