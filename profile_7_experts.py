#!/usr/bin/env python3
"""
Profile the 7-expert configuration with Nsight Systems.
Based on the benchmark results, 7 experts gave the best speedup (4.285x).
"""

import os
import sys
import subprocess
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen_with_prefetch import FiddlerQwenWithPrefetch


class Args:
    """Arguments for Qwen model."""
    def __init__(self):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1


def run_profiled_inference():
    """Run inference with 7 experts (best configuration)."""
    print("\n" + "="*80)
    print("Running QWEN with 7 experts (BEST CONFIG from benchmark)")
    print("="*80)

    args = Args()
    model = FiddlerQwenWithPrefetch(args, num_experts_to_prefetch=7)

    # Warmup run (not profiled)
    print("\nWarmup run (not profiled)...")
    model.generate("The capital of France is", output_token=10)

    # Actual run (will be profiled by nsys wrapper)
    print("\nProfiled run...")
    prefill_time, decode_time, hit_rate = model.generate(
        "The capital of France is",
        output_token=20
    )

    print(f"\n✅ Profiling completed")
    print(f"🎯 Hit rate: {hit_rate:.1%}")
    print(f"⏱️  Total time: {prefill_time + decode_time:.3f}s")


if __name__ == "__main__":
    run_profiled_inference()
