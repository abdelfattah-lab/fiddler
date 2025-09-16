#!/usr/bin/env python3
"""
Profile both baseline FiddlerMixtral and FiddlerMixtralWithPrefetch implementations
using a controlled test to identify optimization opportunities.
"""

import torch
import sys
import os
import time
from argparse import Namespace

# Add src to path
sys.path.insert(0, 'src')

from fiddler.mixtral import FiddlerMixtral
from fiddler.mixtral_with_prefetch import FiddlerMixtralWithPrefetch

# Force CUDA initialization
torch.cuda.init()
torch.cuda.set_device(0)
test_tensor = torch.tensor([1.0]).cuda()
print(f"✅ CUDA initialized successfully on device: {torch.cuda.get_device_name(0)}")
del test_tensor
torch.cuda.empty_cache()

def create_test_args():
    """Create standardized test arguments for both implementations"""
    args = Namespace()
    args.model = "mistralai/Mixtral-8x7B-v0.1"
    args.max_experts_gpu = 0  # Force CPU offloading for consistent test
    args.cpu_offload = 0
    args.beam_width = 1
    return args

def run_baseline_profile():
    """Profile the baseline FiddlerMixtral implementation"""
    print("🔍 Profiling baseline FiddlerMixtral...")

    # Clear GPU memory
    torch.cuda.empty_cache()

    args = create_test_args()
    model = FiddlerMixtral(args)

    # Test prompt - longer than quick_test for better profiling
    test_prompt = "The capital of France is Paris, which is known for its beautiful architecture and rich cultural heritage. The Eiffel Tower stands majestically in the heart of the city, attracting millions of visitors each year who come to admire"

    # Warmup run
    model.generate(test_prompt, output_token=5)

    # Profile run with more tokens for better analysis
    print("🚀 Starting profiled generation...")
    start_time = time.time()

    # This is the section that will be profiled by nsys
    prefill_time, decode_time, hit_rate = model.generate(test_prompt, output_token=15)

    end_time = time.time()

    print(f"✅ Baseline completed:")
    print(f"   - Total time: {end_time - start_time:.3f}s")
    print(f"   - Prefill: {prefill_time:.3f}s")
    print(f"   - Decode: {decode_time:.3f}s")
    print(f"   - Hit rate: {hit_rate:.2%}")
    print(f"   - Generated: '{model.last_generated_text}'")

    del model
    torch.cuda.empty_cache()

def run_prefetch_profile():
    """Profile the FiddlerMixtralWithPrefetch implementation"""
    print("🔍 Profiling FiddlerMixtralWithPrefetch...")

    # Clear GPU memory
    torch.cuda.empty_cache()

    args = create_test_args()
    model = FiddlerMixtralWithPrefetch(args)

    # Test prompt - same as baseline
    test_prompt = "The capital of France is Paris, which is known for its beautiful architecture and rich cultural heritage. The Eiffel Tower stands majestically in the heart of the city, attracting millions of visitors each year who come to admire"

    # Warmup run to build patterns
    model.generate(test_prompt, output_token=5)

    # Profile run with more tokens
    print("🚀 Starting profiled generation...")
    start_time = time.time()

    # This is the section that will be profiled by nsys
    prefill_time, decode_time, hit_rate = model.generate(test_prompt, output_token=15)

    end_time = time.time()

    print(f"✅ Prefetch completed:")
    print(f"   - Total time: {end_time - start_time:.3f}s")
    print(f"   - Prefill: {prefill_time:.3f}s")
    print(f"   - Decode: {decode_time:.3f}s")
    print(f"   - Hit rate: {hit_rate:.2%}")
    print(f"   - Generated: '{model.last_generated_text}'")

    # Show prefetch metrics if available
    if hasattr(model, 'prefetch_metrics'):
        print(f"   - Prefetch hit rate: {model.prefetch_metrics.get_overall_hit_rate():.2%}")

    del model
    torch.cuda.empty_cache()

def main():
    if len(sys.argv) != 2:
        print("Usage: python profile_implementations.py <baseline|prefetch>")
        sys.exit(1)

    implementation = sys.argv[1].lower()

    if implementation == "baseline":
        run_baseline_profile()
    elif implementation == "prefetch":
        run_prefetch_profile()
    else:
        print("Error: Implementation must be 'baseline' or 'prefetch'")
        sys.exit(1)

if __name__ == "__main__":
    main()