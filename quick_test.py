#!/usr/bin/env python3
"""
Quick Testing Infrastructure - Fast iteration for AI agents

Loads models once, generates 2 tokens, verifies correctness, reports timing.
Perfect for rapid development cycles.

Usage: python quick_test.py ModelToTest
Example: python quick_test.py MixtralWithBuffers
"""

import os
import sys
import time
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Quick CUDA init
torch.cuda.init()
torch.cuda.set_device(0)

from fiddler.mixtral import FiddlerMixtral
from fiddler.mixtral_with_buffers import MixtralWithBuffers
from fiddler.mixtral_with_predictor import FiddlerMixtralWithPredictor

MODELS = {
    'FiddlerMixtral': FiddlerMixtral,
    'MixtralWithBuffers': MixtralWithBuffers,
    'FiddlerMixtralWithPredictor': FiddlerMixtralWithPredictor,
}

def create_args():
    class Args:
        def __init__(self):
            self.model = 'mistralai/Mixtral-8x7B-v0.1'
            self.beam_width = 1
            self.cpu_offload = 0
            self.max_experts_gpu = 0
    return Args()

def quick_test(model_name):
    if model_name not in MODELS:
        print(f"❌ Unknown model: {model_name}")
        print(f"Available: {list(MODELS.keys())}")
        return

    print(f"🚀 Quick test: {model_name}")

    # Test modelthou    
    print(f"Loading {model_name}...")
    test_model = MODELS[model_name](create_args())

    start = time.time()
    prefill_time, decode_time, hit_rate = test_model.generate("The capital of France is", output_token=3)
    total_time = time.time() - start
    test_output = getattr(test_model, 'last_generated_text', '')

    print(f"{model_name}: '{test_output}' (prefill: {prefill_time:.3f}s, decode: {decode_time:.3f}s, total: {total_time:.3f}s)")

    del test_model
    torch.cuda.empty_cache()

    # Baseline comparison if not baseline
    if model_name != 'FiddlerMixtral':
        print(f"Loading FiddlerMixtral...")
        baseline_model = FiddlerMixtral(create_args())

        start = time.time()
        base_prefill, base_decode, base_hit = baseline_model.generate("The capital of France is", output_token=3)
        base_total = time.time() - start
        baseline_output = getattr(baseline_model, 'last_generated_text', '')

        print(f"FiddlerMixtral: '{baseline_output}' (prefill: {base_prefill:.3f}s, decode: {base_decode:.3f}s, total: {base_total:.3f}s)")

        # Results
        if test_output == baseline_output:
            total_speedup = base_total / total_time
            prefill_speedup = base_prefill / prefill_time
            decode_speedup = base_decode / decode_time
            print(f"✅ MATCH! Speedup - Total: {total_speedup:.2f}x, Prefill: {prefill_speedup:.2f}x, Decode: {decode_speedup:.2f}x")
        else:
            print(f"❌ MISMATCH!")
            print(f"  {model_name}: '{test_output}'")
            print(f"  Baseline: '{baseline_output}'")

        del baseline_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python quick_test.py ModelName")
        print(f"Available: {list(MODELS.keys())}")
        sys.exit(1)

    quick_test(sys.argv[1])