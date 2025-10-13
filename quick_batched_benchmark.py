#!/usr/bin/env python3
"""
Quick validation that batched generation works correctly with learned prefetch.
Tests key configurations to verify hit rates are non-zero.
"""
import os
import sys
sys.path.insert(0, 'src')

from fiddler.qwen import FiddlerQwen
from fiddler.qwen_with_learned_prefetch import FiddlerQwenWithLearnedPrefetch

class Args:
    def __init__(self):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1

print("="*80)
print("QUICK BATCHED GENERATION VALIDATION")
print("="*80)

# Test 1: Baseline BS=2
print("\n" + "="*80)
print("TEST 1: Baseline (BS=2)")
print("="*80)
args = Args()
model = FiddlerQwen(args)
prefill_t, decode_t, prefill_hr, decode_hr = model.generate(
    ["The capital of France is", "The theory of relativity"],
    output_token=10
)
print(f"✅ Baseline BS=2: Decode time = {decode_t:.3f}s, Hit rate = {decode_hr*100:.1f}%")
del model

# Test 2: Learned-Prefetch BS=1
print("\n" + "="*80)
print("TEST 2: Learned-Prefetch (BS=1)")
print("="*80)
args = Args()
model = FiddlerQwenWithLearnedPrefetch(
    args,
    num_experts_to_prefetch=8,
    enable_cpu_offload=False,
    predictor_path='predictor_checkpoints/best_model.pt'
)
prefill_t, decode_t, prefill_hr, decode_hr = model.generate(
    "The capital of France is",
    output_token=10
)
print(f"✅ Learned-Prefetch BS=1: Decode time = {decode_t:.3f}s, Hit rate = {decode_hr*100:.1f}%")
assert decode_hr > 0.3, f"ERROR: Expected >30% hit rate, got {decode_hr*100:.1f}%"
del model

# Test 3: Learned-Prefetch BS=2
print("\n" + "="*80)
print("TEST 3: Learned-Prefetch (BS=2)")
print("="*80)
args = Args()
model = FiddlerQwenWithLearnedPrefetch(
    args,
    num_experts_to_prefetch=8,
    enable_cpu_offload=False,
    predictor_path='predictor_checkpoints/best_model.pt'
)
prefill_t, decode_t, prefill_hr, decode_hr = model.generate(
    ["The capital of France is", "The theory of relativity"],
    output_token=10
)
print(f"✅ Learned-Prefetch BS=2: Decode time = {decode_t:.3f}s, Hit rate = {decode_hr*100:.1f}%")
assert decode_hr > 0.0, f"ERROR: Hit rate is 0% for batched input!"
assert decode_hr > 0.3, f"ERROR: Expected >30% hit rate, got {decode_hr*100:.1f}%"
del model

# Test 4: Fiddler+Learned-Prefetch BS=2
print("\n" + "="*80)
print("TEST 4: Fiddler+Learned-Prefetch (BS=2)")
print("="*80)
args = Args()
model = FiddlerQwenWithLearnedPrefetch(
    args,
    num_experts_to_prefetch=8,
    enable_cpu_offload=True,
    latency_cpu=0.1,
    latency_gpu=10.0,
    predictor_path='predictor_checkpoints/best_model.pt'
)
prefill_t, decode_t, prefill_hr, decode_hr = model.generate(
    ["The capital of France is", "The theory of relativity"],
    output_token=10
)
print(f"✅ Fiddler+Learned BS=2: Decode time = {decode_t:.3f}s, Hit rate = {decode_hr*100:.1f}%")
assert decode_hr > 0.0, f"ERROR: Hit rate is 0% for batched input with Fiddler!"
assert decode_hr > 0.3, f"ERROR: Expected >30% hit rate, got {decode_hr*100:.1f}%"
del model

print("\n" + "="*80)
print("✅ ALL QUICK VALIDATION TESTS PASSED!")
print("="*80)
print("\nKey findings:")
print("  1. Baseline supports batched generation")
print("  2. Learned-Prefetch works for BS=1 with good hit rates (>30%)")
print("  3. Learned-Prefetch works for BS=2 with good hit rates (>30%)")
print("  4. Fiddler+Learned-Prefetch works for BS=2 with good hit rates (>30%)")
print("\n🎉 Batched generation support is working correctly!")
print("   Ready to run full Phase 5 benchmark with all configurations.")
