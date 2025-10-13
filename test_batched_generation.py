#!/usr/bin/env python3
"""
Validation test for batched generation support.
Tests that batched inputs work correctly with learned prefetch and maintain non-zero hit rates.
"""
import sys
sys.path.insert(0, 'src')
from fiddler.qwen_with_learned_prefetch import FiddlerQwenWithLearnedPrefetch

class Args:
    def __init__(self):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1

args = Args()

# Test 1: Single input (should already work)
print("="*80)
print("TEST 1: Single input (BS=1)")
print("="*80)
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
print(f"✅ Single input: Decode hit rate = {decode_hr*100:.1f}%")
assert decode_hr > 0.0, f"ERROR: Hit rate is 0% for single input!"
assert decode_hr > 0.3, f"Expected >30% hit rate, got {decode_hr*100:.1f}%"
del model

# Test 2: Batched input (THIS IS WHAT NEEDS TO WORK)
print("\n" + "="*80)
print("TEST 2: Batched input (BS=2)")
print("="*80)
model = FiddlerQwenWithLearnedPrefetch(
    args,
    num_experts_to_prefetch=8,
    enable_cpu_offload=False,
    predictor_path='predictor_checkpoints/best_model.pt'
)
prefill_t, decode_t, prefill_hr, decode_hr = model.generate(
    ["The capital of France is", "The theory of relativity was"],
    output_token=10
)
print(f"✅ Batched input: Decode hit rate = {decode_hr*100:.1f}%")
assert decode_hr > 0.0, f"ERROR: Hit rate is 0% for batched input!"
assert decode_hr > 0.3, f"Expected >30% hit rate, got {decode_hr*100:.1f}%"
del model

# Test 3: Larger batch (BS=4)
print("\n" + "="*80)
print("TEST 3: Larger batch (BS=4)")
print("="*80)
model = FiddlerQwenWithLearnedPrefetch(
    args,
    num_experts_to_prefetch=8,
    enable_cpu_offload=False,
    predictor_path='predictor_checkpoints/best_model.pt'
)
prefill_t, decode_t, prefill_hr, decode_hr = model.generate(
    [
        "The capital of France is",
        "The theory of relativity was",
        "Machine learning is used for",
        "The solar system contains"
    ],
    output_token=10
)
print(f"✅ Larger batch: Decode hit rate = {decode_hr*100:.1f}%")
assert decode_hr > 0.0, f"ERROR: Hit rate is 0% for batched input!"
assert decode_hr > 0.3, f"Expected >30% hit rate, got {decode_hr*100:.1f}%"
del model

print("\n" + "="*80)
print("ALL TESTS PASSED ✅")
print("="*80)
print("\nBatched generation support is working correctly!")
print("Hit rates are non-zero for all batch sizes, indicating prefetch hooks are firing correctly.")
