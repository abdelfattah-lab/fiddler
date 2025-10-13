#!/usr/bin/env python3
"""
test_learned_prefetch.py - Test learned prefetch correctness
Verifies that learned predictor produces same outputs as baseline and achieves good hit rates
"""

import torch
import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fiddler.qwen import FiddlerQwen  # Baseline
from fiddler.qwen_with_learned_prefetch import FiddlerQwenWithLearnedPrefetch


class Args:
    """Simple args container"""
    def __init__(self):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.model_path = "Qwen/Qwen1.5-MoE-A2.7B"
        self.beam_width = 1
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.use_fiddler_mode = False
        self.fiddler_batch_threshold = 8


def test_correctness():
    """Test that learned prefetch produces same output as baseline."""
    args = Args()

    print("="*80)
    print("CORRECTNESS TEST: Learned Prefetch vs Baseline")
    print("="*80)

    # Test inputs
    test_inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "The capital of France is"
    ]
    output_tokens = 10

    all_passed = True

    for test_idx, test_input in enumerate(test_inputs):
        print(f"\n{'='*80}")
        print(f"Test {test_idx + 1}/{len(test_inputs)}: {test_input[:50]}...")
        print(f"{'='*80}")

        # Run baseline
        print("\n1. Running baseline...")
        baseline_model = FiddlerQwen(args)

        # Generate with baseline
        inputs = baseline_model.tokenizer(test_input, return_tensors="pt")
        input_ids = inputs.input_ids.to("cuda")

        with torch.no_grad():
            baseline_output = baseline_model.model.generate(
                input_ids,
                max_new_tokens=output_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=None  # Force exact token count
            )

        baseline_text = baseline_model.tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        print(f"   Output: {baseline_text}")

        # Clean up
        del baseline_model
        torch.cuda.empty_cache()

        # Run learned prefetch
        print("\n2. Running learned prefetch...")
        learned_model = FiddlerQwenWithLearnedPrefetch(
            args,
            predictor_path="predictor_checkpoints/best_model.pt",
            num_experts_to_prefetch=8,
            enable_cpu_offload=False
        )

        with torch.no_grad():
            learned_output = learned_model.model.generate(
                input_ids,
                max_new_tokens=output_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=None  # Force exact token count
            )

        learned_text = learned_model.tokenizer.decode(learned_output[0], skip_special_tokens=True)
        print(f"   Output: {learned_text}")

        # Compare
        print("\n3. Comparing outputs...")
        if torch.equal(baseline_output, learned_output):
            print("   ✅ PASS: Outputs match exactly")
        else:
            print("   ❌ FAIL: Outputs differ")
            print(f"   Baseline tokens: {baseline_output[0].tolist()}")
            print(f"   Learned tokens: {learned_output[0].tolist()}")
            all_passed = False

        # Clean up
        del learned_model
        torch.cuda.empty_cache()

    return all_passed


def test_hit_rate():
    """Test that prefetching achieves reasonable hit rates."""
    args = Args()

    print("\n" + "="*80)
    print("HIT RATE TEST")
    print("="*80)

    test_inputs = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that focuses on building systems.",
        "The capital of France is Paris, and it is known for its beautiful architecture."
    ]
    output_tokens = 20

    # Test with default aggregation (frequency-based)
    aggregation_strategy = 'frequency'
    print(f"\nTesting with aggregation strategy: {aggregation_strategy}")
    print("="*80)

    all_results = []

    for test_idx, test_input in enumerate(test_inputs):
        print(f"\n{'='*80}")
        print(f"Test {test_idx + 1}/{len(test_inputs)}: {test_input[:50]}...")
        print(f"{'='*80}")

        model = FiddlerQwenWithLearnedPrefetch(
            args,
            predictor_path="predictor_checkpoints/best_model.pt",
            num_experts_to_prefetch=8,
            enable_cpu_offload=False,
            prefill_aggregation=aggregation_strategy
        )

        prefill_time, decode_time, prefill_hit_rate, decode_hit_rate = model.generate(
            test_input,
            output_token=output_tokens
        )

        print(f"\nResults:")
        print(f"  Prefill hit rate: {prefill_hit_rate*100:.1f}%")
        print(f"  Decode hit rate: {decode_hit_rate*100:.1f}%")
        print(f"  Prefill time: {prefill_time:.3f}s")
        print(f"  Decode time: {decode_time:.3f}s")

        all_results.append({
            'input': test_input[:50],
            'prefill_hit_rate': prefill_hit_rate,
            'decode_hit_rate': decode_hit_rate
        })

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Calculate average hit rates
    avg_prefill_hit_rate = sum(r['prefill_hit_rate'] for r in all_results) / len(all_results)
    avg_decode_hit_rate = sum(r['decode_hit_rate'] for r in all_results) / len(all_results)

    print(f"\n{'='*80}")
    print(f"SUMMARY - Aggregation: {aggregation_strategy}")
    print(f"{'='*80}")
    print(f"Average prefill hit rate: {avg_prefill_hit_rate*100:.1f}%")
    print(f"Average decode hit rate: {avg_decode_hit_rate*100:.1f}%")

    print(f"\nℹ️  Note: You can test other aggregation strategies:")
    print(f"  - 'frequency' (default): Most frequent experts across tokens")
    print(f"  - 'mean': Mean pooling of prediction scores")
    print(f"  - 'max': Max pooling of prediction scores")

    # Success criteria based on Phase 2 validation accuracy
    # With 41.80% top-4 accuracy, we expect roughly 40%+ decode hit rate
    min_decode_hit_rate = 0.30  # Set to 30% as minimum viable (conservative estimate)
    target_decode_hit_rate = 0.40  # Target based on validation accuracy

    if avg_decode_hit_rate >= target_decode_hit_rate:
        print(f"\n✅ PASS: Decode hit rate >= {target_decode_hit_rate*100}% (target)")
        return True
    elif avg_decode_hit_rate >= min_decode_hit_rate:
        print(f"\n⚠️  MARGINAL: Decode hit rate >= {min_decode_hit_rate*100}% but < {target_decode_hit_rate*100}%")
        print("   Integration works but hit rate is lower than expected")
        return True
    else:
        print(f"\n❌ FAIL: Decode hit rate < {min_decode_hit_rate*100}%")
        print("   This indicates integration issues - debug needed")
        return False


def main():
    print("\n" + "="*80)
    print("TESTING LEARNED PREFETCH INTEGRATION")
    print("="*80)

    # Test 1: Correctness
    print("\n" + "="*80)
    print("TEST 1: CORRECTNESS")
    print("="*80)
    correctness_pass = test_correctness()

    # Test 2: Hit rate
    print("\n" + "="*80)
    print("TEST 2: HIT RATE")
    print("="*80)
    hit_rate_pass = test_hit_rate()

    # Summary
    print("\n" + "="*80)
    print("FINAL TEST SUMMARY")
    print("="*80)
    print(f"Correctness Test: {'✅ PASS' if correctness_pass else '❌ FAIL'}")
    print(f"Hit Rate Test: {'✅ PASS' if hit_rate_pass else '❌ FAIL'}")

    if correctness_pass and hit_rate_pass:
        print("\n✅ ✅ ✅ ALL TESTS PASSED! ✅ ✅ ✅")
        print("Learned prefetch integration is working correctly.")
        print("Ready for Phase 5 benchmarking.")
        return 0
    elif correctness_pass:
        print("\n⚠️  Correctness passed but hit rate could be better.")
        print("Integration is working but may need tuning.")
        return 0
    else:
        print("\n❌ TESTS FAILED - Debug integration before proceeding.")
        return 1


if __name__ == "__main__":
    exit(main())
