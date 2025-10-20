#!/usr/bin/env python3
"""
Quick test of inline oracle implementation in benchmark_prediction_methods.py
Verifies that it achieves 100% efficiency for a single batch size.
"""

import os
import sys
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from benchmark_prediction_methods import (
    Args, InlineOracleCollector, run_single_batch_oracle_inline, get_diverse_batch
)


def main():
    """Quick test of inline oracle implementation."""
    print("="*80)
    print("QUICK TEST: Inline Oracle Implementation")
    print("="*80)

    # Test configuration
    batch_sizes = [1, 2, 4]
    output_tokens = 20

    for batch_size in batch_sizes:
        print(f"\n\n{'='*80}")
        print(f"Testing Batch Size {batch_size}")
        print(f"{'='*80}")

        # Create inline oracle collector
        args = Args(use_fiddler_mode=False)
        collector = InlineOracleCollector(
            args,
            num_experts_to_prefetch=4,  # Match top-k
            enable_cpu_offload=False
        )

        # Get test prompts
        prompts = get_diverse_batch(batch_size, seed=0)
        print(f"\nPrompts: {[p[:30] + '...' for p in prompts]}")

        # Run inline oracle test
        print(f"\n🔮 Running inline oracle test...")
        result = run_single_batch_oracle_inline(collector, prompts, output_tokens)

        # Display results
        print(f"\n{'='*80}")
        print(f"RESULTS - Batch Size {batch_size}")
        print(f"{'='*80}")
        print(f"Total time:       {result['total_time']:.3f}s")
        print(f"Prefill time:     {result['prefill_time']:.3f}s")
        print(f"Decode time:      {result['decode_time']:.3f}s ({result['decode_time_per_token']:.3f}s/token)")
        print(f"Tokens/sec:       {result['tokens_per_second']:.1f}")
        print(f"\nPrefill hit rate:  {result['prefill_hit_rate']*100:.1f}%")
        print(f"Decode hit rate:   {result['decode_hit_rate']*100:.1f}%")

        # Check oracle efficiency
        print(f"\n🔮 ORACLE EFFICIENCY:")
        print(f"Overall:  {result['oracle_efficiency_overall']:.1f}%")
        print(f"Prefill:  {result['oracle_efficiency_prefill']:.1f}%")
        print(f"Decode:   {result['oracle_efficiency_decode']:.1f}%")
        print(f"Prefetched: {result['oracle_total_prefetched']} | Used: {result['oracle_total_used']}")

        # Validate 100% efficiency
        if result['oracle_efficiency_overall'] >= 99.9:
            print(f"\n✅ SUCCESS: Achieved 100% oracle efficiency!")
        else:
            print(f"\n❌ FAILED: Efficiency is {result['oracle_efficiency_overall']:.1f}%, not 100%")

        # Cleanup
        del collector
        torch.cuda.empty_cache()

    print(f"\n\n{'='*80}")
    print("✅ QUICK TEST COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
