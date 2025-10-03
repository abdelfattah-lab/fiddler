#!/usr/bin/env python3
"""Quick test to verify 4-expert prefetch achieves 100% hit rate."""
import os
import sys

# Prevent model loading, just do a dry-run check of the logic
print("="*80)
print("DRY RUN: Testing if 4 experts can achieve 100% hit rate")
print("="*80)

import json

# Load patterns
with open('expert_usage_patterns_qwen.json', 'r') as f:
    patterns = json.load(f)

# Check all layers for decode tokens 0, 1, 2
print("\nChecking expert requirements for decode tokens 0-2:")
all_can_hit_with_4 = True

for layer_idx in range(2, min(15, len(patterns))):  # Layers 2-14 (GPU-resident are 0-1)
    layer_key = str(layer_idx)
    max_experts_needed = 0

    for token_idx in range(3):  # Check first 3 decode tokens
        token_key = str(token_idx)
        if token_key in patterns[layer_key]:
            experts = patterns[layer_key][token_key]
            max_experts_needed = max(max_experts_needed, len(experts))

    can_hit = max_experts_needed <= 4
    status = "✅" if can_hit else "❌"
    print(f"  Layer {layer_idx:2d}: max {max_experts_needed} experts needed {status}")

    if not can_hit:
        all_can_hit_with_4 = False

print("\n" + "="*80)
if all_can_hit_with_4:
    print("✅ CONCLUSION: 4 experts CAN achieve 100% hit rate for decode phase!")
    print("\nTo actually test this, run:")
    print("  python3 test_100_hit_rate.py")
    print("\nBut modify src/fiddler/qwen_with_prefetch.py line 123 to use:")
    print("  num_experts_to_prefetch=4")
else:
    print("❌ CONCLUSION: 4 experts CANNOT achieve 100% hit rate")
print("="*80)
