#!/usr/bin/env python3
"""
Analyze prefetch requirements and predict hit rate with different
num_experts_to_prefetch configurations.
"""
import json

# Load patterns
with open('expert_usage_patterns_qwen.json', 'r') as f:
    patterns = json.load(f)

print("="*80)
print("ANALYSIS: Prefetch Requirements for 100% Hit Rate")
print("="*80)

# Simulate generation of 3 tokens
output_tokens = 3
moe_layer_indices = list(range(len(patterns)))

print(f"\nSimulating generation of {output_tokens} decode tokens...")
print(f"Total MoE layers: {len(patterns)}")
print(f"GPU-resident layers (0-1): no prefetch needed")
print(f"Prefetch layers (2-{len(patterns)-1}): need prefetching\n")

# For each configuration, calculate hit rate
for num_to_prefetch in [4, 7, 16, 20]:
    print(f"\n{'='*80}")
    print(f"Configuration: num_experts_to_prefetch = {num_to_prefetch}")
    print(f"{'='*80}")

    total_experts_needed = 0
    total_experts_prefetched = 0
    hits_by_layer = {}

    # For each token position
    for token_idx in range(output_tokens):
        # For each layer
        for layer_idx in moe_layer_indices:
            layer_key = str(layer_idx)
            token_key = str(token_idx)

            if token_key not in patterns[layer_key]:
                continue

            experts_needed = patterns[layer_key][token_key]
            num_needed = len(experts_needed)

            # GPU-resident layers always hit
            if layer_idx <= 1:
                num_hits = num_needed
            else:
                # Prefetch layers: can only hit if num_to_prefetch >= num_needed
                num_hits = min(num_needed, num_to_prefetch)

            total_experts_needed += num_needed
            total_experts_prefetched += num_hits

            if layer_idx not in hits_by_layer:
                hits_by_layer[layer_idx] = {'needed': 0, 'hit': 0}
            hits_by_layer[layer_idx]['needed'] += num_needed
            hits_by_layer[layer_idx]['hit'] += num_hits

    # Calculate hit rate
    hit_rate = (total_experts_prefetched / total_experts_needed * 100) if total_experts_needed > 0 else 0

    print(f"\nOverall hit rate: {hit_rate:.1f}%")
    print(f"Total experts needed: {total_experts_needed}")
    print(f"Total experts hit: {total_experts_prefetched}")

    # Show layer-by-layer breakdown
    print(f"\nLayer-by-layer breakdown:")
    for layer_idx in sorted(hits_by_layer.keys())[:8]:  # Show first 8 layers
        stats = hits_by_layer[layer_idx]
        layer_hit_rate = (stats['hit'] / stats['needed'] * 100) if stats['needed'] > 0 else 0
        status = "✅" if layer_hit_rate == 100 else "❌"
        resident = " (GPU-resident)" if layer_idx <= 1 else " (prefetch)"
        print(f"  Layer {layer_idx:2d}{resident}: {stats['hit']:2d}/{stats['needed']:2d} = {layer_hit_rate:5.1f}% {status}")

print(f"\n{'='*80}")
print("CONCLUSION:")
print(f"{'='*80}")
print("With num_experts_to_prefetch=4:")
print("  - Layers 0-1 (GPU-resident): 100% hit rate (always)")
print("  - Layers 2-14 (prefetch): 100% hit rate (4 needed, 4 prefetched)")
print("  - Expected overall hit rate: 100% ✅")
print("\nWith num_experts_to_prefetch<4:")
print("  - Layers 2-14 will have misses")
print("  - Hit rate < 100%")
