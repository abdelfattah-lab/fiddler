#!/usr/bin/env python3
"""
Verify that during decode phase (autoregressive generation),
each layer needs exactly 4 experts per token.
"""
import json

# Load patterns
with open('expert_usage_patterns_qwen.json', 'r') as f:
    patterns = json.load(f)

print("="*80)
print("VERIFICATION: 4 experts sufficient for decode phase")
print("="*80)

# During generation, token positions increment for each new token
# We want to verify that for each (layer, token_pos) pair,
# we need exactly 4 experts (matching top_k=4)

decode_only_layers = []  # Layers that need exactly 4 experts per token

for layer_idx in range(len(patterns)):
    layer_key = str(layer_idx)
    if layer_key not in patterns:
        continue

    layer_data = patterns[layer_key]

    # Check if all positions in this layer need exactly 4 experts
    all_positions_need_4 = True
    max_experts = 0

    for token_key, experts in layer_data.items():
        num_experts = len(experts)
        max_experts = max(max_experts, num_experts)
        if num_experts != 4:
            all_positions_need_4 = False

    if all_positions_need_4:
        decode_only_layers.append(layer_idx)
        print(f"Layer {layer_idx}: ✅ All positions need exactly 4 experts (decode-only)")
    else:
        print(f"Layer {layer_idx}: ❌ Variable experts per position (max={max_experts}, includes prompt processing)")

print("\n" + "="*80)
print("SUMMARY:")
print("="*80)
print(f"Layers that need exactly 4 experts: {decode_only_layers}")
print(f"Number of layers: {len(decode_only_layers)} out of {len(patterns)}")

# Check MoE layer structure
print("\nNote: According to the code, MoE layers 0-1 are kept permanently on GPU.")
print("So they don't need prefetching at all.")
print(f"\nFor layers 2+, we need 4 experts per token to achieve 100% decode hit rate.")
print(f"Layers that can achieve 100% with 4 experts: {[l for l in decode_only_layers if l >= 2]}")
