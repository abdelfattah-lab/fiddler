#!/usr/bin/env python3
"""
Inline Oracle Benchmark - Achieve 100% Oracle Prefetch Efficiency

This script implements Option 1 from the guide: Inline Oracle Collection.

Key principle: Collect oracle data in the SAME Python process, SAME model instance,
with SAME random state as the usage. This eliminates non-determinism and achieves
100% efficiency.

Approach:
1. For each prompt, run TWO forward passes in sequence using the SAME model instance:
   a. First pass: Record actual expert usage (no prefetch)
   b. Second pass: Use those exact decisions for oracle prefetch
2. Verify 100% efficiency (every prefetched expert is used)

This is fundamentally different from the current approach which:
- Collects oracle in one run
- Saves to file
- Loads and uses in a different run (different model instance)
- Results in non-determinism and <100% efficiency
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import numpy as np
import random

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from benchmark_prediction_methods import get_diverse_batch


class Args:
    def __init__(self):
        self.model = "Qwen/Qwen1.5-MoE-A2.7B"
        self.cpu_offload = 0
        self.max_experts_gpu = 0
        self.beam_width = 1


class InlineOracleModel:
    """
    Model that supports inline oracle collection and usage.

    Collects actual expert usage during one forward pass, then immediately
    uses that data for prefetching in the next forward pass - all in the
    same model instance to ensure perfect determinism.
    """

    def __init__(self, args):
        """Initialize model."""
        from fiddler.qwen import FiddlerQwen

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Set seeds for reproducibility
        self._set_seeds()

        print("Loading FiddlerQwen model...")
        self.model = FiddlerQwen(args)
        self.model.model.eval()

        # Get MoE layer info
        self.moe_layers = self.model.moe_layers
        self.n_experts = self.model.n_expert
        self.top_k = 4  # Qwen uses top-4

        print(f"✅ Model loaded: {len(self.moe_layers)} MoE layers, {self.n_experts} experts each")

        # Oracle data collected during first pass
        self.collected_oracle = {}  # [layer_idx][token_pos] -> [expert_ids]

        # Tracking for efficiency measurement
        self.prefetched_experts = defaultdict(lambda: defaultdict(set))
        self.used_experts = defaultdict(lambda: defaultdict(set))

        # State
        self.mode = 'idle'  # 'collect', 'oracle', 'idle'
        self.current_token_pos = 0
        self.prefill_length = 0

        # Install hooks
        self._install_hooks()

    def _set_seeds(self, seed=42):
        """Set all random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Deterministic CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _install_hooks(self):
        """Install hooks to capture gating decisions."""
        for layer_idx in self.moe_layers:
            layer = self.model.model.model.layers[layer_idx]

            def create_hooked_forward(layer_idx):
                original_forward = layer.mlp.forward

                def hooked_forward(hidden_states):
                    # Capture gating decisions before forward
                    self._capture_gating(hidden_states, layer_idx)
                    # Call original forward
                    return original_forward(hidden_states)

                return hooked_forward

            layer.mlp.forward = create_hooked_forward(layer_idx)

    def _capture_gating(self, hidden_states, layer_idx):
        """Capture gating decisions for this layer."""
        if self.mode == 'idle':
            return

        moe_layer = self.model.model.model.layers[layer_idx].mlp
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Flatten hidden states for gating
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute router logits (same as in actual forward)
        router_logits = moe_layer.gate(hidden_states_flat)

        # Get routing weights and selected experts
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, moe_layer.top_k, dim=-1)

        # Convert to list
        selected_experts_np = selected_experts.cpu().numpy()

        # Record gating decisions
        for seq_idx in range(sequence_length):
            expert_ids = selected_experts_np[seq_idx].tolist()

            # Determine token position
            if sequence_length > 1:
                # Prefill phase
                token_pos = seq_idx
            else:
                # Decode phase
                token_pos = self.current_token_pos

            if self.mode == 'collect':
                # Collection mode: record oracle data
                if layer_idx not in self.collected_oracle:
                    self.collected_oracle[layer_idx] = {}
                self.collected_oracle[layer_idx][token_pos] = expert_ids

            elif self.mode == 'oracle':
                # Oracle mode: record what was actually used (for validation)
                self.used_experts[layer_idx][token_pos].update(expert_ids)

    def _get_oracle_experts_for_layer(self, layer_idx, token_pos, sequence_length=1):
        """
        Get oracle expert predictions for this layer.

        For prefill (sequence_length > 1), aggregate experts from all positions.
        For decode (sequence_length == 1), get experts for current position.
        """
        if layer_idx not in self.collected_oracle:
            return []

        layer_oracle = self.collected_oracle[layer_idx]

        if sequence_length > 1:
            # Prefill: aggregate experts from all positions in the sequence
            # Use Counter to track frequency
            expert_counter = Counter()
            for pos in range(sequence_length):
                if pos in layer_oracle:
                    expert_counter.update(layer_oracle[pos])

            # Return top-k most frequent experts
            most_common = expert_counter.most_common(self.top_k)
            experts = [expert_id for expert_id, count in most_common]

            # Record what we prefetched
            self.prefetched_experts[layer_idx][0].update(experts)

            return experts
        else:
            # Decode: get experts for this specific position
            experts = layer_oracle.get(token_pos, [])

            # Record what we prefetched
            if experts:
                self.prefetched_experts[layer_idx][token_pos].update(experts)

            return experts

    def _generate_with_expert_loading(self, input_ids, attention_mask, output_tokens, mode='collect'):
        """
        Generate tokens with expert loading management.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            output_tokens: Number of tokens to generate
            mode: 'collect' or 'oracle'

        Returns:
            generated_ids: Generated token IDs
        """
        self.mode = mode
        self.current_token_pos = 0

        batch_size = input_ids.shape[0]
        prefill_length = input_ids.shape[1]
        self.prefill_length = prefill_length

        with torch.no_grad():
            current_input_ids = input_ids
            current_attention_mask = attention_mask

            # Generate tokens one by one
            for token_idx in range(output_tokens):
                self.current_token_pos = prefill_length + token_idx

                # In oracle mode, prefetch experts for upcoming layers
                if mode == 'oracle':
                    # Determine sequence length for this step
                    sequence_length = current_input_ids.shape[1] if token_idx == 0 else 1

                    # Prefetch experts for all MoE layers
                    for layer_idx in self.moe_layers:
                        experts = self._get_oracle_experts_for_layer(
                            layer_idx,
                            self.current_token_pos,
                            sequence_length
                        )
                        # In a real implementation, we would load these experts here
                        # For this benchmark, we just track what would be prefetched

                # Forward pass (use_cache=False for determinism)
                outputs = self.model.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    use_cache=False
                )

                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to input
                current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                if current_attention_mask is not None:
                    current_attention_mask = torch.cat([
                        current_attention_mask,
                        torch.ones((batch_size, 1), dtype=current_attention_mask.dtype, device=self.device)
                    ], dim=-1)

                # Only do prefill once
                if token_idx == 0:
                    # After prefill, switch to decode
                    pass

        self.mode = 'idle'
        return current_input_ids

    def run_inline_oracle_test(self, prompts: List[str], output_tokens: int = 20):
        """
        Run inline oracle test: collect and use in same model instance.

        Args:
            prompts: List of input prompts
            output_tokens: Number of tokens to generate

        Returns:
            dict with efficiency metrics
        """
        batch_size = len(prompts)
        print(f"\n{'='*80}")
        print(f"INLINE ORACLE TEST - Batch Size {batch_size}")
        print(f"{'='*80}")

        # Reset state
        self.collected_oracle = {}
        self.prefetched_experts = defaultdict(lambda: defaultdict(set))
        self.used_experts = defaultdict(lambda: defaultdict(set))
        self._set_seeds()  # Reset seeds for this test

        # Tokenize
        if batch_size == 1:
            inputs = self.model.tokenizer(prompts[0], return_tensors="pt")
        else:
            inputs = self.model.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if inputs.attention_mask is not None else None

        # STEP 1: Collect actual expert usage (no prefetch)
        print(f"\n📝 Step 1: Collecting actual expert usage...")
        self._set_seeds()  # Ensure same seed
        generated_ids_collect = self._generate_with_expert_loading(
            input_ids,
            attention_mask,
            output_tokens,
            mode='collect'
        )

        # Verify oracle data collected
        total_positions = len(self.collected_oracle.get(self.moe_layers[0], {}))
        print(f"✅ Collected oracle data for {total_positions} token positions")

        # STEP 2: Use oracle data for prefetching (same model instance, same seeds)
        print(f"\n🔮 Step 2: Using oracle data for prefetching...")
        self._set_seeds()  # Reset to SAME seed as collection
        generated_ids_oracle = self._generate_with_expert_loading(
            input_ids,
            attention_mask,
            output_tokens,
            mode='oracle'
        )

        # STEP 3: Verify outputs are identical
        outputs_match = torch.equal(generated_ids_collect, generated_ids_oracle)
        print(f"\n🔍 Step 3: Verifying outputs...")
        print(f"   Outputs match: {outputs_match}")

        if not outputs_match:
            print(f"   ⚠️  WARNING: Outputs differ! This suggests non-determinism.")

        # STEP 4: Calculate efficiency
        print(f"\n📊 Step 4: Calculating efficiency...")
        efficiency = self._calculate_efficiency()

        print(f"\n{'='*80}")
        print(f"EFFICIENCY RESULTS - Batch Size {batch_size}")
        print(f"{'='*80}")
        print(f"Overall Efficiency:  {efficiency['overall']:.1f}%")
        print(f"Prefill Efficiency:  {efficiency['prefill']:.1f}%")
        print(f"Decode Efficiency:   {efficiency['decode']:.1f}%")
        print(f"\nDetails:")
        print(f"  Prefill: {efficiency['prefill_used']}/{efficiency['prefill_prefetched']} experts used")
        print(f"  Decode:  {efficiency['decode_used']}/{efficiency['decode_prefetched']} experts used")
        print(f"  Total:   {efficiency['total_used']}/{efficiency['total_prefetched']} experts used")

        if efficiency['overall'] >= 99.9:
            print(f"\n✅ SUCCESS: Achieved 100% efficiency!")
        else:
            print(f"\n❌ FAILED: Efficiency is {efficiency['overall']:.1f}%, not 100%")
            self._print_efficiency_details()

        print(f"{'='*80}")

        return efficiency

    def _calculate_efficiency(self):
        """Calculate efficiency = (prefetched ∩ used) / prefetched."""
        total_prefetched = 0
        total_used = 0

        prefill_prefetched = 0
        prefill_used = 0

        decode_prefetched = 0
        decode_used = 0

        for layer_idx in self.prefetched_experts:
            # Handle prefill (token_pos=0)
            if 0 in self.prefetched_experts[layer_idx]:
                prefetched_set = self.prefetched_experts[layer_idx][0]

                # Aggregate usage across all prefill positions
                used_during_prefill = set()
                for pos in range(self.prefill_length):
                    if pos in self.used_experts[layer_idx]:
                        used_during_prefill.update(self.used_experts[layer_idx][pos])

                # How many prefetched were used?
                used_from_prefetched = len(prefetched_set & used_during_prefill)
                num_prefetched = len(prefetched_set)

                total_prefetched += num_prefetched
                total_used += used_from_prefetched
                prefill_prefetched += num_prefetched
                prefill_used += used_from_prefetched

            # Handle decode positions
            for token_pos in self.prefetched_experts[layer_idx]:
                if token_pos >= self.prefill_length:
                    prefetched_set = self.prefetched_experts[layer_idx][token_pos]
                    used_set = self.used_experts[layer_idx].get(token_pos, set())

                    used_from_prefetched = len(prefetched_set & used_set)
                    num_prefetched = len(prefetched_set)

                    total_prefetched += num_prefetched
                    total_used += used_from_prefetched
                    decode_prefetched += num_prefetched
                    decode_used += used_from_prefetched

        overall_efficiency = (total_used / total_prefetched * 100) if total_prefetched > 0 else 0
        prefill_efficiency = (prefill_used / prefill_prefetched * 100) if prefill_prefetched > 0 else 0
        decode_efficiency = (decode_used / decode_prefetched * 100) if decode_prefetched > 0 else 0

        return {
            'overall': overall_efficiency,
            'prefill': prefill_efficiency,
            'decode': decode_efficiency,
            'total_prefetched': total_prefetched,
            'total_used': total_used,
            'prefill_prefetched': prefill_prefetched,
            'prefill_used': prefill_used,
            'decode_prefetched': decode_prefetched,
            'decode_used': decode_used
        }

    def _print_efficiency_details(self):
        """Print detailed efficiency analysis."""
        print(f"\n{'='*80}")
        print("DETAILED EFFICIENCY ANALYSIS")
        print(f"{'='*80}")

        for layer_idx in sorted(self.prefetched_experts.keys()):
            # Handle prefill
            if 0 in self.prefetched_experts[layer_idx]:
                prefetched_set = self.prefetched_experts[layer_idx][0]

                # Aggregate usage
                used_during_prefill = set()
                for pos in range(self.prefill_length):
                    if pos in self.used_experts[layer_idx]:
                        used_during_prefill.update(self.used_experts[layer_idx][pos])

                unused = prefetched_set - used_during_prefill
                if unused:
                    print(f"\n⚠️  Layer {layer_idx}, PREFILL (positions 0-{self.prefill_length-1}):")
                    print(f"    Prefetched: {sorted(prefetched_set)}")
                    print(f"    Used: {sorted(used_during_prefill)}")
                    print(f"    UNUSED: {sorted(unused)}")

            # Handle decode
            for token_pos in self.prefetched_experts[layer_idx]:
                if token_pos >= self.prefill_length:
                    prefetched_set = self.prefetched_experts[layer_idx][token_pos]
                    used_set = self.used_experts[layer_idx].get(token_pos, set())

                    unused = prefetched_set - used_set
                    if unused:
                        print(f"\n⚠️  Layer {layer_idx}, Token {token_pos} (DECODE):")
                        print(f"    Prefetched: {sorted(prefetched_set)}")
                        print(f"    Used: {sorted(used_set)}")
                        print(f"    UNUSED: {sorted(unused)}")

    def save_collected_oracle(self, output_path="oracle_gating_decisions_inline.json"):
        """Save collected oracle data to JSON file."""
        # Convert to JSON-serializable format
        json_oracle = {}
        for layer_idx, layer_data in self.collected_oracle.items():
            json_oracle[str(layer_idx)] = {}
            for token_pos, expert_ids in layer_data.items():
                json_oracle[str(layer_idx)][str(token_pos)] = expert_ids

        oracle_data = {
            "model": "Qwen/Qwen1.5-MoE-A2.7B",
            "collection_method": "Inline Oracle (same model instance)",
            "n_experts": self.n_experts,
            "top_k": self.top_k,
            "moe_layers": self.moe_layers,
            "gating_decisions": json_oracle
        }

        with open(output_path, 'w') as f:
            json.dump(oracle_data, f)

        print(f"\n✅ Saved oracle data to {output_path}")
        print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")


def main():
    """Run inline oracle benchmark for all batch sizes."""
    print("="*80)
    print("INLINE ORACLE BENCHMARK - Achieve 100% Efficiency")
    print("="*80)
    print("\nThis script collects oracle data inline during the benchmark run")
    print("to achieve perfect determinism and 100% efficiency.")
    print("="*80)

    # Configuration
    args = Args()
    batch_sizes = [1, 2, 4, 8, 16]
    output_tokens = 20

    results = []

    # Test each batch size
    for batch_size in batch_sizes:
        # Create fresh model instance for each batch size
        print(f"\n\n{'='*80}")
        print(f"Testing Batch Size {batch_size}")
        print(f"{'='*80}")

        model = InlineOracleModel(args)

        # Get prompts
        prompts = get_diverse_batch(batch_size, seed=0)

        # Run inline oracle test
        efficiency = model.run_inline_oracle_test(prompts, output_tokens)

        results.append({
            'batch_size': batch_size,
            'overall': efficiency['overall'],
            'prefill': efficiency['prefill'],
            'decode': efficiency['decode']
        })

        # Clean up
        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n\n{'='*80}")
    print("SUMMARY - INLINE ORACLE EFFICIENCY")
    print(f"{'='*80}\n")

    print(f"{'BS':<4} {'Overall':<12} {'Prefill':<12} {'Decode':<12} {'Status'}")
    print("-" * 55)

    all_pass = True
    for r in results:
        status = "✅ PASS" if r['overall'] >= 99.9 else "❌ FAIL"
        if r['overall'] < 99.9:
            all_pass = False

        print(f"{r['batch_size']:<4} {r['overall']:>10.1f}% {r['prefill']:>10.1f}% {r['decode']:>10.1f}% {status}")

    print(f"\n{'='*80}")

    if all_pass:
        print("✅ SUCCESS: All batch sizes achieve 100% efficiency!")
        print("   Oracle prefetch is working perfectly with inline collection.")
    else:
        print("❌ FAILURE: Some batch sizes don't achieve 100% efficiency")
        print("   This suggests there's still non-determinism in the model.")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
