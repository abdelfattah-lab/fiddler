#!/usr/bin/env python3
"""
Collect perfect gating decisions (oracle) for benchmark prompts.
This script runs prompts through the model and captures the exact expert selections
made by the gating function at each layer and token position.

The collected data represents the "perfect prediction" - knowing exactly which experts
will be needed before they're actually used.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the same diverse sentences used in benchmark
from benchmark_prediction_methods import DIVERSE_SENTENCES, get_diverse_batch


class GatingCollector:
    """Collects gating decisions from the model."""

    def __init__(self, model_name="Qwen/Qwen1.5-MoE-A2.7B"):
        self.model_name = model_name
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Storage for gating decisions
        self.gating_decisions = {}  # {prompt_idx: {layer_idx: {token_pos: [expert_ids]}}}

        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map="auto",
            use_cache=True
        )

        # Identify MoE layers
        self.moe_layers = []
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                self.moe_layers.append(i)
                if not hasattr(self, 'n_experts'):
                    self.n_experts = len(layer.mlp.experts)
                    self.top_k = layer.mlp.top_k

        print(f"Found {len(self.moe_layers)} MoE layers with {self.n_experts} experts each (top-{self.top_k})")

        # Hook into MoE layers to capture gating decisions
        self._hook_moe_layers()

        # Current generation state
        self.current_prompt_idx = None
        self.current_token_pos = 0

    def _hook_moe_layers(self):
        """Hook into MoE layers to capture gating decisions."""
        print("Hooking into MoE layers to capture gating...")

        for layer_idx in self.moe_layers:
            layer = self.model.model.layers[layer_idx]

            # Save original forward
            layer.mlp.original_forward = layer.mlp.forward

            def create_hooked_forward(layer_idx):
                def hooked_forward(hidden_states):
                    # Capture gating decisions before executing MoE
                    self._capture_gating(hidden_states, layer_idx)
                    # Call original forward
                    return layer.mlp.original_forward(hidden_states)
                return hooked_forward

            # Replace forward method
            layer.mlp.forward = create_hooked_forward(layer_idx)

        print("MoE layers hooked for gating capture")

    def _capture_gating(self, hidden_states, layer_idx):
        """Capture gating decisions for this layer."""
        if self.current_prompt_idx is None:
            return

        moe_layer = self.model.model.layers[layer_idx].mlp
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        # Flatten hidden states for gating
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute router logits (same as in actual forward)
        router_logits = moe_layer.gate(hidden_states_flat)

        # Get routing weights and selected experts (same as in actual forward)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, moe_layer.top_k, dim=-1)

        # Convert to numpy for JSON serialization
        selected_experts_np = selected_experts.cpu().numpy()

        # Store gating decisions
        # selected_experts shape: [batch*seq_len, top_k]
        for batch_idx in range(batch_size):
            for seq_idx in range(sequence_length):
                flat_idx = batch_idx * sequence_length + seq_idx
                expert_ids = selected_experts_np[flat_idx].tolist()

                # Determine absolute token position
                # For prefill: token_pos = seq_idx
                # For decode: token_pos = self.current_token_pos
                if sequence_length > 1:
                    # Prefill phase
                    token_pos = seq_idx
                else:
                    # Decode phase
                    token_pos = self.current_token_pos

                # Store decision
                if self.current_prompt_idx not in self.gating_decisions:
                    self.gating_decisions[self.current_prompt_idx] = {}
                if layer_idx not in self.gating_decisions[self.current_prompt_idx]:
                    self.gating_decisions[self.current_prompt_idx][layer_idx] = {}

                self.gating_decisions[self.current_prompt_idx][layer_idx][token_pos] = expert_ids

    def collect_for_prompt(self, prompt: str, output_tokens: int, prompt_idx: int):
        """
        Collect gating decisions for a single prompt.

        Args:
            prompt: Input prompt text
            output_tokens: Number of tokens to generate
            prompt_idx: Index of this prompt (for storage)
        """
        print(f"\nCollecting gating for prompt {prompt_idx}: '{prompt[:50]}...'")

        # Reset state
        self.current_prompt_idx = prompt_idx
        self.current_token_pos = 0

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device) if inputs.attention_mask is not None else None

        prefill_length = input_ids.shape[1]

        with torch.no_grad():
            # Generate tokens one by one to track token positions properly
            current_input_ids = input_ids
            current_attention_mask = attention_mask

            for token_idx in range(output_tokens):
                # Update token position for decode phase
                if token_idx > 0 or prefill_length == 1:
                    self.current_token_pos = prefill_length + token_idx

                # Generate next token
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    use_cache=True
                )

                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to input
                current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                if current_attention_mask is not None:
                    current_attention_mask = torch.cat([
                        current_attention_mask,
                        torch.ones((1, 1), dtype=current_attention_mask.dtype, device=self.device)
                    ], dim=-1)

                # Clear cache to save memory
                if hasattr(outputs, 'past_key_values'):
                    del outputs.past_key_values

        # Clear cache
        torch.cuda.empty_cache()

        # Verify we captured data
        total_positions = len(self.gating_decisions.get(prompt_idx, {}).get(self.moe_layers[0], {}))
        print(f"  Captured {total_positions} token positions for {len(self.moe_layers)} layers")

        # Reset
        self.current_prompt_idx = None
        self.current_token_pos = 0

    def collect_for_batch(self, batch_size: int, output_tokens: int, seed: int = None):
        """
        Collect gating decisions for a batch of diverse prompts.

        Args:
            batch_size: Number of prompts in the batch
            output_tokens: Number of tokens to generate per prompt
            seed: Random seed for prompt selection
        """
        prompts = get_diverse_batch(batch_size, seed=seed)

        for idx, prompt in enumerate(prompts):
            prompt_idx = f"bs{batch_size}_seed{seed}_prompt{idx}"
            self.collect_for_prompt(prompt, output_tokens, prompt_idx)

    def save(self, output_path: str):
        """Save collected gating decisions to JSON file."""
        # Convert all keys to strings for JSON
        json_data = {}
        for prompt_idx, layers in self.gating_decisions.items():
            json_data[str(prompt_idx)] = {}
            for layer_idx, tokens in layers.items():
                json_data[str(prompt_idx)][str(layer_idx)] = {}
                for token_pos, expert_ids in tokens.items():
                    json_data[str(prompt_idx)][str(layer_idx)][str(token_pos)] = expert_ids

        with open(output_path, 'w') as f:
            json.dump({
                'model': self.model_name,
                'n_experts': self.n_experts,
                'top_k': self.top_k,
                'moe_layers': self.moe_layers,
                'gating_decisions': json_data
            }, f, indent=2)

        print(f"\n✅ Saved gating decisions to: {output_path}")
        print(f"   Total prompts: {len(self.gating_decisions)}")
        print(f"   Total layers: {len(self.moe_layers)}")


def main():
    """Main collection script."""
    print("="*80)
    print("ORACLE GATING DECISION COLLECTION")
    print("="*80)
    print("\nThis script collects perfect gating decisions for benchmark prompts.")
    print("The collected data represents the upper bound performance achievable")
    print("with perfect expert prediction (100% hit rate).")
    print("="*80)

    # Configuration matching benchmark
    batch_sizes = [1, 2, 4, 8, 16]  # Same as benchmark
    output_tokens = 20  # Same as benchmark
    num_trials = 3  # Same as benchmark

    output_path = "oracle_gating_decisions.json"

    # Create collector
    collector = GatingCollector()

    # Collect for all configurations that will be tested in benchmark
    for batch_size in batch_sizes:
        for trial in range(num_trials):
            seed = trial  # Same seed strategy as benchmark
            print(f"\n{'='*80}")
            print(f"Collecting for batch_size={batch_size}, trial={trial}, seed={seed}")
            print(f"{'='*80}")

            collector.collect_for_batch(batch_size, output_tokens, seed=seed)

    # Save results
    collector.save(output_path)

    print("\n" + "="*80)
    print("✅ COLLECTION COMPLETE")
    print("="*80)
    print(f"Output file: {output_path}")
    print(f"Use this file with the oracle prefetch configuration in the benchmark.")
    print("="*80)


if __name__ == "__main__":
    main()
