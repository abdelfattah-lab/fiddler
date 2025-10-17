#!/usr/bin/env python3
"""
FiddlerQwenWithOraclePrefetch - Perfect oracle expert prefetching
Uses pre-collected gating decisions to achieve 100% hit rate (upper bound performance)
"""

import torch
import json
import os

# Import the base prefetch implementation
try:
    from .qwen_with_prefetch import FiddlerQwenWithPrefetch, PrefetchMetrics
except ImportError:
    # If running as script, use absolute import
    from qwen_with_prefetch import FiddlerQwenWithPrefetch, PrefetchMetrics


class FiddlerQwenWithOraclePrefetch(FiddlerQwenWithPrefetch):
    """
    Qwen implementation with oracle (perfect) expert prefetching.
    Uses pre-collected gating decisions to achieve 100% hit rate.
    This represents the upper bound performance achievable with perfect prediction.
    """

    def __init__(
        self,
        args,
        oracle_path="oracle_gating_decisions.json",
        num_experts_to_prefetch=4,  # Default to top-4 since Qwen uses top-4
        enable_cpu_offload=False,
        latency_cpu=0.1,
        latency_gpu=10.0,
        n_gpu_resident_experts=0,
        prompt_key=None  # Key to identify which prompt's decisions to use
    ):
        """
        Initialize model with oracle prefetching.

        Args:
            args: Model arguments (model path, etc.)
            oracle_path: Path to oracle gating decisions JSON file
            num_experts_to_prefetch: Number of experts to prefetch per layer (should match top-k)
            enable_cpu_offload: Whether to use Fiddler CPU offloading
            latency_cpu: CPU expert latency for Fiddler cost model
            latency_gpu: GPU expert latency for Fiddler cost model
            n_gpu_resident_experts: Number of experts to keep on GPU permanently
            prompt_key: Key to identify which prompt's decisions to use (set during generation)
        """
        print("="*80)
        print("INITIALIZING ORACLE PREFETCH MODEL (Perfect Prediction)")
        print("="*80)

        # Store oracle path and load decisions
        self.oracle_path = oracle_path
        self.prompt_key = prompt_key
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        # Load oracle decisions
        print(f"📖 Loading oracle gating decisions from: {oracle_path}")
        self._load_oracle_decisions(oracle_path)

        # Initialize parent class (FiddlerQwenWithPrefetch)
        super().__init__(
            args,
            num_experts_to_prefetch=num_experts_to_prefetch,
            enable_cpu_offload=enable_cpu_offload,
            latency_cpu=latency_cpu,
            latency_gpu=latency_gpu,
            n_gpu_resident_experts=n_gpu_resident_experts
        )

        # Override collection mode - we use oracle, not patterns
        self.collection_mode = False
        self.profiler.collection_mode = False
        self.expert_patterns = {}  # Empty patterns - we use oracle instead

        # Track current token position for decode phase
        self.current_token_pos = 0

        print("✅ Oracle prefetch model initialized")
        print(f"🔮 Using oracle decisions with {num_experts_to_prefetch} experts per layer")
        print(f"📊 Total prompts in oracle: {len(self.oracle_decisions)}")
        print(f"⚠️  This represents UPPER BOUND performance (100% hit rate expected)")
        print("="*80)

    def _load_oracle_decisions(self, oracle_path):
        """Load pre-collected oracle gating decisions."""
        if not os.path.exists(oracle_path):
            raise FileNotFoundError(
                f"Oracle decisions file not found: {oracle_path}\n"
                f"Please run collect_oracle_gating_decisions.py first."
            )

        with open(oracle_path, 'r') as f:
            data = json.load(f)

        self.oracle_decisions = data['gating_decisions']
        self.oracle_moe_layers = data['moe_layers']
        self.oracle_n_experts = data['n_experts']
        self.oracle_top_k = data['top_k']

        print(f"✅ Loaded oracle decisions")
        print(f"   Model: {data['model']}")
        print(f"   MoE layers: {len(self.oracle_moe_layers)}")
        print(f"   Experts per layer: {self.oracle_n_experts}")
        print(f"   Top-k: {self.oracle_top_k}")
        print(f"   Total prompts: {len(self.oracle_decisions)}")

    def set_prompt_keys(self, prompt_keys):
        """
        Set the prompt keys to identify which oracle decisions to use.
        For batched generation, accepts a list of keys and unions their expert decisions.

        Args:
            prompt_keys: Single key (str) or list of keys identifying the prompts
        """
        # Normalize to list
        if isinstance(prompt_keys, str):
            self.prompt_keys = [prompt_keys]
        else:
            self.prompt_keys = prompt_keys

        # Validate all keys exist
        for key in self.prompt_keys:
            if key not in self.oracle_decisions:
                available_keys = list(self.oracle_decisions.keys())[:5]
                raise ValueError(
                    f"Prompt key '{key}' not found in oracle decisions.\n"
                    f"Available keys (first 5): {available_keys}"
                )

        print(f"🔑 Set oracle prompt keys: {self.prompt_keys}")

    def _predict_experts_for_layer(self, layer_idx, token_pos):
        """
        Predict which experts will be needed for this layer using oracle decisions.
        For batched generation, unions experts from all prompt keys.

        Args:
            layer_idx: MoE layer index (2-23 for Qwen)
            token_pos: Token position (for decode phase)

        Returns:
            List of expert indices to prefetch
        """
        # Check if this layer is one we predict for (layers 2-23)
        if layer_idx not in self.moe_layers[2:]:
            return []

        # Check if we have prompt keys set
        if not hasattr(self, 'prompt_keys') or not self.prompt_keys:
            print("⚠️  Warning: No prompt keys set for oracle. Call set_prompt_keys() first.")
            return []

        # Collect expert IDs from all prompt keys and union them
        all_expert_ids = set()

        for prompt_key in self.prompt_keys:
            # Get oracle decisions for this prompt
            prompt_decisions = self.oracle_decisions.get(prompt_key, {})
            layer_decisions = prompt_decisions.get(str(layer_idx), {})

            # Get decisions for this token position
            expert_ids = layer_decisions.get(str(token_pos), [])

            if not expert_ids:
                # Fallback: try previous token position (sometimes token_pos tracking can be off by 1)
                expert_ids = layer_decisions.get(str(token_pos - 1), [])

            # Add to union
            all_expert_ids.update(expert_ids)

        if not all_expert_ids:
            # No oracle data for this position - return empty
            return []

        # Convert to sorted list and limit to num_experts_to_prefetch
        expert_list = sorted(list(all_expert_ids))
        return expert_list[:self.num_experts_to_prefetch]

    def generate(self, text=None, output_token=20, input_token=None, prompt_keys=None):
        """
        Generate with oracle prefetching.

        Args:
            text: Input text (single string or list of strings)
            output_token: Number of tokens to generate
            input_token: Limit on input token count
            prompt_keys: Key(s) to identify which oracle decisions to use (str or list of str)

        Returns:
            Tuple of (prefill_time, decode_time_per_token, prefill_hit_rate, decode_hit_rate)
        """
        # Set prompt keys if provided
        if prompt_keys is not None:
            self.set_prompt_keys(prompt_keys)

        # Reset token position at start of generation
        self.current_token_pos = 0

        # Call parent generate which handles all the generation logic
        return super().generate(text=text, output_token=output_token, input_token=input_token)

    def load_expert_patterns(self):
        """
        Override parent method to skip loading patterns file.
        We use oracle decisions instead of patterns.
        """
        # Return empty dict to indicate we have "patterns" (actually oracle)
        # This prevents collection_mode from being set to True
        return {}

    def save_expert_patterns(self):
        """
        Override parent method to skip saving patterns.
        We use oracle decisions instead of patterns.
        """
        # Don't save patterns - we use oracle decisions
        pass

    def _moe_forward_with_management(self, hidden_states, layer_idx):
        """
        Override to track token position for oracle lookups.
        """
        # Track token position for decode phase
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        if sequence_length == 1:
            # Decode phase - increment token position
            # (this is approximate - actual tracking is done in base class)
            pass  # Token position is handled by base class

        # Call parent implementation
        return super()._moe_forward_with_management(hidden_states, layer_idx)
