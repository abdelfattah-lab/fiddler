#!/usr/bin/env python3
"""
FiddlerQwenWithPrefetch - Prefetch implementation extending working baseline
Adds prefetch capabilities on top of the working FiddlerQwen
"""

import torch
import torch.nn.functional as F
import json
import os
import time
import copy
from .qwen import FiddlerQwen


class ExpertUsageProfiler:
    """Records and provides expert usage patterns for prediction."""

    def __init__(self, n_layers, n_experts):
        # Use direct O(1) lookup: token_position -> layer_id -> [expert_id1, expert_id2, ...]
        self.expert_patterns = {}  # token_pos -> {layer_id: [expert1, expert2, expert3, expert4]}
        self.current_token_pos = 0
        self.collection_mode = True

    def record_layer_experts(self, layer_id, selected_experts):
        """Record the exact top-k experts for this layer at current token position."""
        # Convert to string keys for JSON compatibility
        token_key = str(self.current_token_pos)
        layer_key = str(layer_id)

        if token_key not in self.expert_patterns:
            self.expert_patterns[token_key] = {}

        # Take top-k experts (Qwen uses top_k=4)
        top_k_experts = selected_experts.flatten().tolist()
        self.expert_patterns[token_key][layer_key] = top_k_experts

    def advance_token_position(self):
        """Move to next token position (called after each complete forward pass)."""
        self.current_token_pos += 1

    def get_experts_for_layer(self, layer_id, token_pos):
        """O(1) lookup: return exact experts needed for this layer at this token position."""
        # Convert to string keys for JSON compatibility
        token_key = str(token_pos)
        layer_key = str(layer_id)

        if token_key in self.expert_patterns and layer_key in self.expert_patterns[token_key]:
            return self.expert_patterns[token_key][layer_key]
        return None  # No recorded pattern available


class PrefetchMetrics:
    """Tracks prefetch hit rates and performance metrics."""

    def __init__(self):
        self.total_expert_requests = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0

        # Detailed tracking by phase
        self.prefill_hits = 0
        self.prefill_misses = 0
        self.decode_hits = 0
        self.decode_misses = 0

        # Track current phase
        self.current_phase = "prefill"

    def set_phase(self, phase):
        """Set current phase (prefill or decode)."""
        self.current_phase = phase

    def record_expert_access(self, layer_id, expert_id, was_prefetched):
        """Record whether an expert access was a hit or miss."""
        self.total_expert_requests += 1

        if was_prefetched:
            self.prefetch_hits += 1
            if self.current_phase == "prefill":
                self.prefill_hits += 1
            else:
                self.decode_hits += 1
        else:
            self.prefetch_misses += 1
            if self.current_phase == "prefill":
                self.prefill_misses += 1
            else:
                self.decode_misses += 1

    def get_hit_rate(self):
        """Get overall prefetch hit rate."""
        if self.total_expert_requests == 0:
            return 0.0
        return self.prefetch_hits / self.total_expert_requests

    def get_decode_hit_rate(self):
        """Get decode-phase prefetch hit rate."""
        total_decode = self.decode_hits + self.decode_misses
        if total_decode == 0:
            return 0.0
        return self.decode_hits / total_decode


class FiddlerQwenWithPrefetch(FiddlerQwen):
    """Qwen implementation with prefetch capabilities."""

    def __init__(self, args):
        # Initialize base class
        super().__init__(args)

        # Initialize prefetch components
        self.profiler = ExpertUsageProfiler(len(self.moe_layers), self.n_expert)
        self.metrics = PrefetchMetrics()

        # Load or initialize expert usage patterns
        self.expert_patterns = self.load_expert_patterns()
        self.collection_mode = (self.expert_patterns is None)

        if not self.collection_mode:
            # Load patterns into profiler
            self.profiler.expert_patterns = self.expert_patterns
            self.profiler.collection_mode = False

        # Prefetch state - track prefetched experts for 2 layers ahead
        self.prefetch_cache = {}  # (layer_idx, expert_idx) -> prefetch buffer
        self.prefetch_buffers = {}  # layer_idx -> {expert_buffer_dict}

        # Initialize prefetch buffers for each layer
        if len(self.moe_layers) > 0 and not self.collection_mode:
            print(f"🔧 Initializing prefetch buffers for {len(self.moe_layers)} MoE layers...")
            sample_layer = self.model.model.layers[self.moe_layers[0]]
            sample_expert = sample_layer.mlp.experts[0]

            for layer_idx in self.moe_layers:
                self.prefetch_buffers[layer_idx] = {}
                # Create 4 prefetch buffers per layer (top-k=4 for Qwen)
                for i in range(4):
                    self.prefetch_buffers[layer_idx][i] = copy.deepcopy(sample_expert).to(self.device, dtype=self.dtype)
            print(f"✅ Prefetch buffers initialized")

        print(f"🔮 Prefetch mode: {'Collection' if self.collection_mode else 'Prediction'}")

    def load_expert_patterns(self):
        """Load expert usage patterns from file."""
        pattern_file = "expert_usage_patterns_qwen.json"
        if os.path.exists(pattern_file):
            with open(pattern_file, 'r') as f:
                return json.load(f)
        return None

    def save_expert_patterns(self):
        """Save expert usage patterns to file."""
        if self.collection_mode:
            with open("expert_usage_patterns_qwen.json", 'w') as f:
                json.dump(self.profiler.expert_patterns, f, indent=2)

    def _predict_experts_for_layer(self, layer_idx, token_pos):
        """Predict which experts will be needed for this layer at given token position."""
        if self.collection_mode:
            return []

        predicted_experts = self.profiler.get_experts_for_layer(layer_idx, token_pos)
        return predicted_experts if predicted_experts else []

    def _moe_forward_with_management(self, hidden_states, layer_idx):
        """Enhanced MoE forward with prefetch prediction and CPU-to-GPU loading (matching baseline logic)."""
        moe_layer = self.model.model.layers[layer_idx].mlp

        # Get dimensions
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Router computation (same as baseline)
        router_logits = moe_layer.gate(hidden_states_flat)

        # Routing weights computation (same as baseline)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, moe_layer.top_k, dim=-1)

        if moe_layer.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Convert routing weights to the correct dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        # Record expert usage patterns for future prefetching
        if self.collection_mode:
            self.profiler.record_layer_experts(layer_idx, selected_experts)

        # Initialize output tensor with correct dtype (same as baseline)
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )

        # One-hot encode selected experts (same as baseline)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=moe_layer.num_experts).permute(2, 1, 0)

        # Find active experts (same as baseline)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # Process each active expert (adapted from baseline)
        for i, expert_idx_tensor in enumerate(expert_hit):
            expert_idx = expert_idx_tensor.item()

            # Find tokens assigned to this expert (same as baseline)
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            if len(top_x) == 0:
                continue

            # Get input for this expert (same as baseline)
            current_state = hidden_states_flat[None, top_x].reshape(-1, hidden_dim)

            # Get routing weights for this expert (same as baseline)
            expert_routing_weights = routing_weights[top_x, idx, None]

            # Check if expert was prefetched, use prefetched version if available
            if self._is_expert_prefetched(layer_idx, expert_idx):
                # Use prefetched expert
                expert_buffer = self._get_expert_for_execution_with_prefetch(layer_idx, expert_idx, was_prefetched=True)
                self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched=True)
                self.cnt_expert_hit += len(top_x)
            else:
                # Load expert on-demand using the base class method
                expert_buffer = self._get_expert_for_execution(layer_idx, expert_idx)
                self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched=False)

            self.cnt_expert_all += len(top_x)

            # Execute expert computation on GPU (same as baseline)
            if current_state.device != expert_buffer.gate_proj.weight.device:
                current_state = current_state.to(expert_buffer.gate_proj.weight.device)

            expert_output = expert_buffer(current_state)

            # Apply routing weights (same as baseline)
            current_hidden_states = expert_output * expert_routing_weights.to(expert_output.device)

            # Move back to original device if needed and accumulate (same as baseline)
            if current_hidden_states.device != final_hidden_states.device:
                current_hidden_states = current_hidden_states.to(final_hidden_states.device)

            # Ensure dtype consistency before accumulation (same as baseline)
            current_hidden_states = current_hidden_states.to(final_hidden_states.dtype)

            final_hidden_states.index_add_(0, top_x, current_hidden_states)

        # Add shared expert (same as baseline)
        shared_expert_output = moe_layer.shared_expert(hidden_states_flat)
        shared_expert_gate = F.sigmoid(moe_layer.shared_expert_gate(hidden_states_flat))
        shared_expert_output = shared_expert_gate * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        # Trigger prefetch for layer+2 if not in collection mode
        if not self.collection_mode and layer_idx + 2 in self.moe_layers:
            self._trigger_prefetch_for_layer(layer_idx + 2, self.profiler.current_token_pos)

        # Advance token position after the last MoE layer
        if layer_idx == self.moe_layers[-1]:
            self.profiler.advance_token_position()

        # Return in the same format as original forward (output, router_logits)
        router_logits = router_logits.view(batch_size, sequence_length, -1)
        return final_hidden_states.view(batch_size, sequence_length, hidden_dim), router_logits

    def _is_expert_prefetched(self, layer_idx, expert_idx):
        """Check if the expert is already loaded in prefetch buffers."""
        cache_key = (layer_idx, expert_idx)
        return cache_key in self.prefetch_cache

    def _get_expert_for_execution_with_prefetch(self, layer_idx, expert_idx, was_prefetched):
        """Get expert for execution, using prefetch buffer if available."""
        cache_key = (layer_idx, expert_idx)

        if was_prefetched and cache_key in self.prefetch_cache:
            # Use prefetched expert from buffer
            return self.prefetch_cache[cache_key]
        else:
            # Load expert on-demand using the base class method
            return self._get_expert_for_execution(layer_idx, expert_idx)

    def _trigger_prefetch_for_layer(self, target_layer_idx, token_pos):
        """Trigger prefetch for experts needed at target layer."""
        if target_layer_idx >= len(self.model.model.layers) or target_layer_idx not in self.moe_layers:
            return

        # Get predicted experts for the target layer
        predicted_experts = self._predict_experts_for_layer(target_layer_idx, token_pos)

        if not predicted_experts:
            return

        # Clear old prefetch cache for this layer
        old_keys = [k for k in self.prefetch_cache.keys() if k[0] == target_layer_idx]
        for key in old_keys:
            del self.prefetch_cache[key]

        # Load predicted experts into prefetch buffers
        for i, expert_idx in enumerate(predicted_experts[:4]):  # Limit to 4 buffers
            if expert_idx is not None and isinstance(expert_idx, int) and 0 <= expert_idx < self.n_expert:
                cache_key = (target_layer_idx, expert_idx)

                # Check if we have buffers for this layer
                if target_layer_idx not in self.prefetch_buffers or i not in self.prefetch_buffers[target_layer_idx]:
                    continue

                # Load CPU expert weights into GPU prefetch buffer with correct dtype
                cpu_expert = self.model.model.layers[target_layer_idx].mlp.experts[expert_idx]
                prefetch_buffer = self.prefetch_buffers[target_layer_idx][i]

                # Ensure all weights are loaded with the correct dtype
                state_dict = cpu_expert.state_dict()
                for key, tensor in state_dict.items():
                    if tensor.dtype != self.dtype:
                        state_dict[key] = tensor.to(self.dtype)

                prefetch_buffer.load_state_dict(state_dict)

                # Cache the prefetched expert
                self.prefetch_cache[cache_key] = prefetch_buffer

    def generate(self, text=None, output_token=20, input_token=None):
        """Generate with prefetch-aware processing and metrics tracking."""
        # Handle text input
        if text is None:
            text = "The capital of France is"

        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device) if inputs.attention_mask is not None else None

        # Limit input tokens if specified
        if input_token is not None:
            input_ids = input_ids[:, :input_token]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :input_token]

        # Reset statistics
        self.expert_fetch_count = 0
        self.expert_hit_count = 0
        self.cnt_expert_hit = 0
        self.cnt_expert_all = 0
        self.current_expert = None

        # Reset prefetch metrics
        self.metrics = PrefetchMetrics()

        # Reset profiler token position for new generation
        if not self.collection_mode:
            self.profiler.current_token_pos = 0

        start_time = time.time()

        with torch.no_grad():
            # Reset cache
            self.model.generation_config.use_cache = True
            if hasattr(self.model, 'past_key_values'):
                self.model.past_key_values = None

            # Generate with the built-in generation method but with cache reset after each token
            # This ensures our hooks are called appropriately
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_token,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

            # Token position is advanced in MoE forward during generation

        total_time = time.time() - start_time
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Calculate hit rates
        baseline_hit_rate = self.cnt_expert_hit / self.cnt_expert_all if self.cnt_expert_all > 0 else 0.0
        prefetch_hit_rate = self.metrics.get_hit_rate()

        # Store for comparison
        self.last_generated_text = generated_text

        # Save patterns if in collection mode
        if self.collection_mode:
            self.save_expert_patterns()

        # For now, approximate prefill vs decode timing
        prefill_time = total_time * 0.3  # Rough approximation
        decode_time = total_time * 0.7

        print(f"Generated: {generated_text}")
        print(f"🎯 Prefetch hit rate: {prefetch_hit_rate:.1%}")

        # Return prefetch hit rate instead of baseline hit rate
        return (prefill_time, decode_time, prefetch_hit_rate)

    def get_prefetch_stats(self):
        """Get detailed prefetch statistics."""
        return {
            'overall_hit_rate': self.metrics.get_hit_rate(),
            'decode_hit_rate': self.metrics.get_decode_hit_rate(),
            'total_requests': self.metrics.total_expert_requests,
            'prefetch_hits': self.metrics.prefetch_hits,
            'prefetch_misses': self.metrics.prefetch_misses,
            'decode_hits': self.metrics.decode_hits,
            'decode_misses': self.metrics.decode_misses
        }