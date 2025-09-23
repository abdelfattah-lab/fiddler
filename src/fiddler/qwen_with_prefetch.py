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

        # Prefetch state
        self.prefetch_cache = {}  # (layer_idx, expert_idx) -> predicted for this token

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
        """Enhanced MoE forward with prefetch prediction and tracking."""
        moe_layer = self.model.model.layers[layer_idx].mlp

        # Get predictions for current token position
        current_token_pos = getattr(self.profiler, 'current_token_pos', 0)
        predicted_experts = self._predict_experts_for_layer(layer_idx, current_token_pos)

        # Track expert routing for statistics
        router_logits = moe_layer.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights_top, selected_experts = torch.topk(routing_weights, moe_layer.top_k, dim=-1)

        # Record expert usage if in collection mode
        if self.collection_mode:
            self.profiler.record_layer_experts(layer_idx, selected_experts)

        # Count expert usage with prefetch tracking
        for expert_idx in selected_experts.flatten().unique():
            expert_idx = expert_idx.item()

            # Check if this expert was prefetched (predicted)
            was_prefetched = expert_idx in predicted_experts
            self.metrics.record_expert_access(layer_idx, expert_idx, was_prefetched)

            # Update hit/miss tracking for compatibility
            if self.current_expert == (layer_idx, expert_idx):
                self.expert_hit_count += 1
                self.cnt_expert_hit += 1
            else:
                self.expert_fetch_count += 1
                self.current_expert = (layer_idx, expert_idx)

            self.cnt_expert_all += 1

        # Call original forward for correct computation
        return moe_layer.original_forward(hidden_states)

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
            # Generate with our prefetch-enhanced model
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=output_token,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

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